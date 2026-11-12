"""Fast headless Bot vs Bot matches.

    python -m quoridor.botmatch --bot1 ab:depth=2 --bot2 ab:depth=4 --games 100
    python -m quoridor.botmatch --bot1 ml:model=best.pt --bot2 ab:depth=4 --games 200 --json out.json

* Games run in parallel on all CPU cores (``--workers``), with no drawing.
* Bots swap colours every game (``--no-swap`` keeps Bot 1 on Red), because
  moving first is an advantage.
* Every game starts with 0..``--opening-plies`` random moves from a seeded
  generator. The alpha-beta bot is deterministic, so without this every game
  would be identical. ``--opening-plies 0`` plays from the normal start.
* Same ``--seed`` = same openings, so different bot settings can be compared fairly.

Bot spec syntax: see ``quoridor/bots/factory.py`` (``ab:depth=4,time=2``,
``ml:model=best.pt,sims=64``, ...).
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

from .bots.factory import BotSpec, make_policy, parse_bot_spec
from .ml.arena import MAX_PLIES, random_opening, wilson_lower_bound


@dataclass
class GameRecord:
    index: int
    bot1_side: int           # 0 = Red (moves first), 1 = Blue
    winner_bot: Optional[int]  # 1, 2 or None (draw: move limit reached)
    plies: int
    seconds: tuple[float, float]  # total thinking time of bot 1 / bot 2
    moves: tuple[int, int]        # number of moves of bot 1 / bot 2
    walls_used: tuple[int, int] = field(default=(0, 0))


_policies: dict = {}


def _policy(spec: BotSpec, seed: int):
    key = (spec, seed if spec.random_ties else None)
    if key not in _policies:
        _policies[key] = make_policy(spec, seed)
    return _policies[key]


def _init_worker() -> None:
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def play_one(args) -> GameRecord:
    """Play game ``index`` (runs in a worker process)."""
    index, spec1, spec2, seed, opening_plies, swap = args
    rng = random.Random(seed * 1_000_003 + index)
    board = random_opening(rng, opening_plies)
    bot1_side = index % 2 if swap else 0
    specs = {bot1_side: (1, spec1), 1 - bot1_side: (2, spec2)}
    pols = {side: _policy(spec, seed * 7919 + index) for side, (_, spec) in specs.items()}
    secs = [0.0, 0.0]
    nmoves = [0, 0]
    walls = [0, 0]
    winner_side = None
    plies = 0
    for plies in range(1, MAX_PLIES + 1):
        side = board.turn
        who = specs[side][0] - 1
        t0 = time.perf_counter()
        move = pols[side](board.copy())
        secs[who] += time.perf_counter() - t0
        nmoves[who] += 1
        walls[who] += move[0] != "p"
        winner_side = board.play(move)
        if winner_side is not None:
            break
    winner_bot = None if winner_side is None else specs[winner_side][0]
    return GameRecord(index, bot1_side, winner_bot, plies, (secs[0], secs[1]), (nmoves[0], nmoves[1]),
                      (walls[0], walls[1]))


def summarize(records: list[GameRecord], spec1: BotSpec, spec2: BotSpec) -> dict:
    n = len(records)
    w1 = sum(r.winner_bot == 1 for r in records)
    w2 = sum(r.winner_bot == 2 for r in records)

    def side_rate(bot: int, side: int) -> Optional[float]:
        games = [r for r in records if (r.bot1_side == side) == (bot == 1)]
        return sum(r.winner_bot == bot for r in games) / len(games) if games else None

    def per_bot(bot: int) -> dict:
        i = bot - 1
        moves = sum(r.moves[i] for r in records)
        return {
            "spec": (spec1 if bot == 1 else spec2).to_string(),
            "wins": w1 if bot == 1 else w2,
            "win_rate": (w1 if bot == 1 else w2) / n if n else 0.0,
            "win_rate_ci_low": wilson_lower_bound(w1 if bot == 1 else w2, n),
            "win_rate_as_red": side_rate(bot, 0),
            "win_rate_as_blue": side_rate(bot, 1),
            "avg_seconds_per_move": sum(r.seconds[i] for r in records) / moves if moves else 0.0,
            "avg_walls_per_game": sum(r.walls_used[i] for r in records) / n if n else 0.0,
        }

    return {"games": n, "draws": n - w1 - w2, "avg_plies": sum(r.plies for r in records) / n if n else 0,
            "bot1": per_bot(1), "bot2": per_bot(2)}


def run_match(spec1: BotSpec, spec2: BotSpec, games: int, seed: int = 0, opening_plies: int = 6,
              swap: bool = True, workers: int = 0, progress: bool = True) -> tuple[dict, list[GameRecord]]:
    jobs = [(i, spec1, spec2, seed, opening_plies, swap) for i in range(games)]
    workers = min(workers or (os.cpu_count() or 1), max(1, games))
    records: list[GameRecord] = []
    t0 = time.time()
    if workers == 1:
        _init_worker()
        it = map(play_one, jobs)
        pool = None
    else:
        ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
        pool = ctx.Pool(workers, initializer=_init_worker)
        it = pool.imap_unordered(play_one, jobs)
    try:
        for rec in it:
            records.append(rec)
            if progress:
                w1 = sum(r.winner_bot == 1 for r in records)
                w2 = sum(r.winner_bot == 2 for r in records)
                print(f"\r  {len(records)}/{games} games   Bot 1 {w1} - {w2} Bot 2   "
                      f"({time.time() - t0:.0f}s)", end="", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    if progress:
        print()
    records.sort(key=lambda r: r.index)
    summary = summarize(records, spec1, spec2)
    summary.update(seed=seed, opening_plies=opening_plies, swap=swap, seconds=round(time.time() - t0, 1))
    return summary, records


def format_summary(s: dict) -> str:
    def pct(x: Optional[float]) -> str:
        return "  -  " if x is None else f"{x:5.1%}"

    rows = [f"{'':8}{'spec':34}{'wins':>6}{'win %':>8}{'CI low':>8}{'as Red':>8}{'as Blue':>9}"
            f"{'s/move':>8}{'walls':>7}"]
    for b in ("bot1", "bot2"):
        d = s[b]
        spec = d["spec"]
        if "model=" in spec:  # show only the checkpoint's file name
            head, _, rest = spec.partition("model=")
            path, sep, tail = rest.partition(",")
            spec = head + "model=" + os.path.basename(path) + sep + tail
        rows.append(f"{'Bot ' + b[-1]:8}{spec[:33]:34}{d['wins']:6d}{pct(d['win_rate']):>8}"
                    f"{pct(d['win_rate_ci_low']):>8}{pct(d['win_rate_as_red']):>8}{pct(d['win_rate_as_blue']):>9}"
                    f"{d['avg_seconds_per_move']:8.3f}{d['avg_walls_per_game']:7.1f}")
    rows.append(f"{s['games']} games, {s['draws']} draws, {s['avg_plies']:.0f} moves per game on average, "
                f"{s['seconds']}s total")
    return "\n".join(rows)


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bot1", default="ab:depth=2", help="spec of Bot 1 (default ab:depth=2)")
    p.add_argument("--bot2", default="ab:depth=4", help="spec of Bot 2 (default ab:depth=4)")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed", type=int, default=0, help="opening seed (same seed = same openings)")
    p.add_argument("--opening-plies", type=int, default=6, help="max random opening moves (0 = none)")
    p.add_argument("--no-swap", action="store_true", help="Bot 1 always plays Red instead of alternating")
    p.add_argument("--workers", type=int, default=0, help="parallel processes (0 = all CPU cores)")
    p.add_argument("--json", help="write the summary + every game to this JSON file")
    p.add_argument("--csv", help="write one row per game to this CSV file")
    a = p.parse_args(argv)
    try:
        spec1, spec2 = parse_bot_spec(a.bot1), parse_bot_spec(a.bot2)
    except ValueError as exc:
        p.error(str(exc))

    print(f"Bot 1: {spec1.to_string()}   vs   Bot 2: {spec2.to_string()}   ({a.games} games)")
    summary, records = run_match(spec1, spec2, a.games, a.seed, a.opening_plies, not a.no_swap, a.workers)
    print(format_summary(summary))

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"summary": summary, "games": [vars(r) for r in records]}, f, indent=1)
        print(f"wrote {a.json}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["game", "bot1_color", "winner", "plies", "bot1_seconds", "bot2_seconds",
                        "bot1_walls", "bot2_walls"])
            for r in records:
                w.writerow([r.index, "red" if r.bot1_side == 0 else "blue",
                            "draw" if r.winner_bot is None else f"bot{r.winner_bot}", r.plies,
                            f"{r.seconds[0]:.3f}", f"{r.seconds[1]:.3f}", *r.walls_used])
        print(f"wrote {a.csv}")
    return summary


if __name__ == "__main__":
    main()
