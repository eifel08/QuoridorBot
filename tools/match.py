"""Strength test: two C++ engine configurations play a match.

    python -m tools.match --a futility=1 --b futility=0 --time 0.1 --games 200
    python -m tools.match --a wall_filter=1 --b wall_filter=0 --depth 4 --games 200
    python -m tools.match --a wall_weight=8 --b wall_weight=0 --time 0.05

``--a`` / ``--b`` are comma-separated engine options (see engine/README.md);
``depth=N`` and ``time=SECONDS`` inside them override ``--depth`` / ``--time``
for that side only, e.g. ``--a time=0.2 --b time=0.1``.

* Every opening (a few seeded random moves) is played twice with the colours
  swapped, so neither side profits from an easier opening or from moving first.
* Each side has its own engine process, so nothing is shared between them. The
  transposition table is kept during a game and cleared between games.
* Games longer than ``--max-plies`` are draws (the engines do not detect repetitions).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import sys
import time
from typing import Optional

from quoridor.bots.cpp_engine import DEFAULT_OPTIONS, CppEngine, to_imove
from quoridor.ml.arena import random_opening, wilson_lower_bound


def parse_config(text: str, depth: Optional[int], time_limit: Optional[float]) -> dict:
    cfg: dict = {"depth": depth, "time": time_limit, "options": {}}
    for item in filter(None, (t.strip() for t in text.split(","))):
        key, _, value = item.partition("=")
        key = {"filter": "wall_filter"}.get(key, key)
        if key == "depth":
            cfg["depth"] = int(value)
        elif key == "time":
            cfg["time"] = float(value)
        elif key in DEFAULT_OPTIONS:
            cfg["options"][key] = int(value)
        else:
            raise SystemExit(f"unknown engine option '{key}' (known: depth, time, {', '.join(DEFAULT_OPTIONS)})")
    if not cfg["depth"] and not cfg["time"]:
        raise SystemExit("give a depth and/or a time per move")
    return cfg


_engines: dict = {}


def _engine(slot: int, cfg: dict) -> CppEngine:
    key = (os.getpid(), slot)
    if key not in _engines:
        _engines[key] = CppEngine(options={"hash_mb": 32, **cfg["options"]})
    return _engines[key]


def play_game(job) -> dict:
    """One game; returns who won ("a", "b" or None) and search statistics."""
    index, cfg_a, cfg_b, seed, opening_plies, max_plies = job
    rng = random.Random(seed * 1_000_003 + index // 2)  # games 2k and 2k+1 share the opening
    board = random_opening(rng, opening_plies)
    a_side = index % 2
    sides = {a_side: ("a", cfg_a, _engine(0, cfg_a)), 1 - a_side: ("b", cfg_b, _engine(1, cfg_b))}
    for _, _, engine in sides.values():
        engine.new_game()
    stats = {"a": [0, 0, 0.0, 0], "b": [0, 0, 0.0, 0]}  # moves, depth sum, seconds, nodes
    winner = None
    plies = 0
    for plies in range(1, max_plies + 1):
        who, cfg, engine = sides[board.turn]
        engine.set_position(board)
        move, info = engine.go(cfg["depth"], cfg["time"])
        if move is None:  # no legal move: counted as a loss for the side to move
            winner = "b" if who == "a" else "a"
            break
        st = stats[who]
        st[0] += 1
        st[1] += info["depth_reached"]
        st[2] += info["seconds"]
        st[3] += info["nodes"]
        if board.play(to_imove(move)) is not None:
            winner = who
            break
    return {"index": index, "a_side": a_side, "winner": winner, "plies": plies, "stats": stats}


def run_match(cfg_a: dict, cfg_b: dict, games: int, seed: int = 0, opening_plies: int = 6,
              max_plies: int = 200, workers: int = 0, progress: Optional[bool] = None) -> dict:
    if progress is None:
        progress = sys.stdout.isatty()
    jobs = [(i, cfg_a, cfg_b, seed, opening_plies, max_plies) for i in range(games)]
    workers = min(workers or (os.cpu_count() or 1), max(1, games))
    results = []
    t0 = time.time()
    if workers == 1:
        it, pool = map(play_game, jobs), None
    else:
        pool = mp.get_context("fork" if sys.platform != "win32" else "spawn").Pool(workers)
        it = pool.imap_unordered(play_game, jobs)
    try:
        for r in it:
            results.append(r)
            if progress:
                wa = sum(x["winner"] == "a" for x in results)
                wb = sum(x["winner"] == "b" for x in results)
                print(f"\r  {len(results)}/{games}   A {wa} - {wb} B   ({time.time() - t0:.0f}s)", end="", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    if progress:
        print()

    n = len(results)
    wins = {s: sum(r["winner"] == s for r in results) for s in "ab"}
    draws = n - wins["a"] - wins["b"]

    def side_stats(s: str) -> dict:
        moves = sum(r["stats"][s][0] for r in results) or 1
        first = [r for r in results if (r["a_side"] == 0) == (s == "a")]
        return {
            "wins": wins[s],
            "score": (wins[s] + 0.5 * draws) / n if n else 0.0,
            "win_rate": wins[s] / n if n else 0.0,
            "ci_low": wilson_lower_bound(wins[s], n),
            "wins_moving_first": sum(r["winner"] == s for r in first),
            "games_moving_first": len(first),
            "avg_depth": sum(r["stats"][s][1] for r in results) / moves,
            "avg_seconds": sum(r["stats"][s][2] for r in results) / moves,
            "avg_nodes": sum(r["stats"][s][3] for r in results) / moves,
        }

    return {"games": n, "draws": draws, "avg_plies": sum(r["plies"] for r in results) / n if n else 0,
            "a": side_stats("a"), "b": side_stats("b"), "seconds": time.time() - t0}


def describe(cfg: dict) -> str:
    parts = [f"{k}={v}" for k, v in cfg["options"].items()]
    if cfg["depth"]:
        parts.append(f"depth={cfg['depth']}")
    if cfg["time"]:
        parts.append(f"time={cfg['time']:g}")
    return ",".join(parts)


def format_result(res: dict, cfg_a: dict, cfg_b: dict) -> str:
    lines = [f"{'':3}{'config':48}{'wins':>6}{'win %':>8}{'CI low':>8}{'1st':>9}{'depth':>7}{'nodes/mv':>11}{'s/move':>8}"]
    for s, cfg in (("a", cfg_a), ("b", cfg_b)):
        d = res[s]
        lines.append(f"{s.upper():3}{describe(cfg)[:47]:48}{d['wins']:6d}{d['win_rate']:8.1%}{d['ci_low']:8.1%}"
                     f"{d['wins_moving_first']:5d}/{d['games_moving_first']:<3d}{d['avg_depth']:7.1f}"
                     f"{d['avg_nodes']:11,.0f}{d['avg_seconds']:8.3f}")
    lines.append(f"{res['games']} games, {res['draws']} draws, {res['avg_plies']:.0f} plies per game, "
                 f"{res['seconds']:.0f}s")
    return "\n".join(lines)


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", default="", help="engine options of side A, e.g. futility=1")
    p.add_argument("--b", default="", help="engine options of side B, e.g. futility=0")
    p.add_argument("--depth", type=int, help="fixed search depth for both sides")
    p.add_argument("--time", type=float, help="seconds per move for both sides")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--opening-plies", type=int, default=6, help="max random opening moves")
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--workers", type=int, default=0, help="parallel games (0 = all CPU cores)")
    a = p.parse_args(argv)
    cfg_a, cfg_b = parse_config(a.a, a.depth, a.time), parse_config(a.b, a.depth, a.time)
    print(f"A: {describe(cfg_a)}   vs   B: {describe(cfg_b)}")
    res = run_match(cfg_a, cfg_b, a.games, a.seed, a.opening_plies, a.max_plies, a.workers)
    print(format_result(res, cfg_a, cfg_b))
    return res


if __name__ == "__main__":
    main()
