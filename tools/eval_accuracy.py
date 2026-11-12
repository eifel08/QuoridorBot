"""Eval accuracy of engine settings on ONE fixed set of labelled positions.

    python -m tools.eval_accuracy --positions 500 --out positions.json        # collect + label once
    python -m tools.eval_accuracy --load positions.json --set wall_weight=16 --set wall_weight=6,wall_cap=1

Unlike ``tools.calibration`` (where every setting plays its own games), every
setting here is judged on the same positions, so the numbers are directly
comparable.

1. Positions are sampled from self-play games of the default engine.
2. Each position is labelled with its winner by playing it out with a
   reference bot (``--label-time`` per move, same settings for both sides).
3. Each ``--set`` searches every position for ``--time`` seconds; its score
   (side to move's view) is compared with the label:
     * sign accuracy: eval > 0 <=> the side to move won;
     * false win claims: eval >= +1 step but the side to move lost;
     * race positions: the same, restricted to positions where a player is
       <= 3 steps from its goal (where surplus walls stop mattering).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time

from quoridor.bots.cpp_engine import DEFAULT_OPTIONS, CppEngine, fastboard_command, to_imove
from quoridor.ml.arena import random_opening

from .calibration import parse_options

_engines: dict = {}


def _engine(options: dict) -> CppEngine:
    key = (os.getpid(), tuple(sorted(options.items())))
    if key not in _engines:
        _engines[key] = CppEngine(options={"hash_mb": 16, **options})
    return _engines[key]


def _collect(job) -> list[dict]:
    """Positions of one self-play game (every 3rd ply), with distances and walls."""
    index, seed, time_limit = job
    engine = _engine({})
    engine.new_game()
    board = random_opening(random.Random(seed * 7_919 + index), 6)
    out = []
    for ply in range(200):
        engine.set_position(board)
        if ply % 3 == index % 3:
            d = engine.distances()
            out.append({"pos": fastboard_command(board), "turn": board.turn, "dist": list(d),
                        "walls": list(board.walls_left)})
        move, _ = engine.go(None, time_limit)
        if move is None or board.play(to_imove(move)) is not None:
            break
    return out


def _label(job) -> int:
    """Winner of a position when the reference bot plays both sides (-1: no result)."""
    pos, time_limit = job
    from quoridor.bots.alphabeta import FastBoard  # noqa: F401  (board from the position command)

    engine = _engine({})
    engine.new_game()
    board = _board_from(pos)
    for _ in range(200):
        engine.set_position(board)
        move, _ = engine.go(None, time_limit)
        if move is None:
            return 1 - board.turn
        mover = board.turn
        if board.play(to_imove(move)) is not None:
            return mover
    return -1


def _board_from(pos: str):
    from quoridor import Board, Orientation
    from quoridor.bots.alphabeta import FastBoard

    tok = pos.split()[1:]
    turn, r0, c0, r1, c1, w0, w1 = map(int, tok[:7])
    b = Board()
    b.pawns = [(r0, c0), (r1, c1)]
    b.walls_left = [w0, w1]
    b.walls = {(int(t[1]), int(t[2]), Orientation(t[0])) for t in tok[7:]}
    return FastBoard(b.snapshot(turn))


def _evaluate(job) -> float | None:
    """Search score in steps for the side to move (+-100 for forced results)."""
    pos, options, time_limit = job
    engine = _engine(options)
    engine.new_game()
    engine.command(pos)
    _, info = engine.go(None, time_limit)
    if info["forced"]:
        return 100.0 if info["forced"] == "win" else -100.0
    return info["value"] / options.get("dist_weight", DEFAULT_OPTIONS["dist_weight"])


def _pool():
    return mp.get_context("fork" if sys.platform != "win32" else "spawn").Pool(os.cpu_count() or 1)


def collect_and_label(n_positions: int, seed: int, play_time: float, label_time: float) -> list[dict]:
    with _pool() as pool:
        positions: list[dict] = []
        games = 0
        while len(positions) < n_positions * 2:
            batch = pool.map(_collect, [(games + i, seed, play_time) for i in range(16)])
            games += 16
            positions += [p for game in batch for p in game]
        rng = random.Random(seed)
        rng.shuffle(positions)
        positions = positions[:n_positions]
        labels = pool.map(_label, [(p["pos"], label_time) for p in positions])
    for p, w in zip(positions, labels):
        p["winner"] = w
    return [p for p in positions if p["winner"] >= 0]


def score(positions: list[dict], options: dict, time_limit: float) -> dict:
    with _pool() as pool:
        evals = pool.map(_evaluate, [(p["pos"], options, time_limit) for p in positions])
    s = {k: 0 for k in ("n", "ok", "claims", "false", "race_n", "race_ok", "race_claims", "race_false")}
    for p, ev in zip(positions, evals):
        me = p["turn"]
        won = p["winner"] == me
        race = min(p["dist"]) <= 3
        if ev is None or ev == 0:
            continue
        s["n"] += 1
        s["ok"] += (ev > 0) == won
        if race:
            s["race_n"] += 1
            s["race_ok"] += (ev > 0) == won
        if ev >= 1:
            s["claims"] += 1
            s["false"] += not won
            if race:
                s["race_claims"] += 1
                s["race_false"] += not won
    return s


def fmt(s: dict) -> str:
    def pct(a, b):
        return f"{a / b:6.1%} ({a}/{b})" if b else "  -   (0/0)"
    return (f"sign accuracy {pct(s['ok'], s['n'])}   false win claims {pct(s['false'], s['claims'])}   "
            f"| races: accuracy {pct(s['race_ok'], s['race_n'])}, false claims {pct(s['race_false'], s['race_claims'])}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--positions", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="save the labelled positions here")
    p.add_argument("--load", help="use labelled positions saved with --out")
    p.add_argument("--set", action="append", default=[], help="engine options to score (repeatable)")
    p.add_argument("--time", type=float, default=0.05, help="search time per position when scoring")
    p.add_argument("--label-time", type=float, default=0.1, help="time per move when playing out labels")
    a = p.parse_args()
    t0 = time.time()
    if a.load:
        with open(a.load) as f:
            positions = json.load(f)
    else:
        positions = collect_and_label(a.positions, a.seed, 0.05, a.label_time)
        print(f"{len(positions)} labelled positions ({time.time() - t0:.0f}s)")
        if a.out:
            with open(a.out, "w") as f:
                json.dump(positions, f)
    for text in a.set or [""]:
        s = score(positions, parse_options(text), a.time)
        print(f"{text or 'defaults':28} {fmt(s)}")


if __name__ == "__main__":
    main()
