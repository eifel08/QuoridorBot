"""Benchmark: old Python search vs the C++ engine on fixed positions.

    python -m tools.bench --seconds 2

For each position (opening, middlegame, endgame) it reports

* nodes per second and the depth reached in ``--seconds`` seconds, for the
  Python bot (quoridor/bots/alphabeta.py), the C++ engine with its default
  settings, and the C++ engine with wall filtering and futility pruning off;
* the time both engines need to finish the same fixed depth (``--depth``).

Depths are not directly comparable between the two engines: the Python bot
only considers walls that cut ONE shortest path of the opponent AND improve
the path difference (5-20 moves per node), while the C++ engine searches the
larger attack + defense wall set (or every wall with the filter off).
"""

from __future__ import annotations

import argparse
import time

from quoridor import Board, BoardState, Orientation
from quoridor.bots.alphabeta import search
from quoridor.bots.cpp_engine import CppEngine

from .positions import BENCH_POSITIONS

CPP_CONFIGS = {
    "C++ (default)": {},
    "C++ (filter+futility off)": {"wall_filter": 0, "futility": 0},
}


def to_state(position: str) -> BoardState:
    tok = position.split()[1:]
    turn, r0, c0, r1, c1, w0, w1 = map(int, tok[:7])
    board = Board()
    board.pawns = [(r0, c0), (r1, c1)]
    board.walls_left = [w0, w1]
    board.walls = {(int(t[1]), int(t[2]), Orientation(t[0])) for t in tok[7:]}
    return board.snapshot(turn)


def python_search(position: str, depth: int, seconds: float | None) -> dict:
    t0 = time.perf_counter()
    _, info = search(to_state(position), depth, seconds)
    dt = time.perf_counter() - t0
    return {"depth": info["depth_reached"], "nodes": info["nodes"], "seconds": dt,
            "nps": info["nodes"] / dt if dt > 0 else 0, "forced": info["forced"]}


def cpp_search(engine: CppEngine, position: str, depth: int | None, seconds: float | None) -> dict:
    engine.new_game()
    engine.set_position(position)
    _, info = engine.go(depth, seconds)
    return {"depth": info["depth_reached"], "nodes": info["nodes"], "seconds": info["seconds"],
            "nps": info["nps"], "forced": info["forced"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=2.0, help="time budget per position")
    p.add_argument("--depth", type=int, default=5, help="fixed depth for the time-to-depth comparison")
    a = p.parse_args()

    engines = {name: CppEngine(options=opts) for name, opts in CPP_CONFIGS.items()}
    try:
        print(f"== {a.seconds:g} s per position ==")
        print(f"{'position':12}{'engine':28}{'depth':>6}{'nodes':>14}{'nodes/s':>13}")
        speed: dict[str, list[float]] = {}
        for pos_name, pos in BENCH_POSITIONS.items():
            rows = [("Python (old bot)", python_search(pos, 30, a.seconds))]
            rows += [(name, cpp_search(e, pos, None, a.seconds)) for name, e in engines.items()]
            for name, r in rows:
                note = f"  (forced {r['forced']} found)" if r["forced"] else ""
                print(f"{pos_name:12}{name:28}{r['depth']:6d}{r['nodes']:14,d}{r['nps']:13,.0f}{note}")
                speed.setdefault(name, []).append(r["nps"])
        py = sum(speed["Python (old bot)"]) / len(speed["Python (old bot)"])
        for name in engines:
            avg = sum(speed[name]) / len(speed[name])
            print(f"average nodes/s: {name} {avg:,.0f} = {avg / py:,.0f}x Python ({py:,.0f})")

        print(f"\n== time to finish depth {a.depth} ==")
        print(f"{'position':12}{'engine':28}{'seconds':>10}{'nodes':>14}")
        for pos_name, pos in BENCH_POSITIONS.items():
            rows = [("Python (old bot)", python_search(pos, a.depth, None))]
            rows += [(name, cpp_search(e, pos, a.depth, None)) for name, e in engines.items()]
            for name, r in rows:
                print(f"{pos_name:12}{name:28}{r['seconds']:10.4f}{r['nodes']:14,d}")
    finally:
        for e in engines.values():
            e.close()


if __name__ == "__main__":
    main()
