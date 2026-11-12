"""Search correctness checks on random positions (fixed depth, deterministic).

    python -m tools.verify_search --positions 60

1. Alpha-beta == plain minimax: pruning, move ordering, killers and history
   must not change the root value.
2. TT on == TT off, with futility pruning and wall filtering OFF. Two TT modes
   are checked:
     * ``tt_exact_depth 1``: an entry is only reused at exactly the depth it
       was searched to. The TT is then a pure cache and the value MUST match.
     * default (stored depth >= remaining depth): an entry found via a
       transposition may come from a deeper search than the node is entitled
       to, so the value can legitimately differ (it is based on more
       information). The differences are counted and reported, not failed.
"""

from __future__ import annotations

import argparse
import random
import time

from quoridor.bots.cpp_engine import CppEngine

from .positions import random_position

BASE = {"futility": 0, "decided": 1, "wall_filter": 0, "tt": 0, "tt_exact_depth": 0, "alphabeta": 1}


def root_value(engine: CppEngine, position: str, depth: int, **options: int) -> tuple[int, int]:
    for name, value in {**BASE, **options}.items():
        engine.set_option(name, value)  # every "set" also clears the transposition table
    engine.set_position(position)
    _, info = engine.go(depth=depth)
    assert info["depth_reached"] == depth or info["forced"], info
    return info["value"], info["nodes"]


def run(engine: CppEngine, n_positions: int, seed: int, tt_depth: int, mm_depth: int, log=print) -> dict:
    rng = random.Random(seed)
    stats = {"positions": 0, "ab_mismatch": 0, "tt_exact_mismatch": 0, "tt_default_diff": 0,
             "nodes_no_tt": 0, "nodes_tt": 0, "nodes_minimax": 0, "nodes_ab": 0}
    for i in range(n_positions):
        pos = random_position(engine, rng, rng.randint(4, 70))
        stats["positions"] += 1

        # 1. alpha-beta vs minimax (wall filter on to keep the minimax tree small).
        mm, n_mm = root_value(engine, pos, mm_depth, alphabeta=0, wall_filter=1, killers=0, history=0)
        ab, n_ab = root_value(engine, pos, mm_depth, wall_filter=1)
        stats["nodes_minimax"] += n_mm
        stats["nodes_ab"] += n_ab
        if mm != ab:
            stats["ab_mismatch"] += 1
            log(f"  ALPHA-BETA MISMATCH minimax={mm} alphabeta={ab}: {pos}")

        # 2. TT on vs off.
        off, n_off = root_value(engine, pos, tt_depth)
        exact, _ = root_value(engine, pos, tt_depth, tt=1, tt_exact_depth=1)
        default, n_on = root_value(engine, pos, tt_depth, tt=1)
        stats["nodes_no_tt"] += n_off
        stats["nodes_tt"] += n_on
        if exact != off:
            stats["tt_exact_mismatch"] += 1
            log(f"  TT MISMATCH no-tt={off} tt={exact}: {pos}")
        if default != off:
            stats["tt_default_diff"] += 1
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--positions", type=int, default=60)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tt-depth", type=int, default=4, help="depth of the TT on/off comparison")
    p.add_argument("--minimax-depth", type=int, default=3, help="depth of the minimax comparison")
    a = p.parse_args()
    t0 = time.time()
    with CppEngine() as engine:
        s = run(engine, a.positions, a.seed, a.tt_depth, a.minimax_depth)
    n = s["positions"]
    print(f"{n} random positions ({time.time() - t0:.0f}s)")
    print(f"  alpha-beta vs minimax, depth {a.minimax_depth}: {n - s['ab_mismatch']}/{n} identical root values "
          f"({s['nodes_minimax']:,} -> {s['nodes_ab']:,} nodes)")
    print(f"  TT (exact-depth) vs no TT, depth {a.tt_depth}, futility+filter off: "
          f"{n - s['tt_exact_mismatch']}/{n} identical root values")
    print(f"  TT (default, depth >=) vs no TT: {n - s['tt_default_diff']}/{n} identical "
          f"({s['nodes_no_tt']:,} -> {s['nodes_tt']:,} nodes)")
    if s["ab_mismatch"] or s["tt_exact_mismatch"]:
        raise SystemExit("FAILED")
    print("OK")


if __name__ == "__main__":
    main()
