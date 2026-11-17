"""Generate training positions by engine self-play (for tools.texel_tune and tools.train_nnue).

    python -m tools.selfplay_data --out data/selfplay.jsonl --games 4000 --time 0.05
    python -m tools.selfplay_data --out data/selfplay_gen2.jsonl --profile nnue,hybrid --time 0.1

Each game starts from a random opening (0-6 plies) and both sides play the given
profile (a comma-separated list: each side of each game picks one at random, so
generation 2 mixes nnue-vs-nnue, hybrid-vs-hybrid and nnue-vs-hybrid games; the
opening book is not used, so every position has a real search score); with probability ``--explore`` a move is replaced by a random move among
the first ``--explore-top`` moves of the engine's own search order (plies < 40),
for diversity. Every position records: the position command, the search score of
the side to move (engine units), search depth, the linear features, and the final
winner. One JSON line per game.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time

from quoridor.bots.cpp_engine import WIN_THRESHOLD, fastboard_command
from quoridor.bots.profiles import ProfileEngine, imove
from quoridor.ml.arena import random_opening

_eng = {}


def _engine(profile: str) -> ProfileEngine:
    if profile not in _eng:
        _eng[profile] = ProfileEngine(profile)
    return _eng[profile]


def play(job) -> dict:
    gid, profiles, tl, explore, top, seed = job
    rng = random.Random(seed * 1_000_003 + gid)
    sides = [rng.choice(profiles), rng.choice(profiles)]
    for name in set(sides):
        _engine(name).new_game()
    board = random_opening(rng, 6, wall_prob=0.2)
    rows = []
    winner = None
    for ply in range(200):
        eng = _engine(sides[board.turn]).engine
        pos = fastboard_command(board)
        eng.set_position(pos)
        feats = list(map(int, eng.command("features").split()[1:]))
        move, info = eng.go(64, tl)
        if move is None:
            winner = 1 - board.turn
            break
        text = info["move"]
        rows.append([pos, board.turn, info["value"], info["depth_reached"], feats])
        if ply < 40 and rng.random() < explore and abs(info["value"]) < WIN_THRESHOLD:
            order = eng.command("genmoves").split()[1:1 + top]
            if order:
                text = rng.choice(order)
        side = board.turn
        if board.play(imove(text)) is not None:
            winner = side
            break
    return {"winner": winner, "rows": rows, "profiles": sides}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--profile", default="turbo", help="profile, or a comma-separated list chosen per side")
    p.add_argument("--games", type=int, default=2000)
    p.add_argument("--time", type=float, default=0.05)
    p.add_argument("--explore", type=float, default=0.08)
    p.add_argument("--explore-top", type=int, default=6)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)
    profiles = [x.strip() for x in a.profile.split(",") if x.strip()]
    jobs = [(g, profiles, a.time, a.explore, a.explore_top, a.seed) for g in range(a.games)]
    t0 = time.time()
    n = 0
    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    with ctx.Pool(a.workers) as pool, open(a.out, "a") as f:
        for k, r in enumerate(pool.imap_unordered(play, jobs), 1):
            f.write(json.dumps(r) + "\n")
            n += len(r["rows"])
            if k % 50 == 0:
                print(f"{k}/{a.games} games, {n} positions, {time.time() - t0:.0f}s", flush=True)
    print("done", n, "positions", flush=True)


if __name__ == "__main__":
    main()
