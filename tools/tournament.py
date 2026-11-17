"""Round-robin tournament between bot profiles (quoridor/bots/profiles.py).

    python -m tools.tournament --out results/rr_5s --time 5 --openings 8
    python -m tools.tournament --out results/rr_05s --time 0.5 --openings 50
    python -m tools.tournament_report results/rr_5s

* Every pair of bots plays every opening twice, colours swapped.
* Openings: the start position plus distinct short random openings (1-3 plies, mostly pawn
  steps), the same set for every pair.
* Every bot runs single-threaded in its own engine process; ``--workers`` games run in parallel.
* Time: each move is searched with ``--time`` seconds. A move whose wall-clock time exceeds
  ``--time * 1.1 + 0.05`` is a time violation and loses the game.
* Games longer than ``--max-plies`` are draws.
* Results are appended to ``OUT/games.jsonl`` as games finish; rerunning the same command skips
  the games already played, so an interrupted tournament can be resumed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

from quoridor.bots.alphabeta import FastBoard
from quoridor.bots.profiles import PROFILES, ProfileEngine, imove
from quoridor.ml.arena import random_opening


def make_openings(count: int, seed: int) -> list[list]:
    """``count`` distinct openings as move lists (the first one is the start position)."""
    rng = random.Random(seed)
    seen, out = set(), [[]]
    b0 = FastBoard.initial()
    seen.add((tuple(b0.pawns), tuple(b0.walls_left), (), (), b0.turn))
    tries = 0
    while len(out) < count and tries < 100000:
        tries += 1
        n = rng.randint(1, 3)
        board = FastBoard.initial()
        moves = []
        for _ in range(n):
            me = board.turn
            walls = board.legal_walls(me) if rng.random() < 0.12 else []
            choice = rng.choice(walls) if walls else rng.choice([("p", sq) for sq in board.pawn_moves(me)])
            moves.append(list(choice))
            board.play(choice)
        key = (tuple(board.pawns), tuple(board.walls_left), tuple(sorted(board.h)), tuple(sorted(board.v)), board.turn)
        if key in seen:
            continue
        seen.add(key)
        out.append(moves)
    return out


_engines: dict = {}


def _engine(name: str) -> ProfileEngine:
    if name not in _engines:
        _engines[name] = ProfileEngine(name, seed=os.getpid() * 7919 + len(_engines) + 1)
    return _engines[name]


def play_game(job: dict) -> dict:
    red, blue = job["red"], job["blue"]  # red = player 0 (moves first)
    board = FastBoard.initial()
    for m in job["opening"]:
        board.play(tuple(m))
    engines = {0: _engine(red), 1: _engine(blue)}
    for e in engines.values():
        e.new_game()
    names = {0: red, 1: blue}
    stats = {red: {"moves": 0, "book": 0, "depth": 0, "nodes": 0, "seconds": 0.0, "max_seconds": 0.0},
             blue: {"moves": 0, "book": 0, "depth": 0, "nodes": 0, "seconds": 0.0, "max_seconds": 0.0}}
    if red == blue:
        raise ValueError("a bot cannot play itself here")
    limit = job["time"] * 1.1 + 0.05
    winner, reason, moves = None, "move cap", []
    plies = len(job["opening"])
    while plies < job["max_plies"]:
        side = board.turn
        t0 = time.perf_counter()
        text, info = engines[side].go(board, job["time"])
        wall = time.perf_counter() - t0
        st = stats[names[side]]
        if text is None:
            winner, reason = 1 - side, "no legal move"
            break
        st["moves"] += 1
        st["book"] += int(info.get("book", False))
        if not info.get("book"):
            st["depth"] += info["depth_reached"]
            st["nodes"] += info["nodes"]
            st["seconds"] += info["seconds"]
        st["max_seconds"] = max(st["max_seconds"], wall)
        moves.append([text, info.get("value", 0), info.get("depth_reached", 0), round(wall, 3)])
        if wall > limit:
            winner, reason = 1 - side, f"time violation {wall:.2f}s"
            break
        plies += 1
        if board.play(imove(text)) is not None:
            winner, reason = side, "goal"
            break
    return {"id": job["id"], "red": red, "blue": blue, "opening": job["oi"],
            "winner": None if winner is None else names[winner], "reason": reason,
            "plies": plies, "walls_left": list(board.walls_left), "stats": stats, "moves": moves}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--bots", default=",".join(PROFILES), help="comma-separated profile names")
    p.add_argument("--time", type=float, default=5.0, help="seconds per move")
    p.add_argument("--openings", type=int, default=8, help="openings per pair (each played twice)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)

    bots = [b.strip() for b in args.bots.split(",") if b.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    openings = make_openings(args.openings, args.seed)
    (out / "config.json").write_text(json.dumps({"bots": bots, "time": args.time, "openings": openings,
                                                 "max_plies": args.max_plies, "seed": args.seed}, indent=1))
    jobs = []
    for a, b in itertools.combinations(bots, 2):
        for oi, op in enumerate(openings):
            for red, blue in ((a, b), (b, a)):
                jobs.append({"id": f"{red}|{blue}|{oi}", "red": red, "blue": blue, "oi": oi, "opening": op,
                             "time": args.time, "max_plies": args.max_plies})
    done_path = out / "games.jsonl"
    done = set()
    if done_path.is_file():
        for line in done_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = [j for j in jobs if j["id"] not in done]
    # Interleave pairs so partial results cover every pair.
    random.Random(1).shuffle(todo)
    print(f"{len(jobs)} games, {len(done)} done, {len(todo)} to play, {args.workers} workers", flush=True)
    t0 = time.time()
    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    with ctx.Pool(args.workers, maxtasksperchild=40) as pool, done_path.open("a") as f:
        for k, r in enumerate(pool.imap_unordered(play_game, todo), 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            el = time.time() - t0
            print(f"[{k}/{len(todo)} {el / 60:.1f} min, eta {el / k * (len(todo) - k) / 60:.0f} min] "
                  f"{r['red']} vs {r['blue']} op{r['opening']}: {r['winner']} ({r['reason']}, {r['plies']} plies)",
                  flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
