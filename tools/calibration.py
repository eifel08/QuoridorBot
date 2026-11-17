"""Eval calibration: does the engine's score predict who actually wins?

    python -m tools.calibration --games 200 --time 0.1
    python -m tools.calibration --games 200 --time 0.1 --set wall_weight=6,wall_cap=1

Self-play from seeded random openings (both sides use the same options). Every
move's backed-up search score is recorded from the mover's point of view and
compared with the final result:

* **false win claims**: the mover's eval was >= ``--claim`` steps (it thought it
  was clearly ahead) but it lost the game. Split by how close the opponent was
  to its goal row at that moment (``opp <= 3`` = an endgame race).
* **sudden flips**: the mover claimed >= ``--claim`` steps and its very next
  search found a forced loss (``-M``): the race was already lost and the eval
  had not seen it.
* **sign accuracy**: how often ``eval > 0`` matches the eventual winner, over
  all non-forced evals and over endgame evals (opponent <= 3 steps).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import sys
import time

from quoridor.bots.cpp_engine import DEFAULT_OPTIONS, CppEngine, to_imove
from quoridor.evaluation import Eval
from quoridor.ml.arena import random_opening

_engine: dict = {}


def parse_options(text: str) -> dict:
    opts = {}
    for item in filter(None, (t.strip() for t in text.split(","))):
        key, _, value = item.partition("=")
        if key not in DEFAULT_OPTIONS:
            raise SystemExit(f"unknown engine option '{key}'")
        opts[key] = int(value)
    return opts


def play(job) -> dict:
    index, options, depth, time_limit, seed, opening_plies = job
    key = os.getpid()
    if key not in _engine:
        _engine[key] = CppEngine(options={"hash_mb": 32, **options})
    engine = _engine[key]
    engine.new_game()
    rng = random.Random(seed * 1_000_003 + index)
    board = random_opening(rng, opening_plies)
    records = []
    winner = None
    for _ in range(200):
        me = board.turn
        engine.set_position(board)
        d_me, d_opp = engine.distances() if me == 0 else engine.distances()[::-1]
        move, info = engine.go(depth, time_limit)
        if move is None:
            winner = 1 - me
            break
        scale = options.get("dist_weight", DEFAULT_OPTIONS["dist_weight"])
        info["eval_steps"] = info["value"] / scale if not info["forced"] else None
        ev = Eval.from_info(info, me)
        records.append((me, ev.steps, ev.mate, d_me, d_opp, board.walls_left[me], board.walls_left[1 - me]))
        if board.play(to_imove(move)) is not None:
            winner = me
            break
    return {"winner": winner, "records": records}


def analyse(games: list[dict], claim: float) -> dict:
    s = {"games": len(games), "evals": 0, "claims": 0, "false_claims": 0, "false_claims_endgame": 0,
         "claims_endgame": 0, "flips": 0, "sign_ok": 0, "sign_n": 0, "sign_ok_end": 0, "sign_n_end": 0,
         "examples": []}
    for g in games:
        w = g["winner"]
        if w is None:
            continue
        recs = g["records"]
        last_by_player: dict[int, tuple] = {}
        for i, (me, steps, mate, d_me, d_opp, wm, wo) in enumerate(recs):
            prev = last_by_player.get(me)
            if prev is not None and prev[1] is not None and prev[1] >= claim and mate is not None and mate < 0:
                s["flips"] += 1
                if len(s["examples"]) < 5:
                    s["examples"].append(f"claimed {prev[1]:+.1f} (me {prev[3]} steps, opp {prev[4]}, walls "
                                         f"{prev[5]} vs {prev[6]}) -> next move -M{-mate}")
            last_by_player[me] = (me, steps, mate, d_me, d_opp, wm, wo)
            if steps is None:
                continue
            s["evals"] += 1
            endgame = d_opp <= 3
            if steps != 0:
                ok = (steps > 0) == (w == me)
                s["sign_n"] += 1
                s["sign_ok"] += ok
                if endgame:
                    s["sign_n_end"] += 1
                    s["sign_ok_end"] += ok
            if steps >= claim:
                s["claims"] += 1
                s["claims_endgame"] += endgame
                if w != me:
                    s["false_claims"] += 1
                    s["false_claims_endgame"] += endgame
    return s


def run(options: dict, games: int, depth, time_limit, seed=0, opening_plies=6, workers=0) -> list[dict]:
    jobs = [(i, options, depth, time_limit, seed, opening_plies) for i in range(games)]
    workers = min(workers or (os.cpu_count() or 1), games)
    with mp.get_context("fork" if sys.platform != "win32" else "spawn").Pool(workers) as pool:
        return pool.map(play, jobs)


def report(s: dict, claim: float) -> str:
    def pct(a, b):
        return f"{a}/{b} ({a / b:.1%})" if b else "0/0"
    lines = [
        f"{s['games']} games, {s['evals']} non-forced evals",
        f"  false win claims (eval >= {claim:+g} but lost): {pct(s['false_claims'], s['claims'])}",
        f"    ... when the opponent was <= 3 steps away:     {pct(s['false_claims_endgame'], s['claims_endgame'])}",
        f"  sudden flips (>= {claim:+g}, next move -M):         {s['flips']}",
        f"  sign accuracy (eval > 0 => wins): all {pct(s['sign_ok'], s['sign_n'])}, "
        f"endgame {pct(s['sign_ok_end'], s['sign_n_end'])}",
    ]
    lines += [f"    e.g. {e}" for e in s["examples"][:3]]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", default="", help="engine options, e.g. wall_weight=6,wall_cap=1")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--depth", type=int)
    p.add_argument("--time", type=float, default=0.1)
    p.add_argument("--claim", type=float, default=1.0, help="eval (in steps) that counts as 'clearly ahead'")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    t0 = time.time()
    games = run(parse_options(a.set), a.games, a.depth, None if a.depth else a.time, a.seed)
    print(f"options: {a.set or 'defaults'}  ({time.time() - t0:.0f}s)")
    print(report(analyse(games, a.claim), a.claim))


if __name__ == "__main__":
    main()
