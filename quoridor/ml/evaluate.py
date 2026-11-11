"""Evaluate a checkpoint against the alpha-beta bot.

    python -m quoridor.ml.evaluate --model runs/exp1/best.pt --games 200
    python -m quoridor.ml.evaluate --baseline 6 --games 100   # alpha-beta depth 6 instead of a model
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time

from .arena import alphabeta_policy, match, wilson_lower_bound


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description="Win rate vs the Depth-N alpha-beta bot")
    p.add_argument("--model", help="checkpoint (.pt) to evaluate")
    p.add_argument("--baseline", type=int, help="evaluate alpha-beta at this depth instead of a model")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--sims", type=int, default=64, help="MCTS simulations per agent move (0 = raw policy)")
    p.add_argument("--opponent-depth", type=int, default=4)
    p.add_argument("--seed", type=int, default=2024, help="opening seed (same seed = same openings)")
    p.add_argument("--opening-plies", type=int, default=6)
    p.add_argument("--workers", type=int, default=0, help="0 = all CPU cores")
    a = p.parse_args(argv)
    t0 = time.time()

    if a.baseline:
        r = match(alphabeta_policy(a.baseline), alphabeta_policy(a.opponent_depth), a.games, a.seed, a.opening_plies)
    elif a.model:
        from .model import load_model
        from .selfplay import GameSettings, default_workers, run_eval, serialize_model, worker_init

        net, ckpt = load_model(a.model, "cpu")
        gs = GameSettings(opponent_depth=a.opponent_depth, eval_sims=a.sims, opening_plies=a.opening_plies)
        blob = serialize_model(net)
        ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
        with ctx.Pool(a.workers or default_workers(), initializer=worker_init) as pool:
            res = list(pool.imap_unordered(run_eval, [(i, a.seed, 1, blob, gs) for i in range(a.games)]))
        wins = sum(x["won"] for x in res)
        r = {"games": a.games, "wins": wins, "draws": sum(x["winner"] is None for x in res),
             "win_rate": wins / a.games, "wilson_low": wilson_lower_bound(wins, a.games),
             "avg_plies": sum(x["plies"] for x in res) / a.games,
             "trained_iterations": ckpt.get("iteration")}
    else:
        p.error("give --model or --baseline")
    r["seconds"] = round(time.time() - t0, 1)
    who = a.model or f"alpha-beta depth {a.baseline}"
    print(f"{who} vs Depth-{a.opponent_depth}: win rate {r['win_rate']:.1%} "
          f"({r['wins']}/{r['games']}, 95% CI lower bound {r['wilson_low']:.1%}) in {r['seconds']}s")
    return r


if __name__ == "__main__":
    main()
