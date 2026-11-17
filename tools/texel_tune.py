"""Texel tuning of the phase-tapered linear evaluation (engine eval_mode 1).

    python -m tools.texel_tune data/selfplay_gen1.jsonl

Fits P(side to move wins) = sigmoid(K * eval) on self-play positions, where
eval = sum_i f_i * (w_mg_i * phase/20 + w_eg_i * (1 - phase/20)) with the engine's
features (``features`` command). The target blends the game result with the
search score: ``lambda * result + (1 - lambda) * sigmoid(K * search_value)``.

K is first fitted for the classic hand-set evaluation, so the tuned weights stay in
the same units (10 ~ one step) and the search margins (futility, aspiration) keep
their meaning. Writes engine/nets/linear_weights.txt (2 x NFEAT integers) and
engine/nets/linear_weights.json (report).
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch

from quoridor.bots.profiles import LINEAR_WEIGHTS, NETS_DIR

NFEAT = 10
NAMES = ["dist", "wall_full", "wall_low", "wall_def", "wall_tie", "wall_surplus", "tempo", "narrow", "paths",
         "dist_prod"]
CLASSIC = np.array([10, 16, 10, 6, 1, 1, 5, 0, 0, 0], dtype=np.float64)
WIN_THRESHOLD = 29000


def load(paths, max_games=None, seed=0):
    games = []
    for p in paths:
        with open(p) as f:
            for line in f:
                if line.strip():
                    games.append(json.loads(line))
    random.Random(seed).shuffle(games)
    if max_games:
        games = games[:max_games]
    return games


def to_arrays(games):
    F, PH, Y, V, G = [], [], [], [], []
    for gi, g in enumerate(games):
        if g["winner"] is None:
            continue
        for pos, turn, value, depth, feats in g["rows"]:
            if abs(value) >= WIN_THRESHOLD or depth < 2:
                continue
            F.append(feats[:NFEAT])
            PH.append(min(20, feats[NFEAT + 2]) / 20.0)
            Y.append(1.0 if g["winner"] == turn else 0.0)
            V.append(value)
            G.append(gi)
    return (np.array(F, np.float64), np.array(PH), np.array(Y), np.array(V, np.float64), np.array(G))


def fit_k(score, y):
    """1-D logistic fit of K for a fixed evaluation."""
    best = None
    for k in np.linspace(0.002, 0.08, 157):
        p = 1 / (1 + np.exp(-k * score))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if best is None or ll < best[0]:
            best = (ll, k)
    return best[1], best[0]


def logloss(p, t):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data", nargs="+")
    ap.add_argument("--lam", type=float, default=0.5, help="weight of the game result in the target")
    ap.add_argument("--l2", type=float, default=1e-4, help="pull towards the classic weights")
    ap.add_argument("--max-games", type=int)
    a = ap.parse_args(argv)

    games = load(a.data, a.max_games)
    F, PH, Y, V, G = to_arrays(games)
    n_games = G.max() + 1
    val = G % 10 == 0  # 10% of the games for validation
    tr = ~val
    print(f"{len(Y)} positions from {n_games} games ({val.sum()} validation)")

    classic = F @ CLASSIC
    K, ll_c = fit_k(classic[tr], Y[tr])
    print(f"K = {K:.4f} (classic eval, result-only log-loss {ll_c:.4f})")
    T = a.lam * Y + (1 - a.lam) / (1 + np.exp(-K * V))

    Ft = torch.tensor(F)
    PHt = torch.tensor(PH).unsqueeze(1)
    X = torch.cat([Ft * PHt, Ft * (1 - PHt)], 1)  # mg features, eg features
    Tt = torch.tensor(T)
    w = torch.tensor(np.concatenate([CLASSIC, CLASSIC]) * K, requires_grad=True)
    w0 = w.detach().clone()
    trm = torch.tensor(tr)
    opt = torch.optim.LBFGS([w], lr=0.5, max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = X[trm] @ w
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, Tt[trm]) + a.l2 * ((w - w0) ** 2).sum()
        loss.backward()
        return loss

    for _ in range(3):
        opt.step(closure)
    wt = w.detach().numpy() / K  # back to engine units
    mg, eg = np.round(wt[:NFEAT]).astype(int), np.round(wt[NFEAT:]).astype(int)
    mg[6] = max(mg[6], 0)
    eg[6] = max(eg[6], 0)

    def evaluate(wmg, weg):
        return (F * PH[:, None]) @ wmg + (F * (1 - PH[:, None])) @ weg

    report = {"positions": int(len(Y)), "games": int(n_games), "K": K, "lambda": a.lam}
    for name, (wmg, weg) in {"classic": (CLASSIC, CLASSIC), "tuned": (mg, eg)}.items():
        s = evaluate(wmg.astype(float), weg.astype(float))
        p = 1 / (1 + np.exp(-K * s))
        report[name] = {"val_logloss_target": logloss(p[val], T[val]), "val_logloss_result": logloss(p[val], Y[val]),
                        "val_sign_accuracy": float(np.mean((s[val] > 0) == (Y[val] > 0.5)))}
        print(name, report[name])
    report["weights"] = {"mg": dict(zip(NAMES, mg.tolist())), "eg": dict(zip(NAMES, eg.tolist()))}
    for k in ("mg", "eg"):
        print(k, report["weights"][k])
    NETS_DIR.mkdir(parents=True, exist_ok=True)
    LINEAR_WEIGHTS.write_text(" ".join(map(str, mg.tolist())) + "\n" + " ".join(map(str, eg.tolist())) + "\n")
    (NETS_DIR / "linear_weights.json").write_text(json.dumps(report, indent=1))
    print("wrote", LINEAR_WEIGHTS)


if __name__ == "__main__":
    main()
