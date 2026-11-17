"""Train the NNUE-style evaluation network (engine eval_mode 2 / 3) on self-play data.

    python -m tools.train_nnue data/selfplay_gen1.jsonl --h1 64 --h2 16 --epochs 12

Architecture (must match Search::evaluate_net in engine/src/search.cpp):

    sparse 312 binary inputs (side-to-move perspective; Blue's board is flipped so "me" walks up):
        0..80 my pawn square, 81..161 opponent pawn, 162..225 h walls, 226..289 v walls,
        290..300 my walls left (one-hot), 301..311 opponent walls left
    -> accumulator h1 (sum of embedding rows + bias), clipped ReLU [0, 1]
    ++ dense: the 10 linear features * 0.1, d_me * 0.1, d_opp * 0.1, phase * 0.05
    -> h2 ReLU -> 1 output (a logit). The engine multiplies by out_scale = 1/K to get eval units.

Target: ``lambda * result + (1 - lambda) * sigmoid(K * search_value)`` (K from the Texel fit,
engine/nets/linear_weights.json). Augmentation: left-right mirror. Writes engine/nets/nnue.txt.
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn

from quoridor.bots.profiles import NET_PATH, NETS_DIR

N, NW, NSQ, NWALL, MAXW = 9, 8, 81, 64, 10
NSPARSE = 2 * NSQ + 2 * NWALL + 2 * (MAXW + 1)
NFEAT = 10
NDENSE = NFEAT + 3
MAXACT = 26
WIN_THRESHOLD = 29000


def sparse_of(pos: str, mirror: bool = False) -> list[int]:
    t = pos.split()
    turn, r0, c0, r1, c1, w0, w1 = map(int, t[1:8])
    me = turn
    flip = me == 1

    def sq(r, c):
        if flip:
            r = N - 1 - r
        if mirror:
            c = N - 1 - c
        return r * N + c

    def an(r, c):
        if flip:
            r = NW - 1 - r
        if mirror:
            c = NW - 1 - c
        return r * NW + c

    pawns = [(r0, c0), (r1, c1)]
    wl = [w0, w1]
    idx = [sq(*pawns[me]), NSQ + sq(*pawns[1 - me])]
    for w in t[8:]:
        a = an(int(w[1]), int(w[2]))
        idx.append(2 * NSQ + (0 if w[0] == "h" else NWALL) + a)
    base = 2 * NSQ + 2 * NWALL
    idx += [base + wl[me], base + MAXW + 1 + wl[1 - me]]
    return idx


class Net(nn.Module):
    def __init__(self, h1: int, h2: int) -> None:
        super().__init__()
        self.emb = nn.EmbeddingBag(NSPARSE + 1, h1, mode="sum", padding_idx=NSPARSE)
        self.b1 = nn.Parameter(torch.zeros(h1))
        self.l2 = nn.Linear(h1 + NDENSE, h2)
        self.l3 = nn.Linear(h2, 1)
        nn.init.normal_(self.emb.weight, 0, 0.05)
        with torch.no_grad():
            self.emb.weight[NSPARSE].zero_()

    def forward(self, idx, dense):
        x = torch.clamp(self.emb(idx) + self.b1, 0, 1)
        h = torch.relu(self.l2(torch.cat([x, dense], 1)))
        return self.l3(h).squeeze(1)


def build(games, K, lam):
    rows = []
    for gi, g in enumerate(games):
        if g["winner"] is None:
            continue
        for pos, turn, value, depth, feats in g["rows"]:
            y = 1.0 if g["winner"] == turn else 0.0
            if abs(value) >= WIN_THRESHOLD:
                s = 1.0 if value > 0 else 0.0
            else:
                s = 1 / (1 + np.exp(-K * value))
            t = lam * y + (1 - lam) * s
            dense = [f * 0.1 for f in feats[:NFEAT]] + [feats[NFEAT] * 0.1, feats[NFEAT + 1] * 0.1, feats[NFEAT + 2] * 0.05]
            rows.append((pos, dense, t, y, gi))
    return rows


def tensors(rows, mirror=False):
    idx = np.full((len(rows), MAXACT), NSPARSE, np.int64)
    for i, r in enumerate(rows):
        s = sparse_of(r[0], mirror)
        idx[i, :len(s)] = s
    dense = np.array([r[1] for r in rows], np.float32)
    t = np.array([r[2] for r in rows], np.float32)
    y = np.array([r[3] for r in rows], np.float32)
    return torch.from_numpy(idx), torch.from_numpy(dense), torch.from_numpy(t), torch.from_numpy(y)


def export(net: Net, K: float, path) -> None:
    h1, h2 = net.b1.numel(), net.l3.in_features
    W1 = net.emb.weight.detach()[:NSPARSE].numpy()  # NSPARSE x h1
    W2 = net.l2.weight.detach().numpy().T  # (h1 + NDENSE) x h2
    parts = [f"{h1} {h2} {1.0 / K:.6f}"]
    for arr in (W1.ravel(), net.b1.detach().numpy(), W2.ravel(), net.l2.bias.detach().numpy(),
                net.l3.weight.detach().numpy().ravel(), net.l3.bias.detach().numpy()):
        parts.append(" ".join(f"{x:.6g}" for x in arr))
    path.write_text("\n".join(parts) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data", nargs="+")
    ap.add_argument("--h1", type=int, default=64)
    ap.add_argument("--h2", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--out", default=str(NET_PATH))
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    K = json.loads((NETS_DIR / "linear_weights.json").read_text())["K"]
    games = []
    for p in a.data:
        games += [json.loads(l) for l in open(p) if l.strip()]
    random.Random(0).shuffle(games)
    rows = build(games, K, a.lam)
    va = [r for r in rows if r[4] % 10 == 0]
    tr = [r for r in rows if r[4] % 10 != 0]
    print(f"{len(tr)} train / {len(va)} val positions, K={K:.4f}", flush=True)
    tr_n, tr_m = tensors(tr), tensors(tr, mirror=True)
    tr_all = [torch.cat([x, y]) for x, y in zip(tr_n, tr_m)]
    va_t = tensors(va)
    net = Net(a.h1, a.h2)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    bce = nn.functional.binary_cross_entropy_with_logits
    n = tr_all[0].shape[0]
    best = (9.0, None)
    for ep in range(a.epochs):
        t0 = time.time()
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, a.batch):
            b = perm[i:i + a.batch]
            opt.zero_grad()
            loss = bce(net(tr_all[0][b], tr_all[1][b]), tr_all[2][b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        net.eval()
        with torch.no_grad():
            z = net(va_t[0], va_t[1])
            vl = bce(z, va_t[2]).item()
            vr = bce(z, va_t[3]).item()
            acc = ((z > 0).float() == va_t[3]).float().mean().item()
        print(f"epoch {ep + 1}: train {tot / n:.4f}  val target {vl:.4f}  val result {vr:.4f}  sign acc {acc:.3f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
        if vl < best[0]:
            best = (vl, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1])
    from pathlib import Path
    export(net, K, Path(a.out))
    (Path(a.out).with_suffix(".json")).write_text(json.dumps(
        {"h1": a.h1, "h2": a.h2, "K": K, "lambda": a.lam, "val_target_logloss": best[0],
         "train_positions": len(tr), "val_positions": len(va)}, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
