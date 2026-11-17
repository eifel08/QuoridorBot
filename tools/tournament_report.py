"""Summarise a tournament run by tools.tournament.

    python -m tools.tournament_report results/rr_5s [--md report.md]

Prints the win matrix (row bot's wins against the column bot), W/L/D, score, Elo
(Bradley-Terry maximum likelihood, anchored at baseline = 0, with a +-2 sigma interval
from bootstrap), average game length, search depth, nodes/s, time per move, book use,
and time violations.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def load(path: Path):
    cfg = json.loads((path / "config.json").read_text())
    games = [json.loads(l) for l in (path / "games.jsonl").read_text().splitlines() if l.strip()]
    return cfg, games


def bt_elo(bots, games, iters=200):
    """Bradley-Terry ratings (draws = half wins) by the MM algorithm; Elo units, mean-zero."""
    w = defaultdict(float)
    n = defaultdict(float)
    for g in games:
        a, b = g["red"], g["blue"]
        n[(a, b)] += 1
        n[(b, a)] += 1
        if g["winner"] is None:
            w[a] += 0.5
            w[b] += 0.5
        else:
            w[g["winner"]] += 1
    r = {x: 1.0 for x in bots}
    for _ in range(iters):
        new = {}
        for i in bots:
            den = sum(n[(i, j)] / (r[i] + r[j]) for j in bots if j != i and n[(i, j)])
            new[i] = max(w[i], 0.05) / den if den else r[i]
        s = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        r = {k: v / s for k, v in new.items()}
    return {k: 400 * math.log10(v) for k, v in r.items()}


def report(path: Path) -> str:
    cfg, games = load(path)
    bots = cfg["bots"]
    wins = defaultdict(int)
    played = defaultdict(int)
    tot = {b: defaultdict(float) for b in bots}
    for g in games:
        a, b = g["red"], g["blue"]
        played[(a, b)] += 1
        played[(b, a)] += 1
        if g["winner"]:
            loser = b if g["winner"] == a else a
            wins[(g["winner"], loser)] += 1
        for bot in (a, b):
            t = tot[bot]
            t["games"] += 1
            t["plies"] += g["plies"]
            if g["winner"] == bot:
                t["w"] += 1
                if bot == a:
                    t["w_red"] += 1
            elif g["winner"] is None:
                t["d"] += 1
            else:
                t["l"] += 1
            if bot == a:
                t["red"] += 1
            if "violation" in g["reason"] and g["winner"] != bot:
                t["violations"] += 1
            s = g["stats"][bot]
            for k in ("moves", "book", "depth", "nodes", "seconds"):
                t[k] += s[k]
            t["max_seconds"] = max(t["max_seconds"], s["max_seconds"])
    elo = bt_elo(bots, games)
    anchor = "baseline" if "baseline" in bots else bots[0]
    base = elo.get(anchor, 0.0)
    elo = {k: v - base for k, v in elo.items()}
    rng = random.Random(0)
    boots = defaultdict(list)
    for _ in range(100):
        sample = [rng.choice(games) for _ in games]
        e = bt_elo(bots, sample, 80)
        for k in bots:
            boots[k].append(e[k] - e.get(anchor, 0.0))
    order = sorted(bots, key=lambda b: -elo[b])

    lines = [f"## Tournament `{path.name}`: {cfg['time']} s per move, {len(games)} games, "
             f"{len(cfg['openings'])} openings x 2 colours per pair", ""]
    lines.append("### Results matrix (row's wins - column's wins)")
    lines.append("")
    lines.append("| | " + " | ".join(order) + " | Total W-L-D | Score |")
    lines.append("|---" * (len(order) + 3) + "|")
    for a in order:
        cells = []
        for b in order:
            cells.append("-" if a == b else f"{wins[(a, b)]}-{wins[(b, a)]}" if played[(a, b)] else "")
        t = tot[a]
        score = (t["w"] + 0.5 * t["d"]) / max(t["games"], 1)
        lines.append(f"| **{a}** | " + " | ".join(cells) + f" | {int(t['w'])}-{int(t['l'])}-{int(t['d'])} | {score:.1%} |")
    lines.append("")
    lines.append("### Ranking and engine statistics")
    lines.append("")
    lines.append(f"| Rank | Bot | Elo vs {anchor} (±2σ) | Win % | Win % as Red | Avg game length (plies) | "
                 "Avg depth | Nodes/s | Avg s/move | Max s/move | Book moves | Time violations |")
    lines.append("|---" * 12 + "|")
    for i, b in enumerate(order, 1):
        t = tot[b]
        bs = sorted(boots[b])
        sd = (bs[94] - bs[5]) / 2 if len(bs) >= 95 else 0
        searched = max(t["moves"] - t["book"], 1)
        nps = t["nodes"] / t["seconds"] if t["seconds"] else 0
        lines.append(
            f"| {i} | **{b}** | {elo[b]:+.0f} (±{sd:.0f}) | {t['w'] / max(t['games'], 1):.1%} | "
            f"{t['w_red'] / max(t['red'], 1):.1%} | {t['plies'] / max(t['games'], 1):.1f} | "
            f"{t['depth'] / searched:.1f} | {nps / 1e6:.2f} M | {t['seconds'] / searched:.2f} | "
            f"{t['max_seconds']:.2f} | {int(t['book'])} | {int(t['violations'])} |")
    reasons = defaultdict(int)
    for g in games:
        reasons[g["reason"].split()[0]] += 1
    lines.append("")
    lines.append("Game endings: " + ", ".join(f"{k}: {v}" for k, v in sorted(reasons.items())))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--md")
    a = ap.parse_args(argv)
    text = "\n\n".join(report(Path(d)) for d in a.dirs)
    print(text)
    if a.md:
        Path(a.md).write_text(text + "\n")


if __name__ == "__main__":
    main()
