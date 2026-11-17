"""Post-mortem statistics from tournament move logs (results/*/games.jsonl).

    python -m tools.postmortem results/rr_0.5s results/rr_5s [--md out.md]

Each logged move (after the opening) is [move, value, depth, seconds] with the value from the mover's point of
view (book moves have depth 0 and are skipped). Per bot:

* **Eval accuracy**: share of its searched, non-forced evaluations whose sign agrees with
  the final result, by game phase (plies 0-19 / 20-39 / 40+). A better evaluation
  predicts the winner earlier.
* **Proof ply**: in won games, the ply at which the bot first saw a forced win; in lost
  games, the ply at which it first saw the forced loss. ``lead`` = how many plies before
  its opponent the bot saw the outcome of a game it won.
* **Surprised losses**: lost games where the bot's own score was >= +15 at one of its last
  three evaluations before it saw the forced loss: the loss was over its horizon.
* **Right when both claim the edge**: on consecutive plies 12-39 both sides' scores were
  > +10 (each thinks it is better); how often this bot was the one that won.
* **Walls**: average walls placed in plies 0-19 / 20-39, walls left at the end in wins and
  losses, and how often the side with more walls left at ply 40 won.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

WIN_THRESHOLD = 29000
PHASES = ((0, 20), (20, 40), (40, 999))


def analyse(path: Path) -> str:
    cfg = json.loads((path / "config.json").read_text())
    games = [json.loads(l) for l in (path / "games.jsonl").read_text().splitlines() if l.strip()]
    bots = cfg["bots"]
    acc = {b: [[0, 0] for _ in PHASES] for b in bots}
    proof_win = defaultdict(list)
    proof_loss = defaultdict(list)
    lead = defaultdict(list)
    surprised = defaultdict(int)
    losses = defaultdict(int)
    walls_early = {b: [[0, 0] for _ in PHASES[:2]] for b in bots}  # [walls, games]
    end_walls = defaultdict(lambda: [0, 0, 0, 0])  # win sum, win n, loss sum, loss n
    dispute = defaultdict(lambda: [0, 0])  # both sides claim >= +10 on consecutive plies 12-39: [bot right, n]
    surplus = [0, 0]  # games where the side with more walls at ply 40 won, decided games with a surplus
    for g in games:
        names = (g["red"], g["blue"])
        win = g["winner"]
        first_win = {}
        first_loss = {}
        evals = defaultdict(list)
        walls_at = [10, 10]
        walls40 = None
        start = len(cfg["openings"][g["opening"]])  # opening moves are not logged
        for i_mv, (mv, val, depth, _secs) in enumerate(g["moves"]):
            ply = start + i_mv
            side = ply % 2
            bot = names[side]
            if ply == 40:
                walls40 = tuple(walls_at)
            if mv[0] in "hv":
                walls_at[side] -= 1
                for i, (lo, hi) in enumerate(PHASES[:2]):
                    if lo <= ply < hi:
                        walls_early[bot][i][0] += 1
            if depth == 0:
                continue
            if val >= WIN_THRESHOLD:
                first_win.setdefault(side, ply)
            elif val <= -WIN_THRESHOLD:
                first_loss.setdefault(side, ply)
            else:
                evals[side].append(val)
                if win is not None and val != 0:
                    for i, (lo, hi) in enumerate(PHASES):
                        if lo <= ply < hi:
                            acc[bot][i][1] += 1
                            acc[bot][i][0] += (val > 0) == (win == bot)
        if win is not None:
            seen = {start + i: (v, d) for i, (_m, v, d, _s) in enumerate(g["moves"])}
            for ply in range(12, 40):
                a, b = seen.get(ply), seen.get(ply + 1)
                if a and b and a[1] and b[1] and 10 < a[0] < WIN_THRESHOLD and 10 < b[0] < WIN_THRESHOLD:
                    for bot in names:
                        dispute[bot][1] += 1
                    dispute[win][0] += 1
        for bot in set(names):
            for i in range(2):
                walls_early[bot][i][1] += names.count(bot)
        if win is None:
            continue
        ws = names.index(win)
        ls = 1 - ws
        loser = names[ls]
        if ws in first_win:
            proof_win[win].append(first_win[ws])
            other = first_loss.get(ls, g["plies"])
            lead[win].append(other - first_win[ws])
        if ls in first_loss:
            proof_loss[loser].append(first_loss[ls])
        losses[loser] += 1
        if any(v >= 15 for v in evals[ls][-3:]):
            surprised[loser] += 1
        wl = g["walls_left"]
        end_walls[win][0] += wl[ws]
        end_walls[win][1] += 1
        end_walls[loser][2] += wl[ls]
        end_walls[loser][3] += 1
        if walls40 and walls40[0] != walls40[1]:
            surplus[1] += 1
            surplus[0] += (walls40[ws] > walls40[ls])

    def avg(x):
        return sum(x) / len(x) if x else float("nan")

    lines = [f"## Post-mortem `{path.name}` ({cfg['time']} s/move, {len(games)} games)", "",
             "| Bot | Eval accuracy plies 0-19 / 20-39 / 40+ | Forced win seen at ply | Lead over opponent (plies) "
             "| Forced loss seen at ply | Surprised losses | Walls placed plies 0-19 / 20-39 "
             "| Walls left in wins / losses | Right when both claim the edge |",
             "|---" * 9 + "|"]
    for b in bots:
        a = " / ".join(f"{c / n:.0%}" if n else "-" for c, n in acc[b])
        we = " / ".join(f"{w / n:.1f}" for w, n in walls_early[b])
        ew = end_walls[b]
        lines.append(
            f"| **{b}** | {a} | {avg(proof_win[b]):.1f} | {avg(lead[b]):+.1f} | {avg(proof_loss[b]):.1f} | "
            f"{surprised[b]}/{losses[b]} ({surprised[b] / max(losses[b], 1):.0%}) | {we} | "
            f"{ew[0] / max(ew[1], 1):.1f} / {ew[2] / max(ew[3], 1):.1f} | "
            f"{dispute[b][0]}/{dispute[b][1]} ({dispute[b][0] / max(dispute[b][1], 1):.0%}) |")
    lines.append("")
    lines.append(f"The side with more walls left at ply 40 won {surplus[0]}/{surplus[1]} "
                 f"({surplus[0] / max(surplus[1], 1):.0%}) of decided games with a wall difference.")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--md")
    a = ap.parse_args(argv)
    text = "\n\n".join(analyse(Path(d)) for d in a.dirs)
    print(text)
    if a.md:
        Path(a.md).write_text(text + "\n")


if __name__ == "__main__":
    main()
