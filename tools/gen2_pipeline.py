"""Generation-2 NNUE: self-play by the generation-1 neural bots, training, gating, benchmark.

    nohup setsid python -m tools.gen2_pipeline > results/gen2/pipeline.log 2>&1 &

Every stage is skipped when its output exists, so the pipeline can be restarted.

1. **Self-play** (``data/selfplay_gen2.jsonl``): nnue and hybrid (generation 1, chosen per side),
   ``--time`` s per move, random 0-6 ply openings, 8 % exploration in the first 40 plies.
2. **Training**, three candidates (``engine/nets/nnue_gen2{a,b,c}.txt``), validated on held-out
   generation-2 games:
   a) generation-2 data only, 64x16; b) generation 1 + 2, 64x16; c) generation 1 + 2, 128x16
   with weight decay.
3. **Gating**: each candidate (``nnue-gen2X``) against generation-1 ``nnue``, 0.2 s per move,
   ``--gate-openings`` openings x 2 colours. The best scorer becomes ``engine/nets/nnue_gen2.txt``.
4. **Benchmark** round-robins of nnue, hybrid (generation 1) and nnue-gen2, hybrid-gen2:
   0.5 s per move (``--fast-openings`` x 2 per pair) and 5 s per move (``--slow-openings`` x 2).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NETS = ROOT / "engine" / "nets"
OUT = ROOT / "results" / "gen2"
GEN1_DATA = ROOT / "data" / "gen1.feat.jsonl"
GEN2_DATA = ROOT / "data" / "selfplay_gen2.jsonl"

CANDIDATES = {
    "gen2a": ["--h1", "64", "--h2", "16", "--epochs", "12"],
    "gen2b": ["--h1", "64", "--h2", "16", "--epochs", "12"],
    "gen2c": ["--h1", "128", "--h2", "16", "--epochs", "12", "--wd", "0.01"],
}


def run(cmd: list[str], log: Path | None = None) -> None:
    print("$", " ".join(cmd), flush=True)
    if log is None:
        subprocess.run(cmd, cwd=ROOT, check=True)
    else:
        with open(log, "w") as f:
            subprocess.run(cmd, cwd=ROOT, check=True, stdout=f, stderr=subprocess.STDOUT)


def lines(path: Path) -> int:
    return sum(1 for _ in open(path)) if path.is_file() else 0


def score(results: Path, bot: str) -> tuple[float, int]:
    games = [json.loads(l) for l in open(results / "games.jsonl") if l.strip()]
    pts = sum(1.0 if g["winner"] == bot else 0.5 if g["winner"] is None else 0.0 for g in games)
    return pts / max(len(games), 1), len(games)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=7000)
    ap.add_argument("--time", type=float, default=0.12)
    ap.add_argument("--gate-openings", type=int, default=100)
    ap.add_argument("--fast-openings", type=int, default=50)
    ap.add_argument("--slow-openings", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    t0 = time.time()

    # 1. self-play
    have = lines(GEN2_DATA)
    if have < a.games:
        print(f"== self-play: {have} games so far, {a.games - have} to play", flush=True)
        run([py, "-m", "tools.selfplay_data", "--out", str(GEN2_DATA), "--profile", "nnue,hybrid",
             "--games", str(a.games - have), "--time", str(a.time), "--seed", str(2000 + have),
             "--workers", str(a.workers)])
    print(f"== self-play done: {lines(GEN2_DATA)} games ({time.time() - t0:.0f}s)", flush=True)

    # 2. training
    for tag, extra in CANDIDATES.items():
        net = NETS / f"nnue_{tag}.txt"
        if net.is_file():
            continue
        data = [str(GEN2_DATA)] if tag == "gen2a" else [str(GEN1_DATA), str(GEN2_DATA)]
        print(f"== training {tag}", flush=True)
        run([py, "-m", "tools.train_nnue", *data, "--val-last", "--out", str(net), *extra],
            OUT / f"train_{tag}.log")
        print(open(OUT / f"train_{tag}.log").read(), flush=True)

    # 3. gating against generation 1
    results = {}
    for tag in CANDIDATES:
        out = OUT / f"gate_{tag}"
        run([py, "-m", "tools.tournament", "--out", str(out), "--bots", f"nnue,nnue-{tag}", "--time", "0.2",
             "--openings", str(a.gate_openings), "--workers", str(a.workers)], OUT / f"gate_{tag}.log")
        results[tag] = score(out, f"nnue-{tag}")
        val = json.loads((NETS / f"nnue_{tag}.json").read_text())["val_target_logloss"]
        print(f"== gate {tag}: {results[tag][0]:.1%} of {results[tag][1]} games vs generation 1, "
              f"val loss {val:.4f}", flush=True)
    best = max(results, key=lambda t: results[t][0])
    (OUT / "gate_summary.json").write_text(json.dumps(
        {t: {"score_vs_gen1": s, "games": n, **json.loads((NETS / f"nnue_{t}.json").read_text())}
         for t, (s, n) in results.items()} | {"chosen": best}, indent=1))
    shutil.copy(NETS / f"nnue_{best}.txt", NETS / "nnue_gen2.txt")
    shutil.copy(NETS / f"nnue_{best}.json", NETS / "nnue_gen2.json")
    print(f"== chosen: {best} -> engine/nets/nnue_gen2.txt", flush=True)

    # 4. benchmark
    bots = "nnue,hybrid,nnue-gen2,hybrid-gen2"
    for name, t, n in (("rr_0.5s", "0.5", a.fast_openings), ("rr_5s", "5", a.slow_openings)):
        out = OUT / name
        print(f"== benchmark {name}", flush=True)
        run([py, "-m", "tools.tournament", "--out", str(out), "--bots", bots, "--time", t, "--openings", str(n),
             "--workers", str(a.workers)])
        run([py, "-m", "tools.tournament_report", str(out), "--md", str(out / "report.md")])
        run([py, "-m", "tools.postmortem", str(out), "--md", str(out / "postmortem.md")])
    print(f"GEN2 PIPELINE DONE ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
