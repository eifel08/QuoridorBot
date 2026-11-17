#!/usr/bin/env bash
# Run both round-robin tournaments unattended, then write the reports.
#
#   nohup setsid bash tools/run_tournaments.sh > results/run.log 2>&1 &
#
# Both are resumable: rerunning skips the games already in results/*/games.jsonl.
# Environment overrides: BOTS, WORKERS, FAST_OPENINGS, SLOW_OPENINGS.
set -euo pipefail
cd "$(dirname "$0")/.."
BOTS=${BOTS:-baseline,turbo,tuned,book,nnue,hybrid}
WORKERS=${WORKERS:-4}
FAST_OPENINGS=${FAST_OPENINGS:-50}   # 100 games per pair at 0.5 s/move
SLOW_OPENINGS=${SLOW_OPENINGS:-8}    # 16 games per pair at 5 s/move
mkdir -p results

make -C engine >/dev/null
make -C engine baseline >/dev/null

echo "== fast round robin: 0.5 s per move =="
python3 -m tools.tournament --out results/rr_0.5s --bots "$BOTS" --time 0.5 --openings "$FAST_OPENINGS" --workers "$WORKERS"
python3 -m tools.tournament_report results/rr_0.5s --md results/rr_0.5s/report.md

echo "== headline round robin: 5 s per move =="
python3 -m tools.tournament --out results/rr_5s --bots "$BOTS" --time 5 --openings "$SLOW_OPENINGS" --workers "$WORKERS"
python3 -m tools.tournament_report results/rr_5s --md results/rr_5s/report.md

echo "ALL TOURNAMENTS DONE"
