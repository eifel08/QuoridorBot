#!/usr/bin/env bash
# Unattended: build the opening book, then run both tournaments (tools/run_tournaments.sh).
#   nohup setsid bash tools/run_all.sh > results/run.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
if [ ! -f engine/book/book.json ]; then
  echo "== building opening book =="
  python3 -m tools.build_book --profile nnue --time 15 --alt-time 8 --max-ply 10
fi
bash tools/run_tournaments.sh
