"""Recompute the ``features`` column of self-play data with the current engine binary.

    python -m tools.refeature data/selfplay_gen1.jsonl data/selfplay_gen1.feat.jsonl

The features are a function of the position only, so data generated with an older
binary can be brought up to date (and stays consistent with the engine that uses it).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys

from quoridor.bots.profiles import ProfileEngine

_e = None


def work(line: str) -> str:
    global _e
    if _e is None:
        _e = ProfileEngine("turbo").engine
    g = json.loads(line)
    for row in g["rows"]:
        _e.command(row[0])
        row[4] = list(map(int, _e.command("features").split()[1:]))
    return json.dumps(g)


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    lines = [l for l in open(src) if l.strip()]
    with mp.Pool(4) as pool, open(dst, "w") as f:
        for out in pool.imap(work, lines, chunksize=20):
            f.write(out + "\n")
    print("wrote", dst, len(lines), "games")


if __name__ == "__main__":
    main()
