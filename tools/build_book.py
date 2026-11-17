"""Build the opening book (engine/book/book.json) with long engine searches.

    python -m tools.build_book --profile tuned --time 15 --alt-time 8 --max-ply 10

The tree is expanded level by level from the start position, with the colour-flip /
left-right-mirror canonical form (quoridor.bots.profiles.canonical) merging transpositions
and mirror images:

* every node gets a deep search (``--time``): its best move M1 and score s1;
* the second-best move M2 (same search with M1 excluded, ``--alt-time``) is added to the
  book when s2 >= s1 - ``--margin`` (the book then picks between them, weighted);
* children: the book moves, plus at plies < ``--wide-ply`` every pawn move (so short
  openings that deviate from the main line stay in book), plus seed lines.

**Theory seed lines.** The named openings are only *candidates*; each line is
followed while the deep search keeps its moves within ``--margin`` of the best (otherwise
the search's refutation is what gets stored). Seeds (Red's view; Blue mirrors them):

* centre march: the pawn walks straight up three times (the standard opening);
* Shiller-style: three steps, then a vertical wall beside the opponent's file;
* side wall / Reed-style: early horizontal walls on the 3rd/6th rank to the sides;
* rush: walking straight ahead ignoring walls (the engine plays the refutations).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time

from quoridor.bots.alphabeta import FastBoard
from quoridor.bots.cpp_engine import WIN_THRESHOLD, fastboard_command
from quoridor.bots.profiles import BOOK_PATH, ProfileEngine, canonical, imove, transform_move

SEEDS = [
    ["m74", "m14", "m64", "m24", "m54", "m34"],                       # centre march
    ["m74", "m14", "m64", "m24", "m54", "m34", "v33"],                # Shiller-style vertical wall
    ["m74", "m14", "m64", "m24", "h22", "h55"],                       # Reed-style side walls
    ["m74", "m14", "h62", "h15"],                                     # early side wall
    ["m74", "m14", "m64", "m24", "m54", "h43"],                       # rush answered by a block
]

_eng = {}


def analyse(job):
    pos_moves, profile, tl, alt_tl = job
    if profile not in _eng:
        _eng[profile] = ProfileEngine(profile)
    e = _eng[profile].engine
    board = FastBoard.initial()
    for m in pos_moves:
        board.play(imove(m))
    e.new_game()
    e.set_position(fastboard_command(board))
    m1, i1 = e.go(64, tl)
    out = {"moves": pos_moves, "best": i1["move"], "score": i1["value"], "depth": i1["depth_reached"]}
    if m1 is not None and abs(i1["value"]) < WIN_THRESHOLD and alt_tl > 0:
        tok = e.command(f"go depth 64 time {alt_tl:.3f} exclude {i1['move']}").split()
        if tok[1] != "none":
            out["alt"], out["alt_score"] = tok[1], int(tok[tok.index("score") + 1])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="tuned")
    ap.add_argument("--time", type=float, default=15)
    ap.add_argument("--alt-time", type=float, default=8)
    ap.add_argument("--max-ply", type=int, default=10)
    ap.add_argument("--wide-ply", type=int, default=3)
    ap.add_argument("--alt-ply", type=int, default=7, help="add second-best moves up to this ply")
    ap.add_argument("--margin", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(BOOK_PATH))
    a = ap.parse_args(argv)

    book: dict[str, list] = {}
    info: dict[str, dict] = {}
    seen = set()
    level = [[]]
    t0 = time.time()
    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    with ctx.Pool(a.workers) as pool:
        for ply in range(a.max_ply):
            uniq = []
            for moves in level:
                b = FastBoard.initial()
                for m in moves:
                    b.play(imove(m))
                key = canonical(b)[0]
                if key not in seen:
                    seen.add(key)
                    uniq.append(moves)
            alt_t = a.alt_time if ply <= a.alt_ply else 0
            res = pool.map(analyse, [(m, a.profile, a.time, alt_t) for m in uniq], chunksize=1)
            nxt = []
            for r in res:
                moves = r["moves"]
                b = FastBoard.initial()
                for m in moves:
                    b.play(imove(m))
                key, flip, mirror = canonical(b)
                if r["best"] == "none":
                    continue
                entry = [[transform_move(r["best"], flip, mirror), 3]]
                children = {r["best"]}
                if "alt" in r and r["alt_score"] >= r["score"] - a.margin:
                    entry.append([transform_move(r["alt"], flip, mirror), 1])
                    children.add(r["alt"])
                book[key] = entry
                info[key] = {"line": " ".join(moves), "best": r["best"], "score": r["score"], "depth": r["depth"],
                             "alt": r.get("alt"), "alt_score": r.get("alt_score")}
                if ply < a.wide_ply:
                    children |= {f"m{sq // 9}{sq % 9}" for sq in b.pawn_moves(b.turn)}
                # Seed lines are always expanded (so an opponent playing them meets the stored
                # refutation); they enter the book as OUR move only if the search agrees (best/alt).
                legal = {f"m{x[1] // 9}{x[1] % 9}" if x[0] == "p" else f"{x[0]}{x[1] // 8}{x[1] % 8}"
                         for x in b.legal_moves()}
                for s in SEEDS:
                    if tuple(s[:ply]) == tuple(moves) and ply < len(s) and s[ply] in legal:
                        children.add(s[ply])
                for c in children:
                    if abs(r["score"]) >= WIN_THRESHOLD:
                        break
                    nxt.append(moves + [c])
            print(f"ply {ply}: {len(uniq)} positions analysed, book {len(book)}, {time.time() - t0:.0f}s", flush=True)
            level = nxt
            if not level:
                break
    BOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"profile": a.profile, "time": a.time, "max_ply": a.max_ply, "positions": book, "info": info},
                  f, indent=0)
    print("wrote", a.out, len(book), "positions")


if __name__ == "__main__":
    main()
