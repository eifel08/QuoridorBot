"""Test positions shared by the verification, benchmark and match tools."""

from __future__ import annotations

import random

from quoridor.bots.cpp_engine import CppEngine, format_move
from quoridor.moves import PawnMove

START = "position 0 8 4 0 4 10 10"

#: Fixed benchmark positions (``position`` commands of the engine protocol).
BENCH_POSITIONS: dict[str, str] = {
    # Start position: both players have all 10 walls.
    "opening": START,
    # Move 21 of an engine self-play game: 9 walls on the board, 6 and 5 walls left.
    "middlegame": "position 0 5 3 5 5 6 5 h30 h32 h34 h36 h47 h53 h55 v35 v42",
    # Move 37 of the same game: 18 walls on the board, 2 and 0 walls left, long winding paths.
    "endgame": "position 0 5 5 5 6 2 0 h03 h05 h07 h30 h32 h34 h36 h47 h53 h55 h63 h66 h74 v14 v35 v42 v57 v75",
}


def random_position(engine: CppEngine, rng: random.Random, plies: int, wall_prob: float = 0.4) -> str:
    """Play ``plies`` random legal moves from the start; returns the ``position`` command.

    Restarts if the game ends early, so the result is never a finished game.
    """
    while True:
        pawns = [(8, 4), (0, 4)]
        left = [10, 10]
        walls: list[str] = []
        turn = 0
        engine.set_position(START)
        ok = True
        for _ in range(plies):
            moves = engine.legal_moves()
            pawn = [m for m in moves if isinstance(m, PawnMove)]
            wall = [m for m in moves if not isinstance(m, PawnMove)]
            if not moves:
                ok = False
                break
            move = rng.choice(wall) if wall and (not pawn or rng.random() < wall_prob) else rng.choice(pawn)
            if isinstance(move, PawnMove):
                pawns[turn] = (move.row, move.col)
                if move.row == (0, 8)[turn]:
                    ok = False
                    break
            else:
                walls.append(format_move(move))
                left[turn] -= 1
            engine.play(move)
            turn = 1 - turn
        if ok and engine.legal_moves():
            (r0, c0), (r1, c1) = pawns
            return " ".join(map(str, ["position", turn, r0, c0, r1, c1, left[0], left[1], *walls]))
