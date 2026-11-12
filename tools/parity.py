"""Rules parity: the C++ engine must generate exactly the Python engine's legal moves.

    python -m tools.parity --games 300

Plays random games with the Python rules engine (quoridor/board.py). At every
position it compares, between Python and C++ (wall filtering is not involved:
``moves`` lists every legal move):

* the set of legal pawn moves and legal walls,
* both players' shortest-path lengths,
* the Zobrist hash after replaying the game move by move inside the engine
  (incremental make) against the hash of the same position set up from scratch.
"""

from __future__ import annotations

import argparse
import random
import time

from quoridor import Board, PawnMove, WallMove
from quoridor.bots.cpp_engine import CppEngine, position_command


def board_command(board: Board, player: int) -> str:
    return position_command(player, board.pawns, board.walls_left, ((r, c, o.value) for r, c, o in board.walls))


def python_moves(board: Board, player: int) -> set:
    moves: set = {PawnMove(r, c) for r, c in board.valid_pawn_moves(player)}
    moves.update(WallMove(r, c, o) for r, c, o in board.valid_walls(player))
    return moves


def check_games(engine: CppEngine, games: int, seed: int = 0, wall_prob: float = 0.45,
                max_plies: int = 150) -> dict:
    """Raises AssertionError on the first difference; returns statistics."""
    rng = random.Random(seed)
    replay = CppEngine(engine.path)  # second process: follows the game with make()
    positions = moves_compared = 0
    try:
        for game in range(games):
            board, player = Board(), 0
            replay.set_position(board_command(board, player))
            p_wall = rng.choice((0.15, wall_prob, 0.8))  # open boards as well as dense mazes
            for ply in range(max_plies):
                engine.set_position(board_command(board, player))
                expected = python_moves(board, player)
                got = engine.legal_moves()
                where = f"game {game}, ply {ply}: {board_command(board, player)}"
                assert len(got) == len(set(got)), f"duplicate moves, {where}"
                assert set(got) == expected, (
                    f"legal moves differ, {where}\n  only Python: {sorted(map(str, expected - set(got)))}"
                    f"\n  only C++: {sorted(map(str, set(got) - expected))}")
                dist = (board.shortest_path_length(0), board.shortest_path_length(1))
                assert engine.distances() == dist, f"distances differ, {where}"
                assert replay.command("hash") == engine.command("hash"), f"incremental hash differs, {where}"
                positions += 1
                moves_compared += len(got)

                walls = sorted((m for m in expected if isinstance(m, WallMove)), key=str)
                pawns = sorted((m for m in expected if isinstance(m, PawnMove)), key=str)
                if not expected:
                    break
                move = rng.choice(walls) if walls and (not pawns or rng.random() < p_wall) else rng.choice(pawns)
                board.apply_move(player, move)
                if board.has_won(player):
                    break
                replay.play(move)
                player = 1 - player
    finally:
        replay.close()
    return {"games": games, "positions": positions, "moves": moves_compared}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    t0 = time.time()
    with CppEngine() as engine:
        print("engine self-test:", engine.command("selftest 200 1"))
        stats = check_games(engine, a.games, a.seed)
    print(f"OK: {stats['games']} random games, {stats['positions']} positions, {stats['moves']} legal moves "
          f"identical in Python and C++ ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
