"""Fast game runner for training and evaluation (no PyTorch needed).

Games are played on the bot's :class:`FastBoard` instead of the GUI engine,
because evaluation needs thousands of games. Every game starts from a
*seeded random opening* (a few random legal moves), because the Depth-4
alpha-beta bot is deterministic: without varied openings every evaluation
game would be identical and the win rate would always be exactly 0% or 100%.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional

from ..bots.alphabeta import FastBoard, search_board

IMove = tuple[str, int]
# A "policy" picks a move for the side to move. It gets a copy of the board.
Policy = Callable[[FastBoard], IMove]

MAX_PLIES = 200  # games longer than this are scored as draws


def alphabeta_policy(depth: int = 4, rng: Optional[random.Random] = None) -> Policy:
    """The existing alpha-beta bot as a policy (no time limit => fixed strength)."""
    def policy(board: FastBoard) -> IMove:
        move, _ = search_board(board, depth, None, rng)
        return move
    return policy


def random_opening(rng: random.Random, max_plies: int = 6, wall_prob: float = 0.15) -> FastBoard:
    """Start position followed by 0..max_plies random moves (mostly pawn steps)."""
    board = FastBoard.initial()
    for _ in range(rng.randint(0, max_plies)):
        me = board.turn
        walls = board.legal_walls(me) if rng.random() < wall_prob else []
        moves = [("p", sq) for sq in board.pawn_moves(me)]
        move = rng.choice(walls) if walls else rng.choice(moves)
        if board.play(move) is not None:  # never happens this early, but be safe
            return FastBoard.initial()
    return board


@dataclass
class GameResult:
    winner: Optional[int]  # 0, 1, or None for a draw (move cap)
    plies: int
    moves: list[IMove]


def play_game(policies: tuple[Policy, Policy], board: FastBoard,
              max_plies: int = MAX_PLIES) -> GameResult:
    """Play until someone wins; ``policies[i]`` moves for player ``i``."""
    moves: list[IMove] = []
    for ply in range(max_plies):
        move = policies[board.turn](board.copy())
        moves.append(move)
        winner = board.play(move)
        if winner is not None:
            return GameResult(winner, ply + 1, moves)
    return GameResult(None, max_plies, moves)


def match(agent: Policy, opponent: Policy, n_games: int, seed: int = 0,
          opening_plies: int = 6, progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """Play ``n_games`` from seeded openings, agent alternating Red/Blue.

    The same ``seed`` always produces the same set of openings, so successive
    evaluations of different checkpoints are directly comparable.
    """
    wins = losses = draws = 0
    plies = 0
    for i in range(n_games):
        rng = random.Random(seed * 1_000_003 + i)
        board = random_opening(rng, opening_plies)
        agent_side = i % 2
        pols = (agent, opponent) if agent_side == 0 else (opponent, agent)
        result = play_game(pols, board)
        plies += result.plies
        if result.winner is None:
            draws += 1
        elif result.winner == agent_side:
            wins += 1
        else:
            losses += 1
        if progress:
            progress(i + 1, wins)
    return {
        "games": n_games, "wins": wins, "losses": losses, "draws": draws,
        "win_rate": wins / n_games if n_games else 0.0,
        "wilson_low": wilson_lower_bound(wins, n_games),
        "avg_plies": plies / n_games if n_games else 0.0,
    }


def wilson_lower_bound(wins: int, n: int, z: float = 1.96) -> float:
    """95% lower confidence bound on the true win rate."""
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return (centre - margin) / denom
