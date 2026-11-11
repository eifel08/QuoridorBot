"""Monte Carlo tree search guided by the policy-value network (PUCT)."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..bots.alphabeta import FastBoard
from .encoding import N_ACTIONS, action_to_move, encode, legal_mask, move_to_action

IMove = tuple[str, int]


class Node:
    __slots__ = ("board", "move", "prior", "children", "visits", "value_sum", "terminal_value")

    def __init__(self, board: Optional[FastBoard], prior: float, move: Optional[IMove] = None) -> None:
        self.board = board  # created lazily on the first visit
        self.move = move
        self.prior = prior
        self.children: Optional[dict[int, "Node"]] = None
        self.visits = 0
        self.value_sum = 0.0  # from the point of view of the player who moved INTO this node
        self.terminal_value: Optional[float] = None

    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class MCTS:
    """PUCT search. ``evaluator(planes, mask) -> (priors over canonical actions, value)``."""

    def __init__(self, evaluator, sims: int = 64, c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3, noise_frac: float = 0.25,
                 rng: Optional[np.random.Generator] = None) -> None:
        self.evaluator = evaluator
        self.sims = sims
        self.c_puct = c_puct
        self.alpha = dirichlet_alpha
        self.noise_frac = noise_frac
        self.rng = rng or np.random.default_rng()

    def _expand(self, node: Node) -> float:
        """Create children; returns the network value for the side to move at ``node``."""
        b = node.board
        assert b is not None
        flip = b.turn == 1
        moves = b.legal_moves()
        priors, value = self.evaluator(encode(b), legal_mask(b, moves))
        node.children = {}
        for m in moves:
            a = move_to_action(m, flip)
            node.children[a] = Node(None, float(priors[a]), m)
        return value

    def _select(self, node: Node) -> Node:
        sqrt_n = math.sqrt(max(node.visits, 1))
        c = self.c_puct
        best, best_score = None, -1e18
        for ch in node.children.values():  # type: ignore[union-attr]
            score = (ch.value_sum / ch.visits if ch.visits else 0.0) + c * ch.prior * sqrt_n / (1 + ch.visits)
            if score > best_score:
                best, best_score = ch, score
        assert best is not None
        if best.board is None:  # materialise the child position on first visit
            parent_board = node.board
            assert parent_board is not None and best.move is not None
            best.board = parent_board.copy()
            if best.board.play(best.move) is not None:
                best.terminal_value = 1.0  # the mover wins
        return best

    def run(self, board: FastBoard, add_noise: bool = False) -> np.ndarray:
        """Search from ``board``; returns visit counts over canonical actions."""
        counts = np.zeros(N_ACTIONS, dtype=np.float32)
        flip = board.turn == 1
        # A winning pawn move needs no search.
        for sq in board.pawn_moves(board.turn):
            if sq // 9 == board.goal[board.turn]:
                counts[move_to_action(("p", sq), flip)] = 1
                return counts
        root = Node(board.copy(), 1.0)
        self._expand(root)
        if not root.children:
            return counts
        if add_noise and len(root.children) > 1:
            noise = self.rng.dirichlet([self.alpha] * len(root.children))
            for ch, n in zip(root.children.values(), noise):
                ch.prior = (1 - self.noise_frac) * ch.prior + self.noise_frac * n

        for _ in range(self.sims):
            node, path = root, [root]
            while node.children is not None and node.terminal_value is None:
                node = self._select(node)
                path.append(node)
            if node.terminal_value is not None:
                value = node.terminal_value  # for the player who moved into node
            else:
                value = -self._expand(node)  # evaluator value is for the side to move at node
            for n in reversed(path):
                n.visits += 1
                n.value_sum += value
                value = -value

        for a, ch in root.children.items():
            counts[a] = ch.visits
        return counts


def pick_action(counts: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    if temperature <= 1e-6 or counts.sum() == 0:
        return int(counts.argmax())
    p = counts ** (1.0 / temperature)
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


def greedy_policy_action(evaluator, board: FastBoard) -> int:
    """No search: the network's most likely legal move."""
    for sq in board.pawn_moves(board.turn):  # always take an immediate win
        if sq // 9 == board.goal[board.turn]:
            return move_to_action(("p", sq), board.turn == 1)
    priors, _ = evaluator(encode(board), legal_mask(board))
    return int(priors.argmax())


def agent_move(evaluator, board: FastBoard, sims: int, temperature: float = 0.0,
               rng: Optional[np.random.Generator] = None, add_noise: bool = False) -> tuple[IMove, np.ndarray]:
    """Choose a move. Returns (internal move, policy target over canonical actions)."""
    rng = rng or np.random.default_rng()
    flip = board.turn == 1
    if sims <= 0:
        a = greedy_policy_action(evaluator, board)
        target = np.zeros(N_ACTIONS, dtype=np.float32)
        target[a] = 1
        return action_to_move(a, flip), target
    counts = MCTS(evaluator, sims, rng=rng).run(board, add_noise=add_noise)
    a = pick_action(counts, temperature, rng)
    return action_to_move(a, flip), counts / max(counts.sum(), 1)
