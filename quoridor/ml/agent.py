"""Use a trained network as a normal :class:`~quoridor.player.Player`."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..bots.alphabeta import N, NW, FastBoard
from ..moves import Move, Orientation, PawnMove, WallMove
from ..player import Player
from ..state import BoardState
from .mcts import agent_move
from .model import Evaluator, load_model


def internal_to_move(m: tuple[str, int]) -> Move:
    kind, x = m
    if kind == "p":
        return PawnMove(*divmod(x, N))
    r, c = divmod(x, NW)
    return WallMove(r, c, Orientation(kind))


class MLPlayer(Player):
    """Plays with a trained policy-value network (+ MCTS when ``sims > 0``).

    Stateless like every player: each ``get_move`` builds a scratch board
    from the read-only :class:`BoardState` and discards it.
    """

    def __init__(self, model_path: str, name: str = "ML agent", sims: int = 64,
                 device: Optional[str] = None, heuristic_mix: Optional[float] = None) -> None:
        super().__init__(name)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        net, ckpt = load_model(model_path, device)
        if heuristic_mix is None:  # use the setting the model was trained with
            heuristic_mix = ckpt.get("config", {}).get("heuristic_mix", 0.0)
        self.evaluator = Evaluator(net, device, heuristic_mix)
        self.sims = sims
        self.last_info: dict = {}

    def get_move(self, board_state: BoardState) -> Move:
        board = FastBoard(board_state)
        internal, pi = agent_move(self.evaluator, board, self.sims, 0.0, np.random.default_rng(0))
        self.last_info = {"sims": self.sims, "confidence": float(pi.max())}
        return internal_to_move(internal)
