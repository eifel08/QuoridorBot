"""Game generation for training and evaluation (runs in worker processes).

Three kinds of games, all against / with the alpha-beta bot:

* ``teacher_game``  - alpha-beta vs alpha-beta (random tie-breaks). Used once
  at the start to warm up the network by imitation, so RL does not start
  from random play.
* ``training_game`` - the agent (MCTS + network, with exploration noise) vs
  the Depth-4 bot. Produces the RL training data.
* ``eval_game``     - the agent (greedy, no noise) vs the deterministic
  Depth-4 bot from a seeded opening. Used for the win-rate metric.

A training sample is ``(planes, policy_target, value_target)`` seen from
the side to move; the value target is +1 if that side went on to win,
-1 if it lost and 0 for a draw (move cap).
"""

from __future__ import annotations

import io
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ..bots.alphabeta import FastBoard, search_board
from .arena import MAX_PLIES, random_opening
from .encoding import N_ACTIONS, N_PLANES, encode, move_to_action
from .mcts import agent_move
from .model import Evaluator, ModelConfig, PolicyValueNet

IMove = tuple[str, int]


@dataclass
class GameSettings:
    opponent_depth: int = 4
    teacher_depth: int = 4
    sims: int = 64
    eval_sims: int = 64
    temp_moves: int = 8          # agent samples moves (temperature 1) for its first N moves
    opening_plies: int = 6
    learn_opponent_moves: bool = True  # also imitate the bot's moves in training games
    opponent_random_ties: float = 0.5  # share of training games where the bot breaks ties randomly


# ---------------------------------------------------------------------- #
# Worker-side model cache
# ---------------------------------------------------------------------- #
_worker_model: dict = {"version": None, "evaluator": None}


def worker_init() -> None:
    torch.set_num_threads(1)


def serialize_model(net: PolicyValueNet) -> bytes:
    buf = io.BytesIO()
    torch.save({"cfg": vars(net.cfg), "state": {k: v.cpu() for k, v in net.state_dict().items()}}, buf)
    return buf.getvalue()


def _evaluator(version: int, blob: bytes) -> Evaluator:
    if _worker_model["version"] != version:
        data = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
        net = PolicyValueNet(ModelConfig(**data["cfg"]))
        net.load_state_dict(data["state"])
        _worker_model.update(version=version, evaluator=Evaluator(net, "cpu"))
    return _worker_model["evaluator"]


# ---------------------------------------------------------------------- #
def _one_hot(move: IMove, board: FastBoard) -> np.ndarray:
    pi = np.zeros(N_ACTIONS, dtype=np.float32)
    pi[move_to_action(move, board.turn == 1)] = 1
    return pi


def _finish(records: list[tuple[np.ndarray, np.ndarray, int]], winner: Optional[int]):
    planes = np.stack([r[0] for r in records]).astype(np.float16) if records else np.zeros((0, N_PLANES, 9, 9), np.float16)
    pis = np.stack([r[1] for r in records]).astype(np.float16) if records else np.zeros((0, N_ACTIONS), np.float16)
    z = np.array([0.0 if winner is None else (1.0 if side == winner else -1.0) for _, _, side in records],
                 dtype=np.float32)
    return planes, pis, z


def teacher_game(seed: int, gs: GameSettings):
    rng = random.Random(seed)
    board = random_opening(rng, gs.opening_plies)
    depths = (gs.teacher_depth, gs.opponent_depth) if seed % 2 == 0 else (gs.opponent_depth, gs.teacher_depth)
    records = []
    winner = None
    for _ in range(MAX_PLIES):
        move, _ = search_board(board, depths[board.turn], None, rng)
        records.append((encode(board), _one_hot(move, board), board.turn))
        winner = board.play(move)
        if winner is not None:
            break
    return _finish(records, winner) + ({"winner": winner, "plies": len(records)},)


def training_game(seed: int, version: int, blob: bytes, gs: GameSettings):
    ev = _evaluator(version, blob)
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    board = random_opening(rng, gs.opening_plies)
    agent_side = seed % 2
    bot_rng = rng if rng.random() < gs.opponent_random_ties else None
    records = []
    winner = None
    agent_moves = 0
    for _ in range(MAX_PLIES):
        if board.turn == agent_side:
            temp = 1.0 if agent_moves < gs.temp_moves else 0.0
            move, pi = agent_move(ev, board, gs.sims, temp, nrng, add_noise=True)
            records.append((encode(board), pi, board.turn))
            agent_moves += 1
        else:
            move, _ = search_board(board, gs.opponent_depth, None, bot_rng)
            if gs.learn_opponent_moves:
                records.append((encode(board), _one_hot(move, board), board.turn))
        winner = board.play(move)
        if winner is not None:
            break
    info = {"winner": winner, "agent_won": winner == agent_side, "plies": len(records)}
    return _finish(records, winner) + (info,)


def eval_game(index: int, eval_seed: int, version: int, blob: bytes, gs: GameSettings) -> dict:
    """One evaluation game, same openings as ``arena.match`` for a given seed."""
    ev = _evaluator(version, blob)
    rng = random.Random(eval_seed * 1_000_003 + index)
    board = random_opening(rng, gs.opening_plies)
    agent_side = index % 2
    nrng = np.random.default_rng(index)
    winner = None
    plies = 0
    for plies in range(1, MAX_PLIES + 1):
        if board.turn == agent_side:
            move, _ = agent_move(ev, board, gs.eval_sims, 0.0, nrng, add_noise=False)
        else:
            move, _ = search_board(board, gs.opponent_depth, None, None)
        winner = board.play(move)
        if winner is not None:
            break
    return {"winner": winner, "agent_side": agent_side, "won": winner == agent_side, "plies": plies}


# Thin wrappers so Pool.imap can call them with a single tuple argument.
def run_teacher(args):
    return teacher_game(*args)


def run_training(args):
    return training_game(*args)


def run_eval(args):
    return eval_game(*args)


def default_workers() -> int:
    return max(1, (os.cpu_count() or 1))
