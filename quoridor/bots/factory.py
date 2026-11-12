"""Describe a bot with a short spec string and build it.

Spec syntax (used by ``main.py --bot1/--bot2`` and ``python -m quoridor.botmatch``)::

    ab                      alpha-beta, depth 4, no time limit
    ab:depth=2              alpha-beta, depth 2
    ab:depth=8,time=2       alpha-beta, depth 8, at most 2 s per move
    ab:depth=4,random=1     alpha-beta with random tie-breaking (varied games)
    ml:model=best.pt        trained ML agent (needs PyTorch), 64 MCTS simulations
    ml:model=best.pt,sims=0 ML agent playing its raw policy (fast)

``alphabeta`` may be written instead of ``ab``. Unknown keys are rejected so typos
don't go unnoticed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional

from ..player import Player
from .alphabeta import MAX_DEPTH, AlphaBetaBot, FastBoard, search_board

IMove = tuple[str, int]


@dataclass(frozen=True)
class BotSpec:
    kind: str = "ab"                   # "ab" (alpha-beta) or "ml"
    depth: int = 4                     # alpha-beta search depth, 1..30
    time_limit: Optional[float] = None  # seconds per move (alpha-beta), None = no limit
    random_ties: bool = False          # alpha-beta: break ties between equal moves at random
    model: Optional[str] = None        # ml: checkpoint path
    sims: int = 64                     # ml: MCTS simulations per move

    def label(self) -> str:
        if self.kind == "ml":
            return f"ML ({self.sims} sims)"
        t = f", {self.time_limit:g}s" if self.time_limit else ""
        return f"Alpha-beta d{self.depth}{t}"

    def to_string(self) -> str:
        if self.kind == "ml":
            return f"ml:model={self.model},sims={self.sims}"
        parts = [f"depth={self.depth}"]
        if self.time_limit:
            parts.append(f"time={self.time_limit:g}")
        if self.random_ties:
            parts.append("random=1")
        return "ab:" + ",".join(parts)


def parse_bot_spec(text: str) -> BotSpec:
    """Parse e.g. ``"ab:depth=2,time=1"`` (see module docstring)."""
    text = text.strip()
    kind, _, rest = text.partition(":")
    kind = {"alphabeta": "ab", "alpha-beta": "ab"}.get(kind.lower(), kind.lower())
    if kind not in ("ab", "ml"):
        raise ValueError(f"unknown bot type '{kind}' (use 'ab' or 'ml')")
    opts: dict[str, str] = {}
    for item in filter(None, (p.strip() for p in rest.split(","))):
        key, eq, value = item.partition("=")
        if not eq:
            raise ValueError(f"expected key=value, got '{item}'")
        opts[key.strip().lower()] = value.strip()

    allowed = {"ab": {"depth", "time", "random"}, "ml": {"model", "sims"}}[kind]
    unknown = set(opts) - allowed
    if unknown:
        raise ValueError(f"unknown option(s) for '{kind}': {', '.join(sorted(unknown))} "
                         f"(allowed: {', '.join(sorted(allowed))})")
    try:
        if kind == "ab":
            depth = int(opts.get("depth", 4))
            if not 1 <= depth <= MAX_DEPTH:
                raise ValueError(f"depth must be 1-{MAX_DEPTH}")
            t = float(opts.get("time", 0)) or None
            rnd = opts.get("random", "0").lower() in ("1", "true", "yes")
            return BotSpec("ab", depth=depth, time_limit=t, random_ties=rnd)
        if "model" not in opts:
            raise ValueError("ml bots need model=PATH")
        return BotSpec("ml", model=opts["model"], sims=int(opts.get("sims", 64)))
    except ValueError as exc:
        raise ValueError(f"bad bot spec '{text}': {exc}") from None


def make_player(spec: BotSpec, name: Optional[str] = None) -> Player:
    """A :class:`Player` for the normal game engine (GUI / CLI)."""
    name = name or spec.label()
    if spec.kind == "ml":
        from ..ml.agent import MLPlayer

        return MLPlayer(spec.model, name, spec.sims)  # type: ignore[arg-type]
    bot = AlphaBetaBot(name, spec.depth, spec.time_limit)
    if spec.random_ties:
        bot.rng = random.Random()
    return bot


def make_policy(spec: BotSpec, seed: int = 0) -> Callable[[FastBoard], IMove]:
    """A fast ``FastBoard -> move`` function for headless batch matches."""
    if spec.kind == "ml":
        import numpy as np

        from ..ml.mcts import agent_move
        from ..ml.model import Evaluator, load_model

        net, ckpt = load_model(spec.model, "cpu")  # type: ignore[arg-type]
        ev = Evaluator(net, "cpu", ckpt.get("config", {}).get("heuristic_mix", 0.0))
        rng = np.random.default_rng(seed)
        return lambda board: agent_move(ev, board, spec.sims, 0.0, rng)[0]
    rng = random.Random(seed) if spec.random_ties else None
    return lambda board: search_board(board, spec.depth, spec.time_limit, rng)[0]
