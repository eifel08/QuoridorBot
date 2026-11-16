"""Engine analysis of detected positions, in a background thread.

:class:`Analyzer` takes :class:`~quoridor.vision.position.Position` objects
and reports an :class:`Analysis`: the bot's best move with its backed-up
search score (``+1.5`` / ``M3`` / ``-M5``, see :mod:`quoridor.evaluation`) and
a few alternative moves, each with the reply the engine expects.

Any bot spec works (``ab:depth=30,time=1``, ``ab:depth=8,walls=attack``,
``ab:depth=4,engine=py``, ``ml:model=best.pt``). The spec can be changed while
running. A new position cancels the search of the previous one.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

from ..bots.factory import BotSpec, make_player
from ..evaluation import Eval
from ..moves import Move, PawnMove
from .position import Position, move_notation

DEFAULT_SPEC = BotSpec(depth=30, time_limit=1.0)
MAX_ALTERNATIVES = 3


@dataclass
class Line:
    move: Move
    notation: str
    eval: Optional[Eval] = None      # from the mover's point of view
    reply: Optional[str] = None      # the opponent's expected answer


@dataclass
class Analysis:
    position: Position
    lines: list[Line] = field(default_factory=list)   # best move first
    depth: int = 0
    nodes: int = 0
    seconds: float = 0.0
    engine: str = ""
    complete: bool = False           # alternatives finished too
    error: str = ""

    @property
    def best(self) -> Optional[Line]:
        return self.lines[0] if self.lines else None

    @property
    def eval(self) -> Optional[Eval]:
        return self.lines[0].eval if self.lines else None


def _rank(ev: Optional[Eval]) -> float:
    """Sort key: higher is better for the side the eval belongs to."""
    if ev is None:
        return -1e9
    if ev.mate is not None:
        return 1e6 - ev.mate if ev.mate > 0 else -1e6 - ev.mate
    return ev.steps or 0.0


class Analyzer:
    """Searches the newest submitted position; older ones are dropped."""

    def __init__(self, spec: BotSpec = DEFAULT_SPEC, on_result: Optional[Callable[[Analysis], None]] = None,
                 alternatives: int = MAX_ALTERNATIVES) -> None:
        self.spec = spec
        self.alternatives = alternatives
        self._on_result = on_result or (lambda analysis: None)
        self._cond = threading.Condition()
        self._pending: Optional[Position] = None
        self._generation = 0
        self._closed = False
        self._bots: dict = {}
        self.busy = False
        self._thread = threading.Thread(target=self._run, name="assistant-analysis", daemon=True)
        self._thread.start()

    # -- control ----------------------------------------------------------
    def submit(self, position: Position) -> None:
        with self._cond:
            self._pending = position
            self._generation += 1
            self._cond.notify()
            if self.busy:  # the worker is on an older position (it takes new ones under this lock)
                self._abort()

    def set_spec(self, spec: BotSpec, position: Optional[Position] = None) -> None:
        """Use another bot from now on; ``position`` is searched again with it."""
        with self._cond:
            self.spec = spec
            self._generation += 1
            if position is not None:
                self._pending = position
            self._cond.notify()
            self._abort(drop=True)

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._pending = None
            self._cond.notify()
        self._abort(drop=True)

    def _abort(self, drop: bool = False) -> None:
        """Stop a running C++ search (kills its engine process; it restarts on demand)."""
        for bot in list(self._bots.values()):
            if hasattr(bot, "close"):
                try:
                    bot.close()
                except Exception:
                    pass
        if drop:
            self._bots = {}

    # -- worker -------------------------------------------------------------
    def _bot(self, role: str, spec: BotSpec):
        key = (role, spec)
        bot = self._bots.get(key)
        if bot is None:
            bot = self._bots[key] = make_player(spec, "Assistant")
        return bot

    def _stale(self, generation: int) -> bool:
        return self._closed or generation != self._generation

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._closed:
                    self._cond.wait()
                if self._closed:
                    return
                position, self._pending = self._pending, None
                generation, spec = self._generation, self.spec
                self.busy = True
            try:
                self._analyse(position, spec, generation)
            except Exception as exc:  # engine killed by a newer position, missing model, ...
                if not self._stale(generation):
                    self._on_result(Analysis(position, error=str(exc) or type(exc).__name__, complete=True))
            finally:
                with self._cond:
                    self.busy = False

    def _analyse(self, position: Position, spec: BotSpec, generation: int) -> None:
        if position.turn is None or position.winner is not None:
            return
        me = position.turn
        state = position.to_state()
        bot = self._bot("main", spec)
        start = time.perf_counter()
        move = bot.get_move(state)
        info = getattr(bot, "last_info", None) or {}
        analysis = Analysis(position, [Line(move, move_notation(move), Eval.from_info(info, me, spec.label()))],
                            depth=int(info.get("depth_reached", 0)), nodes=int(info.get("nodes", 0)),
                            seconds=time.perf_counter() - start, engine=_engine_name(spec, info))
        if self._stale(generation):
            return
        self._on_result(replace(analysis, lines=list(analysis.lines)))
        if spec.kind != "ab" or self.alternatives <= 0:
            analysis.complete = True
            self._on_result(analysis)
            return

        # Alternatives: search the position after each candidate move with a
        # smaller budget; the score for the opponent, negated, rates the move.
        candidates = self._candidates(state, bot, move)
        budget = max(0.05, (spec.time_limit or 2.0) / max(1, len(candidates) + 1) * 0.6)
        child_spec = replace(spec, depth=max(1, spec.depth - 1), time_limit=budget, random_ties=False)
        child_bot = self._bot("child", child_spec)
        lines = []
        for cand in [move, *candidates]:
            if self._stale(generation):
                return
            board = position.to_board()
            board.apply_move(me, cand)
            line = Line(cand, move_notation(cand))
            if isinstance(cand, PawnMove) and cand.row == board.goal_row(me):
                line.eval = Eval(me, mate=1)
            else:
                child = board.snapshot(1 - me, move_number=3)
                reply = child_bot.get_move(child)
                line.reply = move_notation(reply)
                ev = Eval.from_info(getattr(child_bot, "last_info", None), 1 - me)
                if ev is not None:
                    ev = ev.for_player(me)
                    if ev.mate is not None and ev.mate > 0:
                        ev = replace(ev, mate=ev.mate + 1)  # plus the candidate move itself
                    line.eval = ev
            lines.append(line)
        if self._stale(generation):
            return
        best = lines[0]
        best.eval = analysis.lines[0].eval or best.eval  # the full-depth score stays authoritative
        others = sorted(lines[1:], key=lambda ln: _rank(ln.eval), reverse=True)[: self.alternatives]
        analysis.lines = [best, *others]
        analysis.seconds = time.perf_counter() - start
        analysis.complete = True
        self._on_result(analysis)

    def _candidates(self, state, bot, best: Move) -> list[Move]:
        """Moves worth comparing with the best one: every pawn move and the search's first walls."""
        moves: list[Move] = [PawnMove(r, c) for r, c in state.valid_pawn_moves]
        engine = getattr(bot, "engine", None)
        if engine is not None:
            try:
                engine.set_position(state)
                for m in engine.search_order():
                    if m not in moves:
                        moves.append(m)
            except Exception:
                pass
        out = [m for m in moves if m != best]
        return out[: max(self.alternatives + 3, 6)]


def _engine_name(spec: BotSpec, info: dict) -> str:
    if spec.kind == "ml":
        return "ML agent"
    return "C++ engine" if info.get("engine") == "cpp" else "Python search"
