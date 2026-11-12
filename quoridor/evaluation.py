"""Evaluation readouts: search scores as ``+1.5`` / ``M3`` / ``-M5``.

An :class:`Eval` is the *backed-up search score* of a position (the value the
alpha-beta search returned for the root), never a static heuristic. It is
always stored from one player's point of view (``player``): positive is good
for that player.

* Normal scores are in **steps**: the engine's evaluation divided by its
  ``dist_weight`` (10 points = 1 step of shortest-path difference; the wall
  reserve and tempo terms are scaled the same way). ``+1.5`` means the player
  is about one and a half steps ahead in the race.
* Forced results use chess-like notation. ``M3``: the player reaches its goal
  in 3 of its own moves whatever the opponent does. ``-M5``: the opponent
  reaches its goal in 5 of the opponent's moves whatever the player does.

:class:`BackgroundEvaluator` runs a bot on every new position in a worker
thread with a strict time limit (1 s by default); the GUI uses it to show an
evaluation in Human vs Human games.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional

from .state import BoardState

#: Eval-engine choices for Human vs Human (key, label). C++ choices fall back to
#: the Python search when the engine has not been built.
EVAL_ENGINES: tuple[tuple[str, str], ...] = (
    ("attack", "C++ attack only"),
    ("both", "C++ attack + defense"),
    ("all", "C++ all walls"),
    ("py", "Python bot"),
)
EVAL_TIME_LIMIT = 1.0  # seconds per position, Human vs Human analysis


@dataclass(frozen=True)
class Eval:
    """A search score from ``player``'s point of view."""

    player: int                     # 0 = Red, 1 = Blue: positive is good for this player
    steps: Optional[float] = None   # value in steps (None for a forced result)
    mate: Optional[int] = None      # +n: player wins in n own moves, -n: opponent wins in n moves, 0: game won
    depth: int = 0
    nodes: int = 0
    seconds: float = 0.0
    source: str = ""                # who computed it (shown in the side panel)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_info(cls, info: Optional[Mapping[str, Any]], player: int, source: str = "") -> Optional["Eval"]:
        """Build from a bot's ``last_info`` (C++ or Python alpha-beta); None if it has no score."""
        if not info:
            return None
        common = dict(player=player, depth=int(info.get("depth_reached", 0)), nodes=int(info.get("nodes", 0)),
                      seconds=float(info.get("seconds", 0.0)), source=source)
        forced, plies = info.get("forced"), info.get("plies")
        if forced in ("win", "loss") and plies:
            return cls(mate=mate_from_plies(forced == "win", int(plies)), **common)
        if info.get("eval_steps") is not None:
            return cls(steps=float(info["eval_steps"]), **common)
        return None

    @classmethod
    def game_won(cls, winner: int) -> "Eval":
        return cls(player=winner, mate=0)

    # ------------------------------------------------------------------ #
    def for_player(self, player: int) -> "Eval":
        """The same score seen from ``player``'s side."""
        if player == self.player:
            return self
        return replace(self, player=player, steps=None if self.steps is None else -self.steps,
                       mate=None if self.mate is None else -self.mate)

    @property
    def winner(self) -> Optional[int]:
        """Player with a forced win, if any."""
        if self.mate is None:
            return None
        return self.player if self.mate >= 0 else 1 - self.player

    def leader(self) -> Optional[int]:
        """Player the score favours (None if dead equal)."""
        if self.mate is not None:
            return self.winner
        if self.steps is None or round(self.steps, 1) == 0:
            return None
        return self.player if self.steps > 0 else 1 - self.player

    def text(self) -> str:
        """``+1.5``, ``-0.8``, ``0.0``, ``M3``, ``-M5`` (or ``Won`` once the game is over)."""
        if self.mate is not None:
            if self.mate == 0:
                return "Won"
            return f"M{self.mate}" if self.mate > 0 else f"-M{-self.mate}"
        value = round(self.steps or 0.0, 1)
        return "0.0" if value == 0 else f"{value:+.1f}"

    def red_share(self) -> float:
        """Fraction of an eval bar to fill in Red (0 = Blue completely winning, 1 = Red)."""
        red = self.for_player(0)
        if red.mate is not None:
            return 1.0 if red.mate >= 0 else 0.0
        s = red.steps or 0.0
        return min(0.97, max(0.03, 0.5 + 0.5 * s / (abs(s) + 4.0)))


def mate_from_plies(win: bool, plies: int) -> int:
    """Signed mate distance from a ply-adjusted search score.

    A win ``plies`` half-moves away (counting the side to move's move as 1)
    takes ``(plies + 1) // 2`` of the winner's moves; a loss ``plies`` away
    takes ``plies // 2`` of the opponent's moves.
    """
    return (plies + 1) // 2 if win else -(plies // 2)


def eval_spec(engine: str = "both"):
    """The BotSpec used for Human vs Human analysis with the chosen engine."""
    from .bots.factory import BotSpec

    if engine == "py":
        return BotSpec(depth=30, time_limit=EVAL_TIME_LIMIT, engine="py")
    from .bots.cpp_engine import wall_filter_from_name

    return BotSpec(depth=30, time_limit=EVAL_TIME_LIMIT).with_wall_filter(wall_filter_from_name(engine))


def eval_engine_label(engine: str) -> str:
    return dict(EVAL_ENGINES).get(engine, engine)


class BackgroundEvaluator:
    """Evaluates positions in a worker thread, one at a time, newest first.

    ``submit(state)`` replaces any position still waiting; a search that is
    already running finishes (it is capped at ``time_limit`` seconds) and its
    result is reported with the move number it belongs to, so the caller can
    ignore stale results. ``on_result(eval, move_number)`` is called from the
    worker thread.
    """

    def __init__(self, engine: str, on_result: Callable[[Eval, int], None],
                 time_limit: float = EVAL_TIME_LIMIT) -> None:
        from .bots.cpp_engine import engine_available

        if engine != "py" and not engine_available():
            engine = "py"  # the C++ choices fall back to the Python search
        self.engine = engine
        self.label = eval_engine_label(engine)
        self.short_label = {"attack": "C++ attack", "both": "C++ att+def", "all": "C++ all walls",
                            "py": "Python"}.get(engine, engine)
        self.time_limit = time_limit
        self._on_result = on_result
        self._cond = threading.Condition()
        self._pending: Optional[BoardState] = None
        self._closed = False
        self._bot = None
        self.busy = False
        self._thread = threading.Thread(target=self._run, name="eval", daemon=True)
        self._thread.start()

    def submit(self, state: BoardState) -> None:
        if state.winner is not None:
            return
        with self._cond:
            self._pending = state
            self._cond.notify()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._pending = None
            self._cond.notify()
        bot, self._bot = self._bot, None
        if bot is not None and hasattr(bot, "close"):
            bot.close()  # kills a running C++ search at once

    def _make_bot(self):
        from .bots.factory import make_player

        spec = replace(eval_spec(self.engine), time_limit=self.time_limit)
        return make_player(spec, "Eval")

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._closed:
                    self._cond.wait()
                if self._closed:
                    return
                state, self._pending = self._pending, None
            try:
                if self._bot is None:
                    self._bot = self._make_bot()
                bot = self._bot
                self.busy = True
                bot.get_move(state)
                ev = Eval.from_info(bot.last_info, state.current_player, self.label)
            except Exception:  # engine closed mid-search, no legal move, ...
                ev = None
            finally:
                self.busy = False
            if ev is not None and not self._closed:
                self._on_result(ev, state.move_number)
