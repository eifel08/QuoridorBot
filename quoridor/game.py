"""Game engine: runs a match between two :class:`Player` objects.

The engine is agnostic to player type and to presentation. Players receive an
immutable :class:`~quoridor.state.BoardState`; views (CLI, GUI, loggers)
subscribe as :class:`GameObserver` objects and are only ever given snapshots,
so nothing outside the engine can mutate the authoritative :class:`Board`.
"""

from __future__ import annotations

from typing import Iterable, Optional

from .board import Board
from .moves import Move
from .player import Player, QuitGame
from .state import BoardState


class GameObserver:
    """Receives game events. All hooks are optional no-ops."""

    def on_game_start(self, game: "Game", state: BoardState) -> None:
        pass

    def on_turn_start(self, game: "Game", state: BoardState) -> None:
        pass

    def on_invalid_move(self, game: "Game", player: int, move: Move, reason: str) -> None:
        pass

    def on_move(self, game: "Game", player: int, move: Move, before: BoardState, after: BoardState) -> None:
        pass

    def on_game_over(self, game: "Game", state: BoardState) -> None:
        pass

    def on_abort(self, game: "Game") -> None:
        pass


class Game:
    """Drives a Quoridor match."""

    def __init__(
        self,
        player1: Player,
        player2: Player,
        observers: Iterable[GameObserver] = (),
        max_invalid_attempts: Optional[int] = None,
    ) -> None:
        self.players = (player1, player2)
        for i, p in enumerate(self.players):
            p.set_index(i)
        self._board = Board()
        self.current = 0
        self.winner: Optional[int] = None
        self.history: list[tuple[int, Move]] = []
        self.observers: list[GameObserver] = list(observers)
        self._max_invalid = max_invalid_attempts
        self._state = self._make_state()

    # ------------------------------------------------------------------ #
    @property
    def names(self) -> tuple[str, str]:
        return (self.players[0].name, self.players[1].name)

    @property
    def state(self) -> BoardState:
        """Current immutable snapshot of the position."""
        return self._state

    def _make_state(self) -> BoardState:
        return self._board.snapshot(self.current, len(self.history), self.winner)

    def _emit(self, hook: str, *args) -> None:
        for obs in self.observers:
            getattr(obs, hook)(self, *args)

    # ------------------------------------------------------------------ #
    def play_turn(self) -> bool:
        """Ask the current player for moves until a legal one is applied.

        Returns True if the move ended the game.
        """
        mover = self.current
        player = self.players[mover]
        before = self._state
        self._emit("on_turn_start", before)
        attempts = 0
        while True:
            move = player.get_move(before)
            error = self._board.move_error(mover, move)
            if error is None:
                break
            player.notify_invalid_move(move, error)
            self._emit("on_invalid_move", mover, move, error)
            attempts += 1
            if self._max_invalid is not None and attempts >= self._max_invalid:
                raise RuntimeError(f"{player.name} made too many invalid moves.")

        self._board.apply_move(mover, move)
        self.history.append((mover, move))
        if self._board.has_won(mover):
            self.winner = mover
        else:
            self.current = 1 - mover
        self._state = self._make_state()

        self._emit("on_move", mover, move, before, self._state)
        for p in self.players:
            p.observe(self._state)
        return self.winner is not None

    def run(self) -> Optional[int]:
        """Play until someone wins. Returns the winner index, or None if aborted."""
        self._emit("on_game_start", self._state)
        try:
            while not self.play_turn():
                pass
        except QuitGame:
            self._emit("on_abort")
            return None
        self._emit("on_game_over", self._state)
        return self.winner
