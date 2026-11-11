"""Game engine: runs a match between two :class:`Player` objects."""

from __future__ import annotations

from typing import Callable, Optional

from .board import Board
from .moves import Move
from .player import Player, QuitGame
from .renderer import PAWN_SYMBOLS, render_board, render_status


class Game:
    """Drives a Quoridor match. Agnostic to player type (human or bot)."""

    def __init__(
        self,
        player1: Player,
        player2: Player,
        output_func: Callable[[str], None] = print,
        show_board: bool = True,
        max_invalid_attempts: Optional[int] = None,
    ) -> None:
        self.players = (player1, player2)
        for i, p in enumerate(self.players):
            p.set_index(i)
        self.board = Board()
        self.current = 0
        self.winner: Optional[int] = None
        self.history: list[tuple[int, Move]] = []
        self._output = output_func
        self._show_board = show_board
        self._max_invalid = max_invalid_attempts

    @property
    def names(self) -> tuple[str, str]:
        return (self.players[0].name, self.players[1].name)

    def display(self) -> None:
        if not self._show_board:
            return
        self._output("")
        self._output(render_board(self.board))
        self._output(render_status(self.board, self.names))

    def play_turn(self) -> bool:
        """Ask the current player for a move until a legal one is applied.

        Returns True if the move ended the game.
        """
        player = self.players[self.current]
        attempts = 0
        while True:
            move = player.get_move(self.board.copy())
            error = self.board.move_error(self.current, move)
            if error is None:
                break
            player.notify_invalid_move(move, error)
            attempts += 1
            if self._max_invalid is not None and attempts >= self._max_invalid:
                raise RuntimeError(f"{player.name} made too many invalid moves.")

        self.board.apply_move(self.current, move)
        self.history.append((self.current, move))
        self._output(f"{player.name} played {move}.")

        if self.board.has_won(self.current):
            self.winner = self.current
            return True
        self.current = 1 - self.current
        return False

    def run(self) -> Optional[int]:
        """Play until someone wins. Returns the winner index, or None if aborted."""
        try:
            while True:
                self.display()
                player = self.players[self.current]
                self._output(
                    f"\n{player.name}'s turn [{PAWN_SYMBOLS[self.current]}] "
                    "(type 'help' for commands)"
                )
                if self.play_turn():
                    break
        except QuitGame:
            self._output("\nGame aborted.")
            return None
        self.display()
        self._output(f"\n*** {self.players[self.winner].name} wins! ***")
        return self.winner
