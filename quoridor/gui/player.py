"""Human player driven by the pygame GUI."""

from __future__ import annotations

from ..moves import Move
from ..player import Player
from ..state import BoardState
from .app import QuoridorGUI


class GuiHumanPlayer(Player):
    """A human who chooses moves with the mouse.

    Like every :class:`Player`, it receives the full immutable
    :class:`BoardState` and returns a ``Move``; the GUI only displays that
    snapshot and never modifies the game board.
    """

    def __init__(self, name: str, gui: QuoridorGUI) -> None:
        super().__init__(name)
        self.gui = gui

    def get_move(self, board_state: BoardState) -> Move:
        return self.gui.wait_for_move(board_state)

    def notify_invalid_move(self, move: Move, reason: str) -> None:
        self.gui.flash(reason)
