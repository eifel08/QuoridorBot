"""Players driven by the pygame GUI."""

from __future__ import annotations

import threading

from ..evaluation import Eval
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


class GuiBotPlayer(Player):
    """Runs any bot :class:`Player` in a background thread.

    The window keeps redrawing (and stays closable) while the bot thinks.
    The wrapped bot only ever sees the immutable ``BoardState``.
    """

    def __init__(self, bot: Player, gui: QuoridorGUI, show_eval: bool = False) -> None:
        super().__init__(bot.name)
        self.bot = bot
        self.gui = gui
        self.show_eval = show_eval  # draw this bot's search score in its eval bar

    def set_index(self, index: int) -> None:
        super().set_index(index)
        self.bot.set_index(index)

    def get_move(self, board_state: BoardState) -> Move:
        result: dict = {}

        def work() -> None:
            try:
                result["move"] = self.bot.get_move(board_state)
            except BaseException as exc:  # surfaced in the GUI thread below
                result["error"] = exc

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        self.gui.wait_for_bot(board_state, thread)
        if "error" in result:
            raise result["error"]
        self.gui.bot_info = getattr(self.bot, "last_info", None)
        if self.show_eval:
            self.gui.set_bot_eval(self.index, Eval.from_info(self.gui.bot_info, self.index, self.name))
        return result["move"]
