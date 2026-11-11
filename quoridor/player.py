"""Player abstractions.

The game engine only ever talks to players through the :class:`Player`
interface, so human and bot players are interchangeable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

from .moves import Move, Orientation, PawnMove, WallMove
from .renderer import render_board
from .state import BoardState

HELP_TEXT = """\
Commands:
  m <row> <col>        Move your pawn to square (row, col).   e.g.  m 7 4
  w <row> <col> <h|v>  Place a wall anchored at (row, col).    e.g.  w 3 4 h
                       row/col are 0-7: the wall touches squares
                       (row,col) and (row,col+1) [h, below them] or
                       (row,col) and (row+1,col) [v, right of them].
  moves                Show your legal pawn moves on the board (*).
  help                 Show this help.
  quit                 Exit the game."""


class QuitGame(Exception):
    """Raised by a player who wants to abort the match."""


class Player(ABC):
    """Abstract player. Subclasses decide how a move is chosen."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.index: int = -1  # assigned by the game (0 or 1)

    def set_index(self, index: int) -> None:
        self.index = index

    @abstractmethod
    def get_move(self, board_state: BoardState) -> Move:
        """Return the move to play.

        ``board_state`` is an immutable, complete snapshot of the position
        (pawns, walls, walls left, legal moves, ...), so implementations do
        not need to remember anything between turns.
        """

    def observe(self, board_state: BoardState) -> None:
        """Called for every player after each move with the new position.

        Optional hook (e.g. for pondering or logging); the default does nothing.
        """

    def notify_invalid_move(self, move: Move, reason: str) -> None:
        """Called by the engine when the returned move was illegal."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


def parse_command(text: str) -> tuple[str, Optional[Move]]:
    """Parse a user command.

    Returns (kind, move) where kind is one of 'move', 'help', 'moves', 'quit'.
    Raises ValueError with a friendly message on malformed input.
    """
    tokens = text.strip().lower().replace(",", " ").split()
    if not tokens:
        raise ValueError("Please enter a command (type 'help' for options).")
    cmd, args = tokens[0], tokens[1:]

    if cmd in ("help", "h", "?"):
        return "help", None
    if cmd in ("quit", "exit", "q"):
        return "quit", None
    if cmd in ("moves", "legal"):
        return "moves", None

    if cmd in ("m", "move"):
        if len(args) != 2:
            raise ValueError("Pawn move format: m <row> <col>   (e.g. m 7 4)")
        row, col = _parse_ints(args)
        return "move", PawnMove(row, col)

    if cmd in ("w", "wall"):
        if len(args) != 3:
            raise ValueError("Wall format: w <row> <col> <h|v>   (e.g. w 3 4 h)")
        row, col = _parse_ints(args[:2])
        o = args[2]
        if o in ("h", "horizontal"):
            orientation = Orientation.HORIZONTAL
        elif o in ("v", "vertical"):
            orientation = Orientation.VERTICAL
        else:
            raise ValueError("Wall orientation must be 'h' (horizontal) or 'v' (vertical).")
        return "move", WallMove(row, col, orientation)

    raise ValueError(f"Unknown command '{cmd}'. Type 'help' for options.")


def _parse_ints(values: list[str]) -> tuple[int, int]:
    try:
        a, b = (int(v) for v in values)
    except ValueError:
        raise ValueError("Row and column must be whole numbers.") from None
    return a, b


class HumanPlayer(Player):
    """A player controlled by a person typing commands in the terminal.

    (The GUI equivalent is :class:`quoridor.gui.GuiHumanPlayer`.)
    """

    def __init__(
        self,
        name: str,
        input_func: Callable[[str], str] = input,
        output_func: Callable[[str], None] = print,
    ) -> None:
        super().__init__(name)
        self._input = input_func
        self._output = output_func

    def get_move(self, board_state: BoardState) -> Move:
        while True:
            try:
                raw = self._input(f"{self.name} > ")
            except EOFError:
                raise QuitGame() from None
            try:
                kind, move = parse_command(raw)
            except ValueError as exc:
                self._output(f"  ! {exc}")
                continue
            if kind == "help":
                self._output(HELP_TEXT)
            elif kind == "quit":
                raise QuitGame()
            elif kind == "moves":
                legal = board_state.valid_pawn_moves
                self._output(render_board(board_state, frozenset(legal)))
                self._output("  Legal pawn moves: " + ", ".join(f"m {r} {c}" for r, c in legal))
            else:
                assert move is not None
                return move

    def notify_invalid_move(self, move: Move, reason: str) -> None:
        self._output(f"  ! Invalid {move}: {reason}")
