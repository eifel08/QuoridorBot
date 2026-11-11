"""ASCII rendering of the board for the CLI."""

from __future__ import annotations

from typing import Union

from .board import Board
from .moves import Orientation
from .state import BoardState

BoardLike = Union[Board, BoardState]

PAWN_SYMBOLS = ("1", "2")
WALL_CHAR = "#"


def render_board(board: BoardLike, highlight: frozenset = frozenset()) -> str:
    """Return a multi-line ASCII picture of the board.

    ``highlight`` is an optional set of squares to mark with ``*``
    (e.g. legal pawn moves for the active player).
    """
    n = board.size
    height, width = 2 * n + 1, 4 * n + 1
    grid = [[" "] * width for _ in range(height)]

    # Base grid.
    for y in range(0, height, 2):
        for x in range(width):
            grid[y][x] = "+" if x % 4 == 0 else "-"
    for y in range(1, height, 2):
        for x in range(0, width, 4):
            grid[y][x] = "|"

    # Cell contents.
    for r in range(n):
        for c in range(n):
            ch = "*" if (r, c) in highlight else "."
            for p, pos in enumerate(board.pawns):
                if pos == (r, c):
                    ch = PAWN_SYMBOLS[p]
            grid[2 * r + 1][4 * c + 2] = ch

    # Walls.
    for r, c, o in board.walls:
        if o is Orientation.HORIZONTAL:
            y = 2 * (r + 1)
            for x in range(4 * c + 1, 4 * (c + 2)):
                grid[y][x] = WALL_CHAR
        else:
            x = 4 * (c + 1)
            for y in range(2 * r + 1, 2 * r + 4):
                grid[y][x] = WALL_CHAR

    lines = []
    header = "    " + "".join(f"  {c} " for c in range(n))
    lines.append(header)
    for y in range(height):
        label = f" {y // 2} " if y % 2 == 1 else "   "
        lines.append(label + " " + "".join(grid[y]))
    return "\n".join(lines)


def render_status(board: BoardLike, names: tuple[str, str]) -> str:
    parts = []
    for p in (0, 1):
        goal = board.goal_rows[p]
        parts.append(f"[{PAWN_SYMBOLS[p]}] {names[p]}: {board.walls_left[p]} walls left (goal: row {goal})")
    return "\n".join(parts)
