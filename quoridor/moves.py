"""Move types exchanged between players and the game engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union


class Orientation(str, Enum):
    """Wall orientation."""

    HORIZONTAL = "h"
    VERTICAL = "v"


@dataclass(frozen=True)
class PawnMove:
    """Move the active player's pawn to the square (row, col)."""

    row: int
    col: int

    def __str__(self) -> str:
        return f"move to ({self.row}, {self.col})"


@dataclass(frozen=True)
class WallMove:
    """Place a wall anchored at the intersection point (row, col).

    The anchor is the top-left square of the 2x2 block of squares the wall
    touches, so ``0 <= row, col <= 7``.

    * Horizontal: the wall lies along the bottom edge of squares
      (row, col) and (row, col + 1), separating them from row + 1.
    * Vertical: the wall lies along the right edge of squares
      (row, col) and (row + 1, col), separating them from col + 1.
    """

    row: int
    col: int
    orientation: Orientation

    def __str__(self) -> str:
        name = "horizontal" if self.orientation is Orientation.HORIZONTAL else "vertical"
        return f"{name} wall at ({self.row}, {self.col})"


Move = Union[PawnMove, WallMove]
