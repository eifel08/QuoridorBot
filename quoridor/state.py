"""Immutable, self-contained snapshot of a Quoridor position.

A :class:`BoardState` is what every :class:`~quoridor.player.Player` receives
on each turn. It contains *everything* needed to choose a move (pawns, walls,
remaining walls, whose turn it is, the legal moves), so a bot never has to keep
its own memory of the board.

Coordinates are always absolute and never flipped for any viewer:
``(row, col)`` with row 0 at the top, Player 1 (index 0) starting at
``(8, 4)`` aiming for row 0 and Player 2 (index 1) starting at ``(0, 4)``
aiming for row 8.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from .moves import Orientation

if TYPE_CHECKING:  # pragma: no cover
    from .board import Board

Square = tuple[int, int]
Wall = tuple[int, int, Orientation]


@dataclass(frozen=True)
class BoardState:
    """Read-only game position.

    Attributes:
        size: Board width/height (9).
        current_player: Index (0 or 1) of the player to move.
        pawns: ``(row, col)`` of each player's pawn, indexed by player.
        goal_rows: Target row of each player.
        walls: All placed walls as ``(row, col, Orientation)``, sorted.
            See :class:`~quoridor.moves.WallMove` for the anchor convention.
        walls_left: Remaining walls of each player.
        valid_pawn_moves: Legal destination squares for ``current_player``
            (jumps included).
        valid_walls: Every legal wall placement for ``current_player``.
        shortest_paths: BFS distance (ignoring pawns) of each player to goal.
        move_number: Number of moves played so far.
        winner: Index of the winner, or ``None`` while the game is running.
    """

    size: int
    current_player: int
    pawns: tuple[Square, Square]
    goal_rows: tuple[int, int]
    walls: tuple[Wall, ...]
    walls_left: tuple[int, int]
    valid_pawn_moves: tuple[Square, ...]
    valid_walls: tuple[Wall, ...]
    shortest_paths: tuple[Optional[int], Optional[int]]
    move_number: int = 0
    winner: Optional[int] = None
    _valid_wall_set: frozenset = field(default=frozenset(), repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_valid_wall_set", frozenset(self.valid_walls))

    # ------------------------------------------------------------------ #
    # Convenience queries
    # ------------------------------------------------------------------ #
    @property
    def opponent(self) -> int:
        return 1 - self.current_player

    @property
    def is_over(self) -> bool:
        return self.winner is not None

    def is_valid_pawn_move(self, square: Square) -> bool:
        return tuple(square) in self.valid_pawn_moves

    def is_valid_wall(self, wall: Wall) -> bool:
        return (wall[0], wall[1], Orientation(wall[2])) in self._valid_wall_set

    # ------------------------------------------------------------------ #
    # Array encodings (plain nested tuples, so no numpy dependency)
    # ------------------------------------------------------------------ #
    def pawn_matrix(self, player: int) -> tuple[tuple[int, ...], ...]:
        """``size x size`` grid with 1 where ``player``'s pawn stands."""
        r0, c0 = self.pawns[player]
        return tuple(
            tuple(1 if (r, c) == (r0, c0) else 0 for c in range(self.size))
            for r in range(self.size)
        )

    def wall_matrix(self, orientation: Orientation) -> tuple[tuple[int, ...], ...]:
        """``(size-1) x (size-1)`` grid with 1 at each anchor holding a wall."""
        n = self.size - 1
        anchors = {(r, c) for r, c, o in self.walls if o is Orientation(orientation)}
        return tuple(tuple(1 if (r, c) in anchors else 0 for c in range(n)) for r in range(n))

    def to_planes(self) -> list[list[list[int]]]:
        """Stack of ``size x size`` feature planes for ML models.

        Planes: [P1 pawn, P2 pawn, horizontal walls, vertical walls]. Wall
        planes are zero-padded on the last row/column to ``size x size``.
        """
        def pad(m: tuple[tuple[int, ...], ...]) -> list[list[int]]:
            rows = [list(row) + [0] for row in m]
            rows.append([0] * self.size)
            return rows

        return [
            [list(r) for r in self.pawn_matrix(0)],
            [list(r) for r in self.pawn_matrix(1)],
            pad(self.wall_matrix(Orientation.HORIZONTAL)),
            pad(self.wall_matrix(Orientation.VERTICAL)),
        ]

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dictionary (orientations as ``"h"``/``"v"``)."""
        return {
            "size": self.size,
            "current_player": self.current_player,
            "pawns": [list(p) for p in self.pawns],
            "goal_rows": list(self.goal_rows),
            "walls": [[r, c, o.value] for r, c, o in self.walls],
            "walls_left": list(self.walls_left),
            "valid_pawn_moves": [list(m) for m in self.valid_pawn_moves],
            "valid_walls": [[r, c, o.value] for r, c, o in self.valid_walls],
            "shortest_paths": list(self.shortest_paths),
            "move_number": self.move_number,
            "winner": self.winner,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoardState":
        return cls(
            size=data["size"],
            current_player=data["current_player"],
            pawns=tuple(tuple(p) for p in data["pawns"]),  # type: ignore[arg-type]
            goal_rows=tuple(data["goal_rows"]),  # type: ignore[arg-type]
            walls=tuple((r, c, Orientation(o)) for r, c, o in data["walls"]),
            walls_left=tuple(data["walls_left"]),  # type: ignore[arg-type]
            valid_pawn_moves=tuple(tuple(m) for m in data["valid_pawn_moves"]),  # type: ignore[misc]
            valid_walls=tuple((r, c, Orientation(o)) for r, c, o in data["valid_walls"]),
            shortest_paths=tuple(data["shortest_paths"]),  # type: ignore[arg-type]
            move_number=data.get("move_number", 0),
            winner=data.get("winner"),
        )

    def to_board(self) -> "Board":
        """A fresh, mutable :class:`Board` for search/simulation."""
        from .board import Board

        board = Board()
        board.size = self.size
        board.pawns = [tuple(p) for p in self.pawns]  # type: ignore[misc]
        board.walls = set(self.walls)
        board.walls_left = list(self.walls_left)
        return board
