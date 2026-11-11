"""Quoridor board state and rules."""

from __future__ import annotations

from collections import deque
from typing import Iterable, Optional

from .moves import Move, Orientation, PawnMove, WallMove

BOARD_SIZE = 9
WALLS_PER_PLAYER = 10

Square = tuple[int, int]
Wall = tuple[int, int, Orientation]

DIRECTIONS: tuple[Square, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


class Board:
    """Full game state for a 2-player Quoridor match.

    Player indices are 0 (Player 1) and 1 (Player 2).
    Player 1 starts at (8, 4) and must reach row 0.
    Player 2 starts at (0, 4) and must reach row 8.
    """

    START_POSITIONS: tuple[Square, Square] = ((BOARD_SIZE - 1, BOARD_SIZE // 2), (0, BOARD_SIZE // 2))
    GOAL_ROWS: tuple[int, int] = (0, BOARD_SIZE - 1)

    def __init__(self) -> None:
        self.size = BOARD_SIZE
        self.pawns: list[Square] = list(self.START_POSITIONS)
        self.walls_left: list[int] = [WALLS_PER_PLAYER, WALLS_PER_PLAYER]
        self.walls: set[Wall] = set()

    # ------------------------------------------------------------------ #
    # Copying
    # ------------------------------------------------------------------ #
    def copy(self) -> "Board":
        other = Board.__new__(Board)
        other.size = self.size
        other.pawns = list(self.pawns)
        other.walls_left = list(self.walls_left)
        other.walls = set(self.walls)
        return other

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def goal_row(self, player: int) -> int:
        return self.GOAL_ROWS[player]

    def has_won(self, player: int) -> bool:
        return self.pawns[player][0] == self.goal_row(player)

    def is_blocked(self, a: Square, b: Square, walls: Optional[Iterable[Wall]] = None) -> bool:
        """Return True if a wall separates the orthogonally adjacent squares a and b."""
        wall_set = self.walls if walls is None else walls
        (r1, c1), (r2, c2) = a, b
        if r1 == r2:  # horizontal step -> blocked by a vertical wall
            c = min(c1, c2)
            return (r1, c, Orientation.VERTICAL) in wall_set or (r1 - 1, c, Orientation.VERTICAL) in wall_set
        # vertical step -> blocked by a horizontal wall
        r = min(r1, r2)
        return (r, c1, Orientation.HORIZONTAL) in wall_set or (r, c1 - 1, Orientation.HORIZONTAL) in wall_set

    def can_step(self, a: Square, b: Square, walls: Optional[Iterable[Wall]] = None) -> bool:
        """True if b is on the board, orthogonally adjacent to a and not walled off."""
        if not self.in_bounds(*b):
            return False
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            return False
        return not self.is_blocked(a, b, walls)

    # ------------------------------------------------------------------ #
    # Pawn movement
    # ------------------------------------------------------------------ #
    def valid_pawn_moves(self, player: int) -> list[Square]:
        """All squares the given player's pawn may legally move to."""
        me = self.pawns[player]
        opp = self.pawns[1 - player]
        moves: list[Square] = []
        for dr, dc in DIRECTIONS:
            step = (me[0] + dr, me[1] + dc)
            if not self.can_step(me, step):
                continue
            if step != opp:
                moves.append(step)
                continue
            # Opponent is adjacent in this direction: try a straight jump.
            jump = (opp[0] + dr, opp[1] + dc)
            if self.can_step(opp, jump):
                moves.append(jump)
                continue
            # Straight jump blocked by a wall or the board edge: diagonal jumps.
            for sdr, sdc in ((dc, dr), (-dc, -dr)):  # the two perpendicular directions
                side = (opp[0] + sdr, opp[1] + sdc)
                if self.can_step(opp, side) and side != me:
                    moves.append(side)
        # Remove duplicates while keeping order.
        return list(dict.fromkeys(moves))

    # ------------------------------------------------------------------ #
    # Wall placement
    # ------------------------------------------------------------------ #
    @staticmethod
    def conflicting_walls(wall: Wall) -> list[Wall]:
        """Walls that would overlap or cross the given wall (including itself)."""
        r, c, o = wall
        if o is Orientation.HORIZONTAL:
            return [
                (r, c, Orientation.HORIZONTAL),
                (r, c - 1, Orientation.HORIZONTAL),
                (r, c + 1, Orientation.HORIZONTAL),
                (r, c, Orientation.VERTICAL),
            ]
        return [
            (r, c, Orientation.VERTICAL),
            (r - 1, c, Orientation.VERTICAL),
            (r + 1, c, Orientation.VERTICAL),
            (r, c, Orientation.HORIZONTAL),
        ]

    def wall_error(self, player: int, wall: Wall) -> Optional[str]:
        """Return a reason the wall cannot be placed, or None if it is legal."""
        r, c, _ = wall
        if self.walls_left[player] <= 0:
            return "You have no walls left."
        if not (0 <= r < self.size - 1 and 0 <= c < self.size - 1):
            return f"Wall anchor must be within 0-{self.size - 2} for both row and column."
        for other in self.conflicting_walls(wall):
            if other in self.walls:
                return "That wall overlaps or crosses an existing wall."
        new_walls = self.walls | {wall}
        for p in (0, 1):
            if self.shortest_path_length(p, new_walls) is None:
                who = "your opponent" if p != player else "you"
                return f"That wall would completely block {who} from reaching the goal."
        return None

    def is_valid_wall(self, player: int, wall: Wall) -> bool:
        return self.wall_error(player, wall) is None

    # ------------------------------------------------------------------ #
    # Pathfinding
    # ------------------------------------------------------------------ #
    def shortest_path_length(self, player: int, walls: Optional[Iterable[Wall]] = None) -> Optional[int]:
        """BFS distance from the player's pawn to its goal row, ignoring pawns.

        Returns None if the goal row is unreachable.
        """
        wall_set = self.walls if walls is None else set(walls)
        start = self.pawns[player]
        goal = self.goal_row(player)
        if start[0] == goal:
            return 0
        seen = {start}
        queue: deque[tuple[Square, int]] = deque([(start, 0)])
        while queue:
            square, dist = queue.popleft()
            for dr, dc in DIRECTIONS:
                nxt = (square[0] + dr, square[1] + dc)
                if nxt in seen or not self.can_step(square, nxt, wall_set):
                    continue
                if nxt[0] == goal:
                    return dist + 1
                seen.add(nxt)
                queue.append((nxt, dist + 1))
        return None

    def has_path(self, player: int, walls: Optional[Iterable[Wall]] = None) -> bool:
        return self.shortest_path_length(player, walls) is not None

    # ------------------------------------------------------------------ #
    # Applying moves
    # ------------------------------------------------------------------ #
    def move_error(self, player: int, move: Move) -> Optional[str]:
        """Return why a move is illegal, or None if it is legal."""
        if isinstance(move, PawnMove):
            if (move.row, move.col) not in self.valid_pawn_moves(player):
                return "your pawn cannot reach that square this turn (type 'moves' to see legal moves)."
            return None
        if isinstance(move, WallMove):
            return self.wall_error(player, (move.row, move.col, move.orientation))
        return "Unknown move type."

    def apply_move(self, player: int, move: Move) -> None:
        """Apply a move after validating it. Raises ValueError if illegal."""
        error = self.move_error(player, move)
        if error:
            raise ValueError(error)
        if isinstance(move, PawnMove):
            self.pawns[player] = (move.row, move.col)
        else:
            self.walls.add((move.row, move.col, move.orientation))
            self.walls_left[player] -= 1
