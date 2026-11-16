"""From parsed frames to positions the engine can search.

:class:`Position` is one detected game position (pawns, walls, walls left,
side to move). :class:`PositionTracker` watches consecutive
:class:`~quoridor.vision.parser.ParseResult` frames and reports a position
only once it has been stable for a few frames and differs from the last one
reported, so the engine is not started on identical frames or on a pawn that
is still sliding. It also works out whose turn it is when the card border is
not visible, from what changed since the previous position.

Notation: files ``a``-``i`` from the left, ranks ``1``-``9`` from the bottom,
so Red starts on ``e1`` and Blue on ``e9``. A pawn move is its target square
(``e2``); a wall is its lower-left square plus ``h`` or ``v`` (``e3h`` lies
above e3 and f3, ``d4v`` lies right of d4 and d5).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from ..board import Board
from ..moves import Move, Orientation, PawnMove, WallMove
from ..state import BoardState
from .parser import BLUE, N, PLAYER_NAMES, RED, ParseResult, Square, WallT, square_label, wall_label

FILES = "abcdefghi"


def move_notation(move: Move) -> str:
    if isinstance(move, PawnMove):
        return square_label(move.row, move.col)
    return wall_label(move.row, move.col, Orientation(move.orientation).value)


def parse_notation(text: str) -> Move:
    """``"e2"`` -> PawnMove, ``"e3h"`` / ``"d4v"`` -> WallMove."""
    text = text.strip().lower()
    if len(text) not in (2, 3) or text[0] not in FILES or not text[1].isdigit():
        raise ValueError(f"bad move '{text}'")
    col, rank = FILES.index(text[0]), int(text[1])
    if len(text) == 2:
        if not 1 <= rank <= N:
            raise ValueError(f"bad square '{text}'")
        return PawnMove(N - rank, col)
    if text[2] not in "hv" or not 1 <= rank <= N - 1 or col > N - 2:
        raise ValueError(f"bad wall '{text}'")
    return WallMove(N - 1 - rank, col, Orientation(text[2]))


@dataclass(frozen=True)
class Position:
    pawns: tuple[Square, Square]        # (Red, Blue)
    walls: tuple[WallT, ...]            # sorted (row, col, "h"|"v")
    walls_left: tuple[int, int]         # (Red, Blue)
    turn: Optional[int] = None          # player to move, None if unknown

    @property
    def board_key(self) -> tuple:
        return self.pawns, self.walls, self.walls_left

    @property
    def winner(self) -> Optional[int]:
        if self.pawns[RED][0] == 0:
            return RED
        if self.pawns[BLUE][0] == N - 1:
            return BLUE
        return None

    def to_board(self) -> Board:
        board = Board()
        board.pawns = [tuple(self.pawns[RED]), tuple(self.pawns[BLUE])]  # type: ignore[list-item]
        board.walls = {(r, c, Orientation(o)) for r, c, o in self.walls}
        board.walls_left = list(self.walls_left)
        return board

    def error(self) -> Optional[str]:
        """Why this cannot be a real Quoridor position (None if it can)."""
        if self.pawns[RED] == self.pawns[BLUE]:
            return "both pawns on one square"
        board = self.to_board()
        for wall in board.walls:
            if any(other in board.walls for other in Board.conflicting_walls(wall) if other != wall):
                return "detected walls overlap"
        if not (board.has_path(RED) and board.has_path(BLUE)):
            return "a pawn is walled in"
        if not all(0 <= n <= 10 for n in self.walls_left):
            return "impossible wall count"
        return None

    def to_state(self) -> BoardState:
        """The position as the bots see it (the turn must be known)."""
        if self.turn is None:
            raise ValueError("side to move unknown")
        return self.to_board().snapshot(self.turn, move_number=2, winner=self.winner)

    def summary(self) -> str:
        turn = PLAYER_NAMES[self.turn] if self.turn is not None else "?"
        return (f"Red {square_label(*self.pawns[RED])} ({self.walls_left[RED]} walls), "
                f"Blue {square_label(*self.pawns[BLUE])} ({self.walls_left[BLUE]} walls), "
                f"{len(self.walls)} walls on board, {turn} to move")


def mover_between(before: Position, after: Position) -> Optional[int]:
    """Who made the single move that turns ``before`` into ``after`` (None if it is not one move)."""
    red_moved = before.pawns[RED] != after.pawns[RED]
    blue_moved = before.pawns[BLUE] != after.pawns[BLUE]
    if before.walls == after.walls:
        if red_moved != blue_moved:
            return RED if red_moved else BLUE
        return None
    if red_moved or blue_moved:
        return None
    if len(after.walls) != len(before.walls) + 1 or not set(before.walls) <= set(after.walls):
        return None
    spent = [before.walls_left[p] - after.walls_left[p] for p in (RED, BLUE)]
    if spent == [1, 0]:
        return RED
    if spent == [0, 1]:
        return BLUE
    return None


class PositionTracker:
    """Turns a stream of parsed frames into "the position changed" events."""

    def __init__(self, stable_frames: int = 2, unsure_frames: int = 6) -> None:
        self.stable_frames = stable_frames
        self.unsure_frames = unsure_frames     # frames to wait when indicators contradict each other
        self.current: Optional[Position] = None
        self.turn_source = ""                  # "border", "outlines", "history", "manual" or ""
        self.forced_turn: Optional[int] = None  # set by the user to override the detection
        self.problem = ""                      # why the last stable frame was rejected
        self._candidate: Optional[tuple] = None
        self._count = 0

    def reset(self) -> None:
        self.current = None
        self._candidate, self._count = None, 0
        self.problem = ""

    def update(self, res: ParseResult) -> Optional[Position]:
        """Feed one frame. Returns the new position when it changed, else None."""
        if not res.ok or res.pawns is None or res.walls_left is None:
            self._candidate, self._count = None, 0
            return None
        frame_turn = self.forced_turn if self.forced_turn is not None else res.turn
        key = (res.pawns, res.walls, res.walls_left, frame_turn)
        if key == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = key, 1

        pos = Position(res.pawns, res.walls, res.walls_left, frame_turn)
        mover = mover_between(self.current, pos) if self.current is not None else None
        source = "manual" if self.forced_turn is not None else (
            "border" if res.border_turn is not None else ("outlines" if res.hint_turn is not None else ""))
        if pos.turn is None:
            if self.current is not None and self.current.board_key == pos.board_key:
                pos = replace(pos, turn=self.current.turn)
                source = self.turn_source
            elif mover is not None:
                pos = replace(pos, turn=1 - mover)
                source = "history"
        # A border that still marks the player who just moved is probably about to change.
        doubtful = (not res.walls_consistent) or (mover is not None and pos.turn == mover
                                                  and self.forced_turn is None)
        if self._count < (self.unsure_frames if doubtful else self.stable_frames):
            return None
        if self.current is not None and pos == self.current:
            return None
        self.problem = pos.error() or ""
        if self.problem:
            return None
        self.current, self.turn_source = pos, source
        return pos
