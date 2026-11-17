"""From parsed frames to positions the engine can search.

:class:`Position` is one detected game position (pawns, walls, walls left,
side to move). :class:`PositionTracker` watches consecutive
:class:`~quoridor.vision.parser.ParseResult` frames and reports a position
only once it has been stable for a few frames and differs from the last one
reported, so the engine is not started on identical frames or on a pawn that
is still sliding. It also works out whose turn it is when the card border is
not visible, from what changed since the previous position.

Orientation: usually Red starts on the bottom row and Blue on the top row,
but a game can be shown the other way round (``Position.flipped``: Red starts
at the top and moves down). The bots always think with Red moving up, so a
flipped position is mirrored top-to-bottom for them and their moves are
mirrored back (:meth:`Position.to_board`, :meth:`Position.screen_move`).

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
    flipped: bool = False               # True: Red's home row is the top row (Red moves down the screen)

    @property
    def board_key(self) -> tuple:
        return self.pawns, self.walls, self.walls_left, self.flipped

    @property
    def home_rows(self) -> tuple[int, int]:
        """Screen row each player started on: (Red, Blue)."""
        return (0, N - 1) if self.flipped else (N - 1, 0)

    @property
    def goal_rows(self) -> tuple[int, int]:
        """Screen row each player has to reach: (Red, Blue)."""
        return self.home_rows[BLUE], self.home_rows[RED]

    @property
    def winner(self) -> Optional[int]:
        for p in (RED, BLUE):
            if self.pawns[p][0] == self.goal_rows[p]:
                return p
        return None

    def mirrored(self) -> "Position":
        """The same position seen upside down (top and bottom rows swapped)."""
        pawns = tuple((N - 1 - r, c) for r, c in self.pawns)
        walls = tuple(sorted((N - 2 - r, c, o) for r, c, o in self.walls))
        return replace(self, pawns=pawns, walls=walls, flipped=not self.flipped)  # type: ignore[arg-type]

    def screen_move(self, move: Move) -> Move:
        """A move of the bots' board (see :meth:`to_board`) as it has to be played on the screen."""
        if not self.flipped:
            return move
        if isinstance(move, PawnMove):
            return PawnMove(N - 1 - move.row, move.col)
        return WallMove(N - 2 - move.row, move.col, move.orientation)

    def to_board(self) -> Board:
        """The bots' board: Red (player 0) always heads for row 0, so a flipped position is mirrored."""
        pos = self.mirrored() if self.flipped else self
        board = Board()
        board.pawns = [tuple(pos.pawns[RED]), tuple(pos.pawns[BLUE])]  # type: ignore[list-item]
        board.walls = {(r, c, Orientation(o)) for r, c, o in pos.walls}
        board.walls_left = list(pos.walls_left)
        return board

    def parity_turn(self) -> int:
        """Side to move guessed from the number of moves played: Red moves first.

        Moves played = walls placed + pawn steps. The pawn steps are estimated
        from how far each pawn is from its start square, which has the right
        parity unless a pawn has jumped over the other one an odd number of
        times, so this is the last resort after the other indicators.
        """
        plies = sum(10 - n for n in self.walls_left)
        for p in (RED, BLUE):
            plies += abs(self.pawns[p][0] - self.home_rows[p]) + abs(self.pawns[p][1] - N // 2)
        return plies % 2

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
        """The position as the bots see it: :meth:`to_board` coordinates (the turn must be known)."""
        if self.turn is None:
            raise ValueError("side to move unknown")
        return self.to_board().snapshot(self.turn, move_number=2, winner=self.winner)

    def summary(self) -> str:
        turn = PLAYER_NAMES[self.turn] if self.turn is not None else "?"
        return (f"Red {square_label(*self.pawns[RED])} ({self.walls_left[RED]} walls), "
                f"Blue {square_label(*self.pawns[BLUE])} ({self.walls_left[BLUE]} walls), "
                f"{len(self.walls)} walls on board, {turn} to move, "
                f"Red moves {'down' if self.flipped else 'up'}")


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
    """Turns a stream of parsed frames into "the position changed" events.

    Side to move, first match wins: the user's choice, the green card border,
    the move list, the legal-move outlines, the move just seen being made
    (history), and finally the parity of the number of moves played.
    """

    def __init__(self, stable_frames: int = 2, unsure_frames: int = 6) -> None:
        self.stable_frames = stable_frames
        self.unsure_frames = unsure_frames     # frames to wait when indicators contradict each other
        self.current: Optional[Position] = None
        self.turn_source = ""                  # "manual", "border", "move list", "outlines", "history", "parity"
        self.orientation_source = ""           # "tint", "pawns" (guessed from where the pawns stand)
        self.forced_turn: Optional[int] = None  # set by the user to override the detection
        self.counted_turn: Optional[int] = None  # set by Auto-Play from its exact move count (Red moves first)
        self.forced_flipped: Optional[bool] = None
        self.problem = ""                      # why the last stable frame was rejected
        self._candidate: Optional[tuple] = None
        self._count = 0
        self._flipped: Optional[bool] = None   # last orientation seen or guessed
        self._parity_fix = 0                   # 1 once a jump has been seen to break the parity rule
        self._overruled: Optional[tuple] = None  # (board, turn, source) of a reading that history overruled

    def reset(self) -> None:
        self.current = None
        self._candidate, self._count = None, 0
        self._flipped, self._parity_fix, self._overruled = None, 0, None
        self.problem = ""

    def _orientation(self, res: ParseResult) -> bool:
        if self.forced_flipped is not None:
            self.orientation_source = "manual"
            return self.forced_flipped
        if res.flipped is not None:
            self._flipped, self.orientation_source = res.flipped, "tint"
        elif self._flipped is None:
            # No tinted rows: early in a game each pawn is still on its own side.
            self._flipped = res.pawns[RED][0] < res.pawns[BLUE][0] and res.pawns[RED][0] <= 3
            self.orientation_source = "pawns"
        return self._flipped

    def update(self, res: ParseResult) -> Optional[Position]:
        """Feed one frame. Returns the new position when it changed, else None."""
        if not res.ok or res.pawns is None or res.walls_left is None:
            self._candidate, self._count = None, 0
            return None
        flipped = self._orientation(res)
        given = self.forced_turn if self.forced_turn is not None else self.counted_turn
        frame_turn = given if given is not None else res.turn
        key = (res.pawns, res.walls, res.walls_left, frame_turn, flipped)
        if key == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = key, 1

        pos = Position(res.pawns, res.walls, res.walls_left, frame_turn, flipped)
        mover = mover_between(self.current, pos) if self.current is not None else None
        source = "manual" if self.forced_turn is not None else (
            "move count" if self.counted_turn is not None else res.turn_source)
        if pos.turn is None:
            if self.current is not None and self.current.board_key == pos.board_key:
                pos = replace(pos, turn=self.current.turn)
                source = self.turn_source
            elif mover is not None:
                pos = replace(pos, turn=1 - mover)
                source = "history"
            else:
                pos = replace(pos, turn=pos.parity_turn() ^ self._parity_fix)
                source = "parity"
        if self._overruled == (pos.board_key, pos.turn, source):
            pos, source = replace(pos, turn=1 - pos.turn), "history"  # same misreading as before
        # An indicator that still names the player who just moved (a border or a
        # move list that has not caught up yet) is probably about to change.
        contradicted = mover is not None and pos.turn == mover and source not in ("manual", "move count")
        doubtful = (not res.walls_consistent) or contradicted
        if self._count < (self.unsure_frames if doubtful else self.stable_frames):
            return None
        if contradicted and source in ("move list", "parity"):
            # It did not change: the move we saw being made is the better evidence
            # (a misread list row, or a jump that broke the parity rule).
            self._overruled = (pos.board_key, pos.turn, source)
            pos, source = replace(pos, turn=1 - mover), "history"
        if self.current is not None and pos == self.current:
            return None
        self.problem = pos.error() or ""
        if self.problem:
            return None
        if source != "parity" and pos.turn is not None:
            self._parity_fix = pos.turn ^ pos.parity_turn()
        self.current, self.turn_source = pos, source
        return pos
