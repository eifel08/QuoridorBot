"""Pure screen geometry for the GUI (no pygame needed, easy to unit-test).

Maps between board coordinates ``(row, col)`` and pixels. The mapping is the
same for every player: row 0 is always drawn at the top, so the board is
never flipped or re-oriented for the viewer.
"""

from __future__ import annotations

from typing import Optional

from ..board import BOARD_SIZE, WALLS_PER_PLAYER
from ..moves import Orientation

Rect = tuple[float, float, float, float]  # x, y, w, h
Square = tuple[int, int]
Wall = tuple[int, int, Orientation]

H, V = Orientation.HORIZONTAL, Orientation.VERTICAL


class Layout:
    """Pixel geometry of the window."""

    def __init__(self, cell: int = 52, gap: int = 14, margin: int = 40, frame: int = 14,
                 panel_width: int = 280, size: int = BOARD_SIZE, eval_bar: int = 16) -> None:
        self.n = size
        self.cell = cell
        self.gap = gap
        self.pitch = cell + gap
        self.margin = margin
        self.frame = frame
        self.board_px = size * cell + (size - 1) * gap
        frame_px = self.board_px + 2 * frame
        # Left of the board: [eval bar of Red / analysis] [row numbers] [board].
        left = 18
        self.frame_x = left + eval_bar + 32
        self.row_label_x = self.frame_x - 16
        self.ox = self.frame_x + frame
        self.oy = margin + frame
        self.frame_rect: Rect = (self.frame_x, margin, frame_px, frame_px)
        frame_right = self.frame_x + frame_px
        # Eval bars on opposite sides of the board: slot 0 (Red) left, slot 1 (Blue) right.
        self.eval_bar_rects: tuple[Rect, Rect] = (
            (left, margin, eval_bar, frame_px),
            (frame_right + 14, margin, eval_bar, frame_px),
        )
        self.panel_x = frame_right + 14 + eval_bar + 16
        self.panel_width = panel_width
        self.width = self.panel_x + panel_width + margin
        self.height = 2 * margin + frame_px
        # Eval readouts: Red's below its bar (Red starts at the bottom), Blue's above its bar.
        self.eval_text_pos = (
            (left + eval_bar / 2, self.height - margin / 2),
            (frame_right + 14 + eval_bar / 2, margin / 2),
        )

        # Player cards in the side panel: Player 2 (top of board) at the
        # top, Player 1 (bottom of board) at the bottom.
        card_h = 180
        self.card_rects: tuple[Rect, Rect] = (
            (self.panel_x, margin + frame_px - card_h, panel_width, card_h),
            (self.panel_x, margin, panel_width, card_h),
        )
        self.button_rect: Rect = (self.panel_x, self.height / 2 + 62, panel_width, 38)

        # Reserve stubs (walls left, drawn upright) inside each card.
        self.stub_w = 12
        self.stub_h = 2 * cell + gap - 48
        self.stub_step = (panel_width - 40 - self.stub_w) / (WALLS_PER_PLAYER - 1)

    # ------------------------------------------------------------------ #
    # Squares
    # ------------------------------------------------------------------ #
    def square_rect(self, row: int, col: int) -> Rect:
        return (self.ox + col * self.pitch, self.oy + row * self.pitch, self.cell, self.cell)

    def square_center(self, row: int, col: int) -> tuple[float, float]:
        x, y, w, h = self.square_rect(row, col)
        return (x + w / 2, y + h / 2)

    def square_at(self, px: float, py: float, forgiving: bool = False) -> Optional[Square]:
        """Square under the pixel, or None.

        With ``forgiving`` the nearest square is returned for points in the
        grooves too (useful when dropping a dragged pawn).
        """
        lx, ly = px - self.ox, py - self.oy
        if forgiving:
            if not (-self.gap <= lx <= self.board_px + self.gap and -self.gap <= ly <= self.board_px + self.gap):
                return None
            col = round((lx - self.cell / 2) / self.pitch)
            row = round((ly - self.cell / 2) / self.pitch)
            return (min(max(row, 0), self.n - 1), min(max(col, 0), self.n - 1))
        if lx < 0 or ly < 0:
            return None
        col, fx = divmod(lx, self.pitch)
        row, fy = divmod(ly, self.pitch)
        if col >= self.n or row >= self.n or fx > self.cell or fy > self.cell:
            return None
        return (int(row), int(col))

    # ------------------------------------------------------------------ #
    # Walls
    # ------------------------------------------------------------------ #
    def wall_rect(self, wall: Wall) -> Rect:
        r, c, o = wall
        long = 2 * self.cell + self.gap
        if Orientation(o) is H:
            return (self.ox + c * self.pitch, self.oy + r * self.pitch + self.cell, long, self.gap)
        return (self.ox + c * self.pitch + self.cell, self.oy + r * self.pitch, self.gap, long)

    def wall_candidates(self, px: float, py: float) -> list[Wall]:
        """Walls the cursor could mean, most likely first ([] if off-board).

        The nearest groove decides the orientation:

        * horizontal groove: the wall covers the hovered column and extends
          **left** one square (shifted right at the left edge);
        * vertical groove: the wall covers the hovered row and extends
          **down** one square (shifted up at the bottom edge).

        The second candidate is the same groove shifted by one square, used
        as a fallback when the first placement is illegal.
        """
        lx, ly = px - self.ox, py - self.oy
        tol = self.cell / 2
        if not (-tol <= lx <= self.board_px + tol and -tol <= ly <= self.board_px + tol):
            return []
        lx = min(max(lx, 0.0), self.board_px - 1)
        ly = min(max(ly, 0.0), self.board_px - 1)
        col = min(int(lx // self.pitch), self.n - 1)
        row = min(int(ly // self.pitch), self.n - 1)

        def nearest_groove(v: float) -> tuple[int, float]:
            k = round((v - self.cell - self.gap / 2) / self.pitch)
            k = min(max(k, 0), self.n - 2)
            return k, abs(v - (k * self.pitch + self.cell + self.gap / 2))

        kv, dv = nearest_groove(lx)
        kh, dh = nearest_groove(ly)
        last = self.n - 2
        if dv < dh:
            anchor = min(row, last)
            alt = anchor - 1 if row < self.n - 1 else None
            cands = [(anchor, kv, V)]
            if alt is not None and alt >= 0:
                cands.append((alt, kv, V))
            return cands
        anchor = max(col - 1, 0)
        cands = [(kh, anchor, H)]
        if col > 0 and anchor + 1 <= last:
            cands.append((kh, anchor + 1, H))
        return cands

    # ------------------------------------------------------------------ #
    # Side panel
    # ------------------------------------------------------------------ #
    def stub_rect(self, player: int, index: int) -> Rect:
        x, y, w, h = self.card_rects[player]
        return (x + 20 + index * self.stub_step, y + h - self.stub_h - 14, self.stub_w, self.stub_h)

    def reserve_rect(self, player: int) -> Rect:
        x, y, w, h = self.card_rects[player]
        return (x + 10, y + h - self.stub_h - 28, w - 20, self.stub_h + 20)

    @staticmethod
    def contains(rect: Rect, px: float, py: float) -> bool:
        x, y, w, h = rect
        return x <= px <= x + w and y <= py <= y + h
