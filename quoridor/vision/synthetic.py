"""Draws a mock screenshot of the Quoridor UI the screen assistant reads.

Used by the tests and by ``main.py --assistant-demo`` to try the parser
without the real site: a 9x9 board with files ``a``-``i`` under it and ranks
``1``-``9`` on its right, a blue pawn (top) and a red pawn (bottom), walls in
the grooves, optional legal-move outlines, and two player cards on the right
with a coloured dot, a name, a tick bar and an ``X/10`` readout. The card of
the player to move has a green border.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from .parser import BLUE, N, RED, Geometry

RGB = tuple[int, int, int]


@dataclass(frozen=True)
class Theme:
    page: RGB = (17, 19, 24)
    board: RGB = (27, 30, 38)        # also the groove colour
    cell: RGB = (58, 63, 77)
    hint: RGB = (205, 210, 220)      # legal-move outline
    label: RGB = (140, 146, 160)
    card: RGB = (30, 33, 42)
    text: RGB = (228, 230, 236)
    red: RGB = (239, 68, 68)
    blue: RGB = (59, 130, 246)
    green: RGB = (34, 197, 94)
    wall: Optional[RGB] = (226, 190, 110)   # None: walls in their owner's colour
    spent_tick: Optional[RGB] = (62, 66, 78)  # None: spent ticks are not drawn; "dim": see dim_spent
    dim_spent: bool = False          # draw spent ticks as a dark shade of the player colour


DARK = Theme()
LIGHT = Theme(page=(236, 238, 242), board=(120, 98, 80), cell=(232, 214, 180), hint=(255, 255, 255),
              label=(90, 94, 104), card=(250, 250, 252), text=(30, 32, 40), wall=None, spent_tick=None)
THEMES = {"dark": DARK, "light": LIGHT}


def _bgr(rgb: RGB) -> tuple[int, int, int]:
    return rgb[2], rgb[1], rgb[0]


def _rounded(img, x1, y1, x2, y2, color, radius, thickness=-1) -> None:
    radius = int(max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2)))
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1, cv2.LINE_AA)
        for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
            cv2.circle(img, (cx, cy), radius, color, -1, cv2.LINE_AA)
        return
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
    for (cx, cy), start in (((x1 + radius, y1 + radius), 180), ((x2 - radius, y1 + radius), 270),
                            ((x2 - radius, y2 - radius), 0), ((x1 + radius, y2 - radius), 90)):
        cv2.ellipse(img, (cx, cy), (radius, radius), start, 0, 90, color, thickness, cv2.LINE_AA)


def render(pawns: Sequence[tuple[int, int]] = ((8, 4), (0, 4)),
           walls: Iterable[tuple[int, int, str]] = (),
           walls_left: Sequence[int] = (10, 10),
           turn: Optional[int] = RED,
           hints: Iterable[tuple[int, int]] = (),
           wall_owners: Optional[Sequence[int]] = None,
           theme: Theme = DARK,
           cell: int = 56, gap: int = 12,
           size: tuple[int, int] = (1280, 800),
           origin: tuple[int, int] = (90, 110),
           names: Sequence[str] = ("You", "Opponent"),
           layout: str = "classic",
           flipped: bool = False,
           tint: bool = True,
           moves: Sequence[str] = (),
           clocks: Sequence[str] = ("5:02", "4:32"),
           low_time: Optional[int] = None,
           active_clock: Optional[int] = None,
           ) -> tuple[np.ndarray, Geometry]:
    """Render a screenshot; returns ``(bgr image, true board geometry)``.

    ``pawns`` and ``walls_left`` are ``(Red, Blue)``; ``walls`` use the
    project's anchors ``(row, col, "h"|"v")``; ``wall_owners`` (one player per
    wall) colours the walls when the theme has no neutral wall colour.
    """
    w, h = size
    img = np.full((h, w, 3), _bgr(theme.page), np.uint8)
    # some unrelated page content: a header bar and a button
    cv2.rectangle(img, (0, 0), (w, 48), _bgr(theme.card), -1)
    cv2.putText(img, "Quoridor Online", (24, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, _bgr(theme.text), 2, cv2.LINE_AA)

    pitch = cell + gap
    ox, oy = origin
    board_px = N * cell + (N - 1) * gap
    pad = gap
    _rounded(img, ox - pad, oy - pad, ox + board_px + pad, oy + board_px + pad, _bgr(theme.board), 10)
    geom = Geometry(ox + cell / 2, oy + cell / 2, pitch, pitch, cell)

    player_rgb = (theme.red, theme.blue)
    top_player = RED if flipped else BLUE
    home = {0: top_player, N - 1: 1 - top_player}
    for r in range(N):
        color = theme.cell
        if tint and r in home:
            color = tuple(round(0.84 * a + 0.16 * b) for a, b in zip(theme.cell, player_rgb[home[r]]))
        for c in range(N):
            x, y = ox + c * pitch, oy + r * pitch
            _rounded(img, x, y, x + cell, y + cell, _bgr(color), max(2, cell // 9))
    for r, c in hints:
        x, y = ox + c * pitch, oy + r * pitch
        _rounded(img, x + 1, y + 1, x + cell - 1, y + cell - 1, _bgr(theme.hint), max(2, cell // 9), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = cell / 80
    for c in range(N):
        cv2.putText(img, "abcdefghi"[c], (ox + c * pitch + cell // 2 - 6, oy + board_px + pad + int(26 * fs + 8)),
                    font, 0.6 * max(fs, 0.7), _bgr(theme.label), 1, cv2.LINE_AA)
    for r in range(N):
        cv2.putText(img, str(N - r), (ox + board_px + pad + 8, oy + r * pitch + cell // 2 + 6),
                    font, 0.6 * max(fs, 0.7), _bgr(theme.label), 1, cv2.LINE_AA)

    for i, (r, c, o) in enumerate(walls):
        color = theme.wall if theme.wall is not None else player_rgb[wall_owners[i] if wall_owners else i % 2]
        t = max(4, gap - 4)
        cx, cy = ox + (c + 1) * pitch - gap / 2, oy + (r + 1) * pitch - gap / 2
        half = cell + gap / 2 - 2
        if o == "h":
            _rounded(img, cx - half, cy - t / 2, cx + half, cy + t / 2, _bgr(color), t // 2)
        else:
            _rounded(img, cx - t / 2, cy - half, cx + t / 2, cy + half, _bgr(color), t // 2)

    for p in (RED, BLUE):
        r, c = pawns[p]
        center = (int(ox + c * pitch + cell / 2), int(oy + r * pitch + cell / 2))
        cv2.circle(img, center, int(cell * 0.36), _bgr(player_rgb[p]), -1, cv2.LINE_AA)
        cv2.circle(img, (center[0] - cell // 9, center[1] - cell // 9), max(2, cell // 12),
                   _bgr(tuple(min(255, v + 70) for v in player_rgb[p])), -1, cv2.LINE_AA)

    # player cards: the player whose home row is the top row has the top card
    timed = layout == "timed"
    card_x = ox + board_px + pad + int(0.9 * cell)
    card_w, card_h = int(4.6 * cell), int((1.9 if timed else 2.1) * cell)
    small = 0.5 * max(fs, 0.8)
    for p, card_y in ((top_player, oy), (1 - top_player, oy + board_px - card_h)):
        _rounded(img, card_x, card_y, card_x + card_w, card_y + card_h, _bgr(theme.card), 12)
        if turn == p and not timed:
            _rounded(img, card_x, card_y, card_x + card_w, card_y + card_h, _bgr(theme.green), 12, 3)
        dot = (card_x + int(0.5 * cell), card_y + int(0.55 * cell))
        cv2.circle(img, dot, int(0.17 * cell), _bgr(player_rgb[p]), -1, cv2.LINE_AA)
        cv2.putText(img, names[p], (dot[0] + int(0.4 * cell), dot[1] + 7), font, 0.65 * max(fs, 0.8),
                    _bgr(theme.text), 2, cv2.LINE_AA)
        if timed:
            if active_clock == p:  # the running clock sits in a brighter box
                bx = card_x + card_w - int(1.45 * cell)
                box = tuple(min(255, v + 45) for v in theme.card) if sum(theme.card) < 380 else \
                    tuple(max(0, v - 45) for v in theme.card)
                _rounded(img, bx, dot[1] - int(0.3 * cell), card_x + card_w - 8, dot[1] + int(0.3 * cell), _bgr(box), 6)
            clock_color = theme.red if low_time == p else theme.text
            cv2.putText(img, clocks[p], (card_x + card_w - int(1.3 * cell), dot[1] + 8), font, 0.8 * max(fs, 0.8),
                        _bgr(clock_color), 2, cv2.LINE_AA)
        tick_w, tick_h, step = max(4, cell // 8), int(0.5 * cell), max(8, int(0.3 * cell))
        ty = card_y + int((1.05 if timed else 1.15) * cell)
        for i in range(10):
            tx = card_x + int(0.35 * cell) + i * step
            if i < walls_left[p]:
                color = player_rgb[p]
            elif theme.dim_spent:
                color = tuple(int(v * 0.3) for v in player_rgb[p])
            elif theme.spent_tick is not None:
                color = theme.spent_tick
            else:
                continue
            _rounded(img, tx, ty, tx + tick_w, ty + tick_h, _bgr(color), 2)
        cv2.putText(img, f"{walls_left[p]}/10", (card_x + int(0.5 * cell) + 10 * step, ty + tick_h - 4), font,
                    0.6 * max(fs, 0.8), _bgr(theme.text), 1, cv2.LINE_AA)

    if timed:
        # move list: "1. e2 e8" per row, the newest rows visible
        my1 = oy + card_h + int(0.2 * cell)
        my2 = my1 + int(2.9 * cell)
        _rounded(img, card_x, my1, card_x + card_w, my2, _bgr(theme.card), 10)
        cv2.putText(img, "Moves", (card_x + 14, my1 + int(0.42 * cell)), font, small, _bgr(theme.label), 1, cv2.LINE_AA)
        rows = [(i // 2 + 1, moves[i:i + 2]) for i in range(0, len(moves), 2)]
        line_h = int(0.42 * cell)
        visible = rows[-max(1, (my2 - my1 - int(0.6 * cell)) // line_h):]
        for k, (number, pair) in enumerate(visible):
            y = my1 + int(0.9 * cell) + k * line_h
            cv2.putText(img, f"{number}.", (card_x + 14, y), font, small, _bgr(theme.label), 1, cv2.LINE_AA)
            for j, mv in enumerate(pair):
                cv2.putText(img, mv, (card_x + int(0.9 * cell) + j * int(1.3 * cell), y), font, small,
                            _bgr(theme.text), 1, cv2.LINE_AA)
        # chat box with coloured name dots (these must not be taken for the card icons)
        cy1 = my2 + int(0.2 * cell)
        cy2 = oy + board_px - card_h - int(0.2 * cell)
        if cy2 - cy1 > cell:
            _rounded(img, card_x, cy1, card_x + card_w, cy2, _bgr(theme.card), 10)
            cv2.putText(img, "Chat", (card_x + 14, cy1 + int(0.42 * cell)), font, small, _bgr(theme.label), 1,
                        cv2.LINE_AA)
            for k, (who, text) in enumerate(((RED, "good luck"), (BLUE, "thanks, 10/10 game"), (RED, "gg"))):
                y = cy1 + int(0.85 * cell) + k * line_h
                if y > cy2 - 8:
                    break
                cv2.circle(img, (card_x + 22, y - 5), max(4, int(0.13 * cell)), _bgr(player_rgb[who]), -1, cv2.LINE_AA)
                cv2.putText(img, text, (card_x + 40, y), font, small, _bgr(theme.text), 1, cv2.LINE_AA)
    return img, geom


def render_state(state, theme: Theme = DARK, show_hints: bool = True, hint_player: Optional[int] = RED,
                 **kwargs) -> tuple[np.ndarray, Geometry]:
    """Render a :class:`~quoridor.state.BoardState` (Red at the bottom).

    Legal-move outlines are drawn only on ``hint_player``'s turn, like the real UI
    shows them only to the local player.
    """
    walls = [(r, c, getattr(o, "value", o)) for r, c, o in state.walls]
    hints = state.valid_pawn_moves if (show_hints and state.current_player == hint_player) else ()
    return render(pawns=state.pawns, walls=walls, walls_left=state.walls_left, turn=state.current_player,
                  hints=hints, theme=theme, **kwargs)
