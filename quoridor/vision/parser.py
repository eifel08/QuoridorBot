"""Read a Quoridor position from a screenshot.

The parser works on a BGR image (``numpy`` array, as OpenCV and ``mss`` use)
and needs no pixel coordinates from the user:

1. **Board.** It looks for a 9x9 lattice of equally sized square cells
   (:func:`find_board`). Only the geometry matters, not the colours, so light
   and dark themes both work. Once found, the lattice is re-checked cheaply
   on every frame (:func:`verify_board`) instead of being searched again.
2. **Pawns.** The red and the blue pawn are the cells whose centre is red /
   blue *and* differs from the cell's own corners, so tinted cells and the
   white/grey legal-move outlines are not mistaken for pawns.
3. **Walls.** Every groove segment between two cells is compared with the
   typical groove colour; occupied segments are paired into two-square walls.
   The colour of a wall does not matter (neutral or per player).
4. **Player cards** (right of the board). The blue and the red dot icon locate
   the two cards. Walls left are read from the tick bar, from the ``X/10``
   text when ``pytesseract`` is installed, and cross-checked against the
   number of walls on the board.
5. **Turn.** The card with the green border is the player to move. The
   legal-move outlines on the board are a second, independent indicator.

Coordinates follow the rest of the project: ``(row, col)`` with row 0 at the
top; Red is player 0 (starts at the bottom, ``e1``), Blue is player 1 (starts
at the top, ``e9``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, fields
from typing import Optional, Sequence

import cv2
import numpy as np

N = 9
RED, BLUE = 0, 1
PLAYER_NAMES = ("Red", "Blue")

Square = tuple[int, int]
WallT = tuple[int, int, str]          # (row, col, "h" | "v"), anchor as in quoridor.moves.WallMove
Rect = tuple[int, int, int, int]      # x1, y1, x2, y2 (exclusive)


# ---------------------------------------------------------------------- #
# Tunable colour thresholds
# ---------------------------------------------------------------------- #
@dataclass
class Profile:
    """Colour thresholds of the screen parser (OpenCV HSV: hue 0-179).

    The defaults fit a UI with a red and a blue pawn and a green border around
    the active player's card. Save changed values as JSON and load them with
    ``--vision-profile``.
    """

    red_hues: tuple = ((0, 12), (165, 179))
    blue_hues: tuple = ((95, 135),)
    active_hues: tuple = ((38, 90),)   # border of the active player's card (green)
    min_saturation: int = 90
    min_value: int = 90
    pawn_fill: float = 0.30            # share of a cell's centre that must have the pawn colour
    wall_delta: float = 45.0           # colour distance (BGR) between a wall and an empty groove
    hint_delta: float = 14.0           # brightness step of a legal-move outline
    sidebar_width: float = 1.3         # width of the area searched for the player cards, in board widths
    ocr: bool = True                   # read "X/10" with pytesseract when it is installed

    @classmethod
    def load(cls, path: str) -> "Profile":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown profile key(s): {', '.join(sorted(unknown))}")
        for key in ("red_hues", "blue_hues", "active_hues"):
            if key in data:
                data[key] = tuple(tuple(int(v) for v in pair) for pair in data[key])
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps({f.name: getattr(self, f.name) for f in fields(self)}, indent=2)


def hue_mask(hsv: np.ndarray, ranges: Sequence[Sequence[int]], profile: Profile) -> np.ndarray:
    """Pixels whose hue is in one of ``ranges`` and that are saturated and bright enough (0/255)."""
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, (int(lo), profile.min_saturation, profile.min_value), (int(hi), 255, 255))
    return mask


# ---------------------------------------------------------------------- #
# Board geometry
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class Geometry:
    """Where the 9x9 board is, in image pixels."""

    x0: float       # centre of the top-left square (a9)
    y0: float
    pitch_x: float  # distance between the centres of neighbouring squares
    pitch_y: float
    side: float     # edge length of a square

    @property
    def pitch(self) -> float:
        return (self.pitch_x + self.pitch_y) / 2

    @property
    def gap(self) -> float:
        return max(0.0, self.pitch - self.side)

    def center(self, row: float, col: float) -> tuple[float, float]:
        return self.x0 + col * self.pitch_x, self.y0 + row * self.pitch_y

    def cell_rect(self, row: int, col: int) -> Rect:
        cx, cy = self.center(row, col)
        h = self.side / 2
        return round(cx - h), round(cy - h), round(cx + h), round(cy + h)

    def board_rect(self) -> Rect:
        x1, y1, _, _ = self.cell_rect(0, 0)
        _, _, x2, y2 = self.cell_rect(N - 1, N - 1)
        return x1, y1, x2, y2

    def wall_rect(self, row: int, col: int, orientation: str, thickness: Optional[float] = None) -> Rect:
        """Pixel rectangle of the wall slot anchored at ``(row, col)`` (two squares long)."""
        t = (thickness if thickness is not None else max(self.gap, 0.12 * self.side)) / 2
        cx, cy = self.center(row + 0.5, col + 0.5)
        half = (self.pitch + self.side) / 2
        if orientation == "h":
            return round(cx - half), round(cy - t), round(cx + half), round(cy + t)
        return round(cx - t), round(cy - half), round(cx + t), round(cy + half)

    def shifted(self, dx: float, dy: float) -> "Geometry":
        return Geometry(self.x0 + dx, self.y0 + dy, self.pitch_x, self.pitch_y, self.side)

    def scaled(self, f: float) -> "Geometry":
        return Geometry(self.x0 * f, self.y0 * f, self.pitch_x * f, self.pitch_y * f, self.side * f)


def _binaries(bgr: np.ndarray):
    """Black/white versions of the image in which the cells may stand out, most likely first."""
    # Areas of constant colour, whatever the colours are: everything that is not
    # an edge. A wall next to a cell has its own colour, so it cannot merge with it.
    kernel = np.ones((3, 3), np.uint8)
    edges = (cv2.dilate(bgr, kernel).astype(np.int16) - cv2.erode(bgr, kernel)).max(axis=2)
    for limit in (14, 30):
        yield np.where(edges > limit, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    planes = [gray, *cv2.split(bgr)]
    for i, plane in enumerate(planes):
        plane = cv2.GaussianBlur(plane, (3, 3), 0)
        for src in (plane, 255 - plane):
            yield cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 5)
        if i == 0:
            _, otsu = cv2.threshold(plane, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            yield otsu
            yield 255 - otsu


def _squares(binary: np.ndarray, max_side: float) -> list[tuple[float, float, float]]:
    """(centre x, centre y, side) of every filled, square-shaped blob."""
    out = []
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 12 or h < 12 or w > max_side or h > max_side:
            continue
        if not 0.87 < w / h < 1.15 or cv2.contourArea(cnt) < 0.82 * w * h:
            continue
        out.append((x + w / 2, y + h / 2, (w + h) / 2))
    return out


def _axis_fit(v: np.ndarray, pitch: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Assign integer lattice indices to coordinates ``v``; returns (index, inlier, offset, pitch)."""
    ang = 2 * np.pi * v / pitch
    offset = float(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) / (2 * np.pi) * pitch)
    k = np.round((v - offset) / pitch)
    inlier = np.abs(v - offset - k * pitch) < 0.2 * pitch
    for _ in range(2):  # least-squares refinement of offset and pitch
        if inlier.sum() < 3 or np.ptp(k[inlier]) == 0:
            break
        pitch_new, offset_new = np.polyfit(k[inlier], v[inlier], 1)
        if abs(pitch_new - pitch) > 0.1 * pitch:
            break
        pitch, offset = float(pitch_new), float(offset_new)
        k = np.round((v - offset) / pitch)
        inlier = np.abs(v - offset - k * pitch) < 0.15 * pitch
    return k.astype(int), inlier, offset, pitch


def _fit_lattice(cands: np.ndarray, min_cells: int) -> Optional[tuple[Geometry, int]]:
    """Best 9x9 lattice among square candidates ``(x, y, side)`` of similar size."""
    if len(cands) < min_cells:
        return None
    order = np.argsort(cands[:, 2])
    sides = cands[order, 2]
    best: Optional[tuple[Geometry, int]] = None
    start = 0
    while start < len(sides):
        # group of squares whose sides are within +-12 % of each other
        end = int(np.searchsorted(sides, sides[start] * 1.25, side="right"))
        if end - start >= min_cells:
            group = cands[order[start:end]]
            fit = _fit_group(group, min_cells)
            if fit and (best is None or fit[1] > best[1]):
                best = fit
        nxt = int(np.searchsorted(sides, sides[start] * 1.08, side="right"))
        start = max(nxt, start + 1)
    return best


def _fit_group(group: np.ndarray, min_cells: int) -> Optional[tuple[Geometry, int]]:
    x, y = group[:, 0], group[:, 1]
    s = float(np.median(group[:, 2]))
    dx, dy = x[None, :] - x[:, None], y[None, :] - y[:, None]
    mx = (np.abs(dy) < 0.3 * s) & (dx > 0.95 * s) & (dx < 1.7 * s)
    my = (np.abs(dx) < 0.3 * s) & (dy > 0.95 * s) & (dy < 1.7 * s)
    if mx.sum() < 12 or my.sum() < 12:
        return None
    px, py = float(np.median(dx[mx])), float(np.median(dy[my]))
    if abs(px - py) > 0.08 * px:
        return None
    kx, inx, ox, px = _axis_fit(x, px)
    ky, iny, oy, py = _axis_fit(y, py)
    ok = inx & iny
    cells = set(zip(kx[ok].tolist(), ky[ok].tolist()))
    if len(cells) < min_cells:
        return None
    xs, ys = [c[0] for c in cells], [c[1] for c in cells]
    best_count, best_origin = 0, (0, 0)
    for ax in range(min(xs), max(xs) - N + 2):
        for ay in range(min(ys), max(ys) - N + 2):
            count = sum(1 for cx, cy in cells if ax <= cx < ax + N and ay <= cy < ay + N)
            if count > best_count:
                best_count, best_origin = count, (ax, ay)
    if best_count < min_cells:
        return None
    ax, ay = best_origin
    inside = ok & (kx >= ax) & (kx < ax + N) & (ky >= ay) & (ky < ay + N)
    side = float(np.median(group[inside, 2]))
    return Geometry(ox + ax * px, oy + ay * py, px, py, side), best_count


def find_board(bgr: np.ndarray, min_cells: int = 40) -> Optional[Geometry]:
    """Locate the 9x9 board anywhere in the image (None if there is none)."""
    h, w = bgr.shape[:2]
    scale = 1.0
    if max(h, w) > 2200:  # very large screens: search on a smaller copy
        scale = 2200 / max(h, w)
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = bgr.shape[:2]
    max_side = min(h, w) / N
    best: Optional[tuple[Geometry, int]] = None
    cands: list[tuple[float, float, float]] = []
    # Walls touching two cells can merge them into one blob; opening the mask
    # with a kernel thicker than a wall separates them again.
    for kernel in (0, 9, 19, 33):
        for binary in _binaries(bgr):
            if kernel:
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((kernel, kernel), np.uint8))
            found = _squares(binary, max_side)
            if len(found) < 8:
                continue
            cands.extend(found)
            fit = _fit_lattice(_dedupe(cands), min_cells)
            if fit and (best is None or fit[1] > best[1]):
                best = fit
            if best and best[1] >= 72:
                break
        if best and best[1] >= 72:
            break
    if best is None:
        return None
    return best[0].scaled(1 / scale) if scale != 1.0 else best[0]


def _dedupe(cands: list[tuple[float, float, float]]) -> np.ndarray:
    seen: dict[tuple[int, int, int], tuple[float, float, float]] = {}
    for x, y, s in cands:
        q = max(4.0, s / 3)
        seen.setdefault((round(x / q), round(y / q), round(np.log(s) / 0.1)), (x, y, s))
    return np.array(list(seen.values()), dtype=float).reshape(-1, 3)


# ---------------------------------------------------------------------- #
# Sampling helpers
# ---------------------------------------------------------------------- #
def _patch_median(img: np.ndarray, x: float, y: float, hw: float, hh: float) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    x1, x2 = max(0, round(x - hw)), min(w, round(x + hw) + 1)
    y1, y2 = max(0, round(y - hh)), min(h, round(y + hh) + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return np.median(img[y1:y2, x1:x2].reshape(-1, img.shape[2]), axis=0)


def _patch_mean(img: np.ndarray, x: float, y: float) -> Optional[np.ndarray]:
    """Mean colour of the 3x3 pixels around (x, y)."""
    xi, yi = round(x), round(y)
    if xi < 1 or yi < 1 or xi + 2 > img.shape[1] or yi + 2 > img.shape[0]:
        return None
    return img[yi - 1:yi + 2, xi - 1:xi + 2].reshape(-1, img.shape[2]).mean(axis=0)


def _cell_backgrounds(bgr: np.ndarray, g: Geometry) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Background colour of every cell, measured at its four corners (9x9x3), and the
    largest colour difference between a cell's corners (9x9)."""
    out, spread = np.zeros((N, N, 3)), np.zeros((N, N))
    d = 0.33 * g.side
    for r in range(N):
        for c in range(N):
            cx, cy = g.center(r, c)
            samples = [_patch_mean(bgr, cx + sx * d, cy + sy * d) for sx in (-1, 1) for sy in (-1, 1)]
            if any(s is None for s in samples):
                return None
            arr = np.array(samples)
            out[r, c] = np.median(arr, axis=0)
            spread[r, c] = np.abs(arr - out[r, c]).sum(axis=1).max()
    return out, spread


def _segment_colors(bgr: np.ndarray, g: Geometry) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Colour of each groove segment: ``h[r, c]`` below cell (r, c), ``v[r, c]`` right of it."""
    half_len = 0.3 * g.side
    half_t = max(1.0, 0.3 * g.gap)
    hseg, vseg = np.zeros((N - 1, N, 3)), np.zeros((N, N - 1, 3))
    for r in range(N):
        for c in range(N):
            if r < N - 1:
                x, y = g.center(r + 0.5, c)
                m = _patch_median(bgr, x, y, half_len, half_t)
                if m is None:
                    return None
                hseg[r, c] = m
            if c < N - 1:
                x, y = g.center(r, c + 0.5)
                m = _patch_median(bgr, x, y, half_t, half_len)
                if m is None:
                    return None
                vseg[r, c] = m
    return hseg, vseg


def sample_board(bgr: np.ndarray, g: Geometry):
    """Cell and groove colours at ``g``, or None if they do not look like a board there.

    This is the cheap per-frame check that the board is still exactly where it was.
    """
    x1, y1, x2, y2 = g.board_rect()
    if x1 < 0 or y1 < 0 or x2 > bgr.shape[1] or y2 > bgr.shape[0]:
        return None
    backgrounds = _cell_backgrounds(bgr, g)
    segs = _segment_colors(bgr, g)
    if backgrounds is None or segs is None:
        return None
    cells, spread = backgrounds
    cell_ref = np.median(cells.reshape(-1, 3), axis=0)
    seg_all = np.concatenate([segs[0].reshape(-1, 3), segs[1].reshape(-1, 3)])
    seg_ref = np.median(seg_all, axis=0)
    # A cell counts when all four corners show the usual cell colour; if the board
    # has moved even a few pixels, corners land in the grooves and this fails.
    cells_ok = int(((np.linalg.norm(cells - cell_ref, axis=2) < 30) & (spread < 45)).sum())
    segs_ok = int((np.linalg.norm(seg_all - seg_ref, axis=1) < 30).sum())
    # Goal rows may be tinted and up to 40 groove segments can hold walls.
    if cells_ok >= 54 and segs_ok >= 96 and float(np.linalg.norm(cell_ref - seg_ref)) > 18:
        return cells, segs
    return None


def verify_board(bgr: np.ndarray, g: Geometry) -> bool:
    return sample_board(bgr, g) is not None


# ---------------------------------------------------------------------- #
# Result
# ---------------------------------------------------------------------- #
@dataclass
class CardInfo:
    """What was found of one player's card in the sidebar."""

    dot: Optional[Rect] = None
    ticks: list[Rect] = field(default_factory=list)
    tick_count: Optional[int] = None
    ocr_count: Optional[int] = None


@dataclass
class ParseResult:
    ok: bool = False
    reason: str = ""                      # why the frame is not usable (when not ok)
    geometry: Optional[Geometry] = None
    pawns: Optional[tuple[Square, Square]] = None      # (Red, Blue)
    walls: tuple[WallT, ...] = ()
    walls_left: Optional[tuple[int, int]] = None       # (Red, Blue)
    walls_source: str = ""                             # "text", "ticks", "ticks+board", "board"
    walls_consistent: bool = False                     # sidebar counts agree with the walls on the board
    border_turn: Optional[int] = None                  # player whose card has the green border
    hint_turn: Optional[int] = None                    # player whose legal moves are outlined
    hints: tuple[Square, ...] = ()
    active_rect: Optional[Rect] = None
    cards: tuple[CardInfo, CardInfo] = field(default_factory=lambda: (CardInfo(), CardInfo()))
    sidebar: Optional[Rect] = None
    warnings: list[str] = field(default_factory=list)

    @property
    def turn(self) -> Optional[int]:
        """Player to move as far as this single frame shows it."""
        return self.border_turn if self.border_turn is not None else self.hint_turn

    def game_rect(self) -> Optional[Rect]:
        """Image rectangle holding the board and whatever was found of the player cards."""
        if self.geometry is None:
            return None
        x1, y1, x2, y2 = self.geometry.board_rect()
        pad = round(0.6 * self.geometry.pitch)
        found = [r for card in self.cards for r in ([card.dot] if card.dot else []) + card.ticks]
        if self.active_rect:
            found.append(self.active_rect)
        if found:
            x2 = max(x2, max(r[2] for r in found) + pad)  # room for the "X/10" text after the ticks
            y1, y2 = min(y1, min(r[1] for r in found)), max(y2, max(r[3] for r in found))
        elif self.sidebar:
            x2 = self.sidebar[2]
        return x1 - pad, y1 - pad, x2 + pad, y2 + pad


# ---------------------------------------------------------------------- #
# Parser
# ---------------------------------------------------------------------- #
class ScreenParser:
    """Parses frames; remembers where the board was to stay fast."""

    def __init__(self, profile: Optional[Profile] = None) -> None:
        self.profile = profile or Profile()
        self.geometry: Optional[Geometry] = None
        self._ocr_cache: dict[bytes, Optional[int]] = {}
        self._tesseract = None  # None = not tried, False = unavailable
        self._samples = None

    def reset(self) -> None:
        """Forget the board position (search the whole frame again)."""
        self.geometry = None

    # -- board ----------------------------------------------------------
    def locate(self, bgr: np.ndarray) -> Optional[Geometry]:
        self._samples = sample_board(bgr, self.geometry) if self.geometry is not None else None
        if self._samples is None:
            self.geometry = find_board(bgr)
            if self.geometry is not None:
                self._samples = sample_board(bgr, self.geometry)
                if self._samples is None:
                    self.geometry = None
        return self.geometry

    # -- everything -----------------------------------------------------
    def parse(self, bgr: np.ndarray) -> ParseResult:
        res = ParseResult()
        g = self.locate(bgr)
        if g is None:
            res.reason = "no 9x9 board found on the screen"
            return res
        res.geometry = g
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        masks = (hue_mask(hsv, self.profile.red_hues, self.profile),
                 hue_mask(hsv, self.profile.blue_hues, self.profile))
        cells, segs = self._samples

        self._read_sidebar(bgr, hsv, masks, g, res)
        problems = []
        pawns = self._find_pawns(bgr, masks, cells, g)
        if isinstance(pawns, str):
            problems.append(pawns)
        else:
            res.pawns = pawns
        walls = self._find_walls(segs)
        if isinstance(walls, str):
            problems.append(walls)
        else:
            res.walls = walls
        if problems:
            res.reason = "; ".join(problems)
            return res

        res.hints = self._find_hints(bgr, g, res.pawns)
        res.hint_turn = _hint_turn(res)
        self._resolve_walls_left(res)
        if res.border_turn is not None and res.hint_turn is not None and res.border_turn != res.hint_turn:
            res.warnings.append("card border and legal-move outlines disagree about the turn")
        res.ok = True
        return res

    # -- pawns ----------------------------------------------------------
    def _find_pawns(self, bgr, masks, cells, g: Geometry):
        d = max(2, round(0.28 * g.side))
        scores = np.zeros((2, N, N))
        for r in range(N):
            for c in range(N):
                cx, cy = (round(v) for v in g.center(r, c))
                patch = bgr[cy - d:cy + d + 1, cx - d:cx + d + 1].astype(float)
                differs = np.linalg.norm(patch - cells[r, c], axis=2) > 40
                for p in (RED, BLUE):
                    colored = masks[p][cy - d:cy + d + 1, cx - d:cx + d + 1] > 0
                    scores[p, r, c] = float((colored & differs).mean())
        found = []
        for p in (RED, BLUE):
            flat = scores[p].ravel()
            best = int(flat.argmax())
            ranked = np.sort(flat)
            if ranked[-1] < self.profile.pawn_fill:
                return f"{PLAYER_NAMES[p].lower()} pawn not found"
            if ranked[-2] > 0.6 * ranked[-1]:
                return f"more than one {PLAYER_NAMES[p].lower()} pawn candidate (pawn moving?)"
            found.append(divmod(best, N))
        if found[0] == found[1]:
            return "both pawns on the same square"
        return (found[0], found[1])

    # -- walls ----------------------------------------------------------
    def _find_walls(self, segs):
        hseg, vseg = segs
        ref = np.median(np.concatenate([hseg.reshape(-1, 3), vseg.reshape(-1, 3)]), axis=0)
        hocc = np.linalg.norm(hseg - ref, axis=2) > self.profile.wall_delta   # [8][9]
        vocc = np.linalg.norm(vseg - ref, axis=2) > self.profile.wall_delta   # [9][8]
        walls: list[WallT] = []
        for r in range(N - 1):
            runs = _runs(hocc[r])
            if runs is None:
                return "wall segments do not pair up (wall being placed?)"
            walls += [(r, c, "h") for c in runs]
        for c in range(N - 1):
            runs = _runs(vocc[:, c])
            if runs is None:
                return "wall segments do not pair up (wall being placed?)"
            walls += [(r, c, "v") for r in runs]
        if len(walls) > 20:
            return "more than 20 walls detected"
        return tuple(sorted(walls))

    # -- legal-move outlines ---------------------------------------------
    def _find_hints(self, bgr, g: Geometry, pawns) -> tuple[Square, ...]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(float)
        band = max(2, round(0.1 * g.side))
        inner = max(2, round(0.2 * g.side))
        diff = np.zeros((N, N))
        for r in range(N):
            for c in range(N):
                x1, y1, x2, y2 = g.cell_rect(r, c)
                cell = gray[y1:y2, x1:x2]
                if cell.shape[0] <= 2 * band or cell.shape[1] <= 2 * band:
                    return ()
                ring = np.concatenate([cell[:band].ravel(), cell[-band:].ravel(),
                                       cell[band:-band, :band].ravel(), cell[band:-band, -band:].ravel()])
                corners = np.concatenate([cell[band + 1:inner + band, band + 1:inner + band].ravel(),
                                          cell[-inner - band:-band - 1, -inner - band:-band - 1].ravel()])
                # the brightest part of the ring vs. the cell's own background
                diff[r, c] = np.percentile(ring, 80) - np.median(corners)
        base = float(np.median(diff))
        hints = [(r, c) for r in range(N) for c in range(N)
                 if diff[r, c] - base > self.profile.hint_delta and (r, c) not in pawns]
        return tuple(hints) if len(hints) <= 6 else ()

    # -- sidebar ----------------------------------------------------------
    def _read_sidebar(self, bgr, hsv, masks, g: Geometry, res: ParseResult) -> None:
        bx1, by1, bx2, by2 = g.board_rect()
        h, w = bgr.shape[:2]
        x1 = min(w, round(bx2 + 0.35 * g.pitch))
        x2 = min(w, round(bx2 + self.profile.sidebar_width * (bx2 - bx1)))
        y1, y2 = max(0, round(by1 - g.pitch)), min(h, round(by2 + g.pitch))
        if x2 - x1 < g.pitch or y2 - y1 < g.pitch:
            res.warnings.append("no room right of the board for the player cards")
            return
        res.sidebar = (x1, y1, x2, y2)
        comps = [_components(masks[p][y1:y2, x1:x2], x1, y1, min_area=(0.06 * g.pitch) ** 2) for p in (RED, BLUE)]
        dots: list[Optional[Rect]] = []
        for p in (RED, BLUE):
            round_ones = [(rect, area) for rect, area in comps[p] if _is_round(rect, area, g.pitch)]
            # the icon is in the card's header line: the topmost round blob of that colour
            dots.append(min(round_ones, key=lambda ra: ra[0][1])[0] if round_ones else None)
            res.cards[p].dot = dots[p]
        for p in (RED, BLUE):
            dot = dots[p]
            if dot is None:
                res.warnings.append(f"{PLAYER_NAMES[p].lower()} player card not found")
                continue
            other = dots[1 - p]
            lo, hi = y1, y2
            if other is not None:  # stay on this card's side of the other card's icon
                if other[1] > dot[1]:
                    hi = other[1]
                else:
                    lo = other[3]
            ticks = [rect for rect, _ in comps[p]
                     if rect != dot and lo <= rect[1] and rect[3] <= hi and rect[1] >= dot[1] - (dot[3] - dot[1])]
            ticks = _tick_group(ticks, bgr, hsv)
            res.cards[p].ticks = ticks
            res.cards[p].tick_count = len(ticks) if len(ticks) <= 10 else None
            if self.profile.ocr:
                res.cards[p].ocr_count = self._ocr_count(bgr, dot, (x1, max(lo, y1), x2, min(hi, y2)), g)

        # active card: a large hollow rectangle in the highlight colour around one of the icons
        active = hue_mask(hsv[y1:y2, x1:x2], self.profile.active_hues, self.profile)
        active = cv2.morphologyEx(active, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(active, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            if cw < 1.5 * g.pitch or ch < 0.6 * g.pitch:
                continue
            if cv2.countNonZero(active[cy:cy + ch, cx:cx + cw]) > 0.4 * cw * ch:
                continue  # a filled green area (button, banner), not a border
            rect = (x1 + cx, y1 + cy, x1 + cx + cw, y1 + cy + ch)
            inside = [p for p in (RED, BLUE) if dots[p] is not None and _contains(rect, dots[p])]
            if len(inside) == 1:
                res.border_turn = inside[0]
                res.active_rect = rect
                break

    def _ocr_count(self, bgr, dot: Rect, area: Rect, g: Geometry) -> Optional[int]:
        """``X`` of the ``X/10`` text in the card around ``dot`` (needs pytesseract)."""
        if self._tesseract is None:
            try:
                import pytesseract

                pytesseract.get_tesseract_version()
                self._tesseract = pytesseract
            except Exception:
                self._tesseract = False
        if not self._tesseract:
            return None
        x1, y1, x2, y2 = area
        y1 = max(y1, round(dot[1] - 0.5 * g.pitch))
        y2 = min(y2, round(dot[3] + 2.5 * g.pitch))
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        key = crop.tobytes()
        if key not in self._ocr_cache:
            if len(self._ocr_cache) > 64:
                self._ocr_cache.clear()
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            scale = max(1.0, 90 / max(8.0, 0.3 * g.pitch) / 3)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            if np.median(gray) < 128:
                gray = 255 - gray  # tesseract prefers dark text on a light background
            value = None
            try:
                text = self._tesseract.image_to_string(gray, config="--psm 11")
                m = re.search(r"(\d{1,2})\s*/\s*10\b", text)
                if m and 0 <= int(m.group(1)) <= 10:
                    value = int(m.group(1))
            except Exception:
                value = None
            self._ocr_cache[key] = value
        return self._ocr_cache[key]

    # -- walls left --------------------------------------------------------
    def _resolve_walls_left(self, res: ParseResult) -> None:
        """Combine text, tick bars and the walls on the board into (Red, Blue) walls left."""
        placed = len(res.walls)
        ocr = [res.cards[p].ocr_count for p in (RED, BLUE)]
        ticks = [res.cards[p].tick_count for p in (RED, BLUE)]

        def fits(pair) -> bool:
            return None not in pair and 20 - pair[0] - pair[1] == placed

        mixed = [ocr[p] if ocr[p] is not None else ticks[p] for p in (RED, BLUE)]
        for pair, source in ((ocr, "text"), (ticks, "ticks"), (mixed, "text+ticks")):
            if fits(pair):
                res.walls_left, res.walls_source, res.walls_consistent = (pair[0], pair[1]), source, True
                return
        # One card unreadable (no ticks found counts as unreadable here, since it did
        # not fit as "0 left" above): the walls on the board tell the rest.
        unreadable = [p for p in (RED, BLUE) if mixed[p] is None or (ocr[p] is None and ticks[p] == 0)]
        known = [p for p in (RED, BLUE) if mixed[p] is not None]
        if len(unreadable) == 1:
            p = 1 - unreadable[0]
            other = 20 - placed - mixed[p]
            if 0 <= other <= 10:
                pair = [0, 0]
                pair[p], pair[1 - p] = mixed[p], other
                res.walls_left, res.walls_source, res.walls_consistent = (pair[0], pair[1]), "ticks+board", True
                return
        if len(known) == 2:
            res.walls_left = (mixed[0], mixed[1])
            res.walls_source = "text" if None not in ocr else "ticks"
            res.warnings.append(f"cards show {20 - mixed[0] - mixed[1]} walls used but {placed} are on the board")
            return
        # Nothing readable: assume both players used the same number of walls.
        used_red = placed // 2
        res.walls_left = (max(0, 10 - used_red), max(0, 10 - (placed - used_red)))
        res.walls_source = "board"
        res.warnings.append("wall counts not readable from the cards; assuming an even split")


# ---------------------------------------------------------------------- #
# Small helpers
# ---------------------------------------------------------------------- #
def _runs(occupied: np.ndarray) -> Optional[list[int]]:
    """Start index of every wall in a line of groove segments (None if a run has odd length)."""
    starts, i, n = [], 0, len(occupied)
    while i < n:
        if not occupied[i]:
            i += 1
            continue
        j = i
        while j < n and occupied[j]:
            j += 1
        if (j - i) % 2:
            return None
        starts += list(range(i, j, 2))
        i = j
    return starts


def _components(mask: np.ndarray, ox: int, oy: int, min_area: float) -> list[tuple[Rect, int]]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area >= min_area:
            out.append(((ox + x, oy + y, ox + x + w, oy + y + h), area))
    return out


def _is_round(rect: Rect, area: int, pitch: float) -> bool:
    w, h = rect[2] - rect[0], rect[3] - rect[1]
    if not 0.12 * pitch <= max(w, h) <= 0.9 * pitch:
        return False
    return 0.8 < w / h < 1.25 and 0.62 < area / (w * h) < 0.92


def _contains(outer: Rect, inner: Rect) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _tick_group(rects: list[Rect], bgr: np.ndarray, hsv: np.ndarray) -> list[Rect]:
    """The wall ticks among coloured blobs: the largest set of same-sized blobs on one line.

    Spent ticks drawn in a dimmed version of the player colour are dropped.
    """
    if not rects:
        return []
    best: list[Rect] = []
    for ref in rects:
        rw, rh = ref[2] - ref[0], ref[3] - ref[1]
        ry = (ref[1] + ref[3]) / 2
        group = [r for r in rects
                 if abs((r[1] + r[3]) / 2 - ry) <= 0.35 * rh
                 and abs((r[3] - r[1]) - rh) <= max(2, 0.25 * rh)
                 and abs((r[2] - r[0]) - rw) <= max(2, 0.25 * rw)]
        if len(group) > len(best):
            best = group
    if len(best) >= 2:
        strength = np.array([float(hsv[r[1]:r[3], r[0]:r[2], 1].astype(float).mean()
                                   * hsv[r[1]:r[3], r[0]:r[2], 2].astype(float).mean()) for r in best])
        top = strength.max()
        order = np.sort(strength)
        gaps = np.diff(order)
        if len(gaps) and gaps.max() > 0.3 * top:  # two clearly separate brightness levels
            cut = order[int(gaps.argmax())]
            best = [r for r, s in zip(best, strength) if s > cut]
    return sorted(best)


def _hint_turn(res: ParseResult) -> Optional[int]:
    """Whose legal pawn moves the outlined squares are (None if unclear)."""
    if not res.hints or res.pawns is None:
        return None
    from ..board import Board
    from ..moves import Orientation

    board = Board()
    board.pawns = [res.pawns[RED], res.pawns[BLUE]]
    board.walls = {(r, c, Orientation(o)) for r, c, o in res.walls}
    hints = set(res.hints)
    owners = [p for p in (RED, BLUE) if hints <= set(board.valid_pawn_moves(p))]
    return owners[0] if len(owners) == 1 else None


# ---------------------------------------------------------------------- #
# Debug drawing
# ---------------------------------------------------------------------- #
def annotate(bgr: np.ndarray, res: ParseResult, turn: Optional[int] = None) -> np.ndarray:
    """A copy of the frame with everything the parser detected drawn on it.

    Cropped to the board and the player cards when a board was found. ``turn``
    overrides the turn shown (e.g. the one resolved from the move history).
    """
    out = bgr.copy()
    g = res.geometry
    if g is None:
        cv2.putText(out, res.reason or "no board", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return out
    yellow, white = (0, 255, 255), (255, 255, 255)
    colors = ((60, 60, 255), (255, 140, 40))  # BGR red, blue
    lw = max(1, round(g.side / 28))
    for r in range(N):
        for c in range(N):
            x1, y1, x2, y2 = g.cell_rect(r, c)
            cv2.rectangle(out, (x1, y1), (x2, y2), yellow, 1)
    for r, c, o in res.walls:
        x1, y1, x2, y2 = g.wall_rect(r, c, o)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 255), max(2, lw + 1))
        cv2.putText(out, wall_label(r, c, o), (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
    for r, c in res.hints:
        cx, cy = (round(v) for v in g.center(r, c))
        cv2.drawMarker(out, (cx, cy), white, cv2.MARKER_TILTED_CROSS, round(g.side / 4), lw)
    if res.pawns:
        for p in (RED, BLUE):
            cx, cy = (round(v) for v in g.center(*res.pawns[p]))
            cv2.circle(out, (cx, cy), round(0.46 * g.side), white, lw + 2)
            cv2.circle(out, (cx, cy), round(0.46 * g.side), colors[p], lw)
            cv2.putText(out, square_label(*res.pawns[p]), (cx - 12, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        white, 2)
    for p in (RED, BLUE):
        card = res.cards[p]
        if card.dot:
            cv2.rectangle(out, card.dot[:2], card.dot[2:], white, 2)
        for t in card.ticks:
            cv2.rectangle(out, t[:2], t[2:], yellow, 1)
        if card.dot:
            bits = [f"ticks {card.tick_count}" if card.tick_count is not None else "ticks ?"]
            if card.ocr_count is not None:
                bits.append(f"text {card.ocr_count}/10")
            cv2.putText(out, f"{PLAYER_NAMES[p]}: " + ", ".join(bits), (card.dot[0], card.dot[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, yellow, 1)
    if res.active_rect:
        x1, y1, x2, y2 = res.active_rect
        cv2.rectangle(out, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), yellow, 2)

    x1, y1, x2, y2 = res.game_rect()
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(out.shape[1], x2), min(out.shape[0], y2)
    out = out[y1:y2, x1:x2].copy()

    shown = turn if turn is not None else res.turn
    lines = [("OK" if res.ok else "NOT READABLE: " + res.reason)]
    if res.ok:
        lines.append(f"turn: {PLAYER_NAMES[shown] if shown is not None else '?'}   "
                     f"walls left: Blue {res.walls_left[BLUE]}  Red {res.walls_left[RED]} ({res.walls_source})   "
                     f"walls on board: {len(res.walls)}")
    lines += ["! " + w for w in res.warnings]
    bar = np.zeros((22 * len(lines) + 8, out.shape[1], 3), np.uint8)
    for i, text in enumerate(lines):
        color = (0, 255, 0) if (i == 0 and res.ok) else ((0, 200, 255) if text.startswith("!") else white)
        if i == 0 and not res.ok:
            color = (80, 80, 255)
        cv2.putText(bar, text, (6, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return np.vstack([bar, out])


def square_label(row: int, col: int) -> str:
    """``(8, 4)`` -> ``"e1"`` (files a-i from the left, ranks 1-9 from the bottom)."""
    return "abcdefghi"[col] + str(N - row)


def wall_label(row: int, col: int, orientation: str) -> str:
    """``(5, 4, "h")`` -> ``"e3h"``: the wall's lower-left square, then h or v."""
    return "abcdefghi"[col] + str(N - 1 - row) + orientation
