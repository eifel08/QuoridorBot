"""A self-playing mock of the target UI, to try the Screen Assistant without a real game.

``python main.py --assistant-demo`` opens this window next to the assistant:
two quick bots play each other on a board drawn by
:mod:`quoridor.vision.synthetic`, and the assistant reads it off the screen
like any other game.
"""

from __future__ import annotations

import base64
import os
import random
import tkinter as tk

import cv2

from ..board import Board
from ..bots.alphabeta import AlphaBetaBot
from ..moves import PawnMove
from .position import Position, move_notation
from .synthetic import DARK, render


class DemoGame:
    """``layout``: "classic" (green border on the active card, Red at the bottom) or "timed"
    (online layout: clocks, move list, chat, no border, Blue at the bottom)."""

    def __init__(self, root: tk.Misc, delay_ms: int = 2500, x: int = 420, y: int = 60,
                 layout: str = "classic") -> None:
        self.window = tk.Toplevel(root, bg="#111318")
        self.window.title("Quoridor (demo game for the Screen Assistant)")
        self.window.geometry(f"+{x}+{y}")
        self.label = tk.Label(self.window, bd=0)
        self.label.pack()
        self.delay_ms = delay_ms
        self.layout = layout
        self.flipped = layout == "timed"
        self.bot = AlphaBetaBot("demo", depth=2, time_limit=0.5)
        self.bot.rng = random.Random()
        self._image = None
        self._new_game()
        self.window.after(delay_ms, self._step)

    def _new_game(self) -> None:
        self.board, self.turn, self.history = Board(), 0, []
        self._draw()

    def _position(self) -> Position:
        walls = tuple(sorted((r, c, o.value) for r, c, o in self.board.walls))
        pos = Position((self.board.pawns[0], self.board.pawns[1]), walls, tuple(self.board.walls_left), self.turn)
        return pos.mirrored() if self.flipped else pos

    def _draw(self) -> None:
        pos = self._position()
        local = 1 if self.flipped else 0  # the player at the bottom sees the legal-move outlines
        hints = []
        if self.turn == local and pos.winner is None:
            hints = [(m.row, m.col) for m in (pos.screen_move(PawnMove(r, c))
                                              for r, c in self.board.valid_pawn_moves(self.turn))]
        image, _ = render(pawns=pos.pawns, walls=pos.walls, walls_left=pos.walls_left, turn=pos.turn, hints=hints,
                          theme=DARK, size=(1080, 720), origin=(70, 80), layout=self.layout, flipped=self.flipped,
                          moves=self.history, names=("Opponent", "You") if self.flipped else ("You", "Opponent"))
        ok, png = cv2.imencode(".png", image)
        if ok:
            self._image = tk.PhotoImage(data=base64.b64encode(png.tobytes()))
            self.label.config(image=self._image)

    def _step(self) -> None:
        if not self.window.winfo_exists():
            return
        if any(self.board.has_won(p) for p in (0, 1)) or len(self.history) > 150:
            self._new_game()
        else:
            state = self.board.snapshot(self.turn, len(self.history))
            move = self.bot.get_move(state)
            if len(self.history) < 2 and not isinstance(move, PawnMove):
                move = PawnMove(*state.valid_pawn_moves[0])
            self.history.append(move_notation(self._position().screen_move(move)))
            self.board.apply_move(self.turn, move)
            self.turn = 1 - self.turn
            self._draw()
        self.window.after(self.delay_ms, self._step)


class PlayableDemo(DemoGame):
    """A mock online game that is played with the mouse, for trying Auto-Play.

    The bottom player moves by clicking: a legal target square moves the pawn; a
    click in the gap between two squares places the wall that is previewed while
    hovering there. The top player is a quick bot. Clocks run, the move list
    fills, and no card has a green border (the "timed" layout).

    ``wall_rule``: the wall under the cursor starts at the hovered square
    ("first": it extends to the right / downwards) or ends there ("second").
    """

    def __init__(self, root: tk.Misc, flipped: bool = False, wall_rule: str = "first", reply_ms: int = 1500,
                 x: int = 420, y: int = 60) -> None:
        self.local = 1 if flipped else 0          # the bottom player's colour (0 Red, 1 Blue)
        self.wall_rule = os.environ.get("QUORIDOR_DEMO_WALL_RULE", wall_rule)  # to try the click-point check
        self.reply_ms = reply_ms
        self.seconds = [300, 300]
        self.preview = None                        # engine-board wall under the cursor
        self.geometry = None
        self.clicks = 0
        super().__init__(root, delay_ms=reply_ms, x=x, y=y, layout="timed")
        self.flipped = flipped
        self._draw()
        self.label.bind("<Button-1>", self._click)
        self.label.bind("<Motion>", self._motion)
        self.label.bind("<Leave>", lambda _e: self._set_preview(None))
        self.window.after(1000, self._second)

    # -- drawing ---------------------------------------------------------------
    def _draw(self) -> None:
        pos = self._position()
        hints = []
        over = pos.winner is not None
        if self.turn == self.local and not over:
            hints = [(m.row, m.col) for m in (pos.screen_move(PawnMove(r, c))
                                              for r, c in self.board.valid_pawn_moves(self.turn))]
        walls = list(pos.walls)
        if self.preview is not None and not over:
            shown = pos.screen_move(self.preview)
            walls.append((shown.row, shown.col, shown.orientation.value))
        clocks = [f"{s // 60}:{s % 60:02d}" for s in self.seconds]
        image, self.geometry = render(
            pawns=pos.pawns, walls=walls, walls_left=pos.walls_left, turn=pos.turn, hints=hints, theme=DARK,
            size=(1080, 720), origin=(70, 80), layout="timed", flipped=self.flipped, moves=self.history,
            clocks=clocks, active_clock=None if over else self.turn,
            names=("Opponent", "You") if self.flipped else ("You", "Opponent"))
        ok, png = cv2.imencode(".png", image)
        if ok:
            self._image = tk.PhotoImage(data=base64.b64encode(png.tobytes()))
            self.label.config(image=self._image)

    def _second(self) -> None:
        if not self.window.winfo_exists():
            return
        if not any(self.board.has_won(p) for p in (0, 1)):
            self.seconds[self.turn] = max(0, self.seconds[self.turn] - 1)
            self._draw()
        self.window.after(1000, self._second)

    # -- the game ------------------------------------------------------------------
    def _new_game(self) -> None:
        super()._new_game()
        self.seconds = [300, 300]

    def _step(self) -> None:
        """The top player's reply (also the first move when the top player is Red)."""
        if not self.window.winfo_exists():
            return
        over = any(self.board.has_won(p) for p in (0, 1))
        if not over and self.turn != self.local:
            state = self.board.snapshot(self.turn, len(self.history))
            self._apply(self.bot.get_move(state))
        self.window.after(self.reply_ms, self._step)

    def _apply(self, move) -> None:
        self.history.append(move_notation(self._position().screen_move(move)))
        self.board.apply_move(self.turn, move)
        self.turn = 1 - self.turn
        self.preview = None
        self._draw()

    # -- mouse -----------------------------------------------------------------------
    def _hit(self, x: float, y: float):
        """What is under the cursor: a pawn move or a wall, in the engine's board coordinates."""
        from ..moves import Orientation, WallMove

        g = self.geometry
        pos = self._position()
        col, row = (x - g.x0) / g.pitch_x, (y - g.y0) / g.pitch_y
        c, r = round(col), round(row)
        if not (0 <= c <= 8 and 0 <= r <= 8):
            return None
        half = g.side / 2 / g.pitch
        dc, dr = col - c, row - r
        if abs(dc) <= half and abs(dr) <= half:
            return pos.screen_move(PawnMove(r, c))
        back = 1 if self.wall_rule == "second" else 0
        if abs(dc) <= half and abs(dr) > half:       # horizontal gap above or below square (r, c)
            wr, wc = (r if dr > 0 else r - 1), c - back
            wall = WallMove(wr, min(max(wc, 0), 7), Orientation.HORIZONTAL)
        elif abs(dr) <= half and abs(dc) > half:     # vertical gap left or right of square (r, c)
            wr, wc = r - back, (c if dc > 0 else c - 1)
            wall = WallMove(min(max(wr, 0), 7), wc, Orientation.VERTICAL)
        else:
            return None
        if not (0 <= wall.row <= 7 and 0 <= wall.col <= 7):
            return None
        return pos.screen_move(wall)

    def _set_preview(self, wall) -> None:
        if wall != self.preview:
            self.preview = wall
            self._draw()

    def _motion(self, event) -> None:
        from ..moves import WallMove

        hit = self._hit(event.x, event.y) if self.turn == self.local else None
        legal = isinstance(hit, WallMove) and self.board.move_error(self.turn, hit) is None
        self._set_preview(hit if legal else None)

    def _click(self, event) -> None:
        self.clicks += 1
        if self.turn != self.local or any(self.board.has_won(p) for p in (0, 1)):
            return
        hit = self._hit(event.x, event.y)
        if hit is not None and self.board.move_error(self.turn, hit) is None:
            self._apply(hit)
