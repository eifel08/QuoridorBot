"""A self-playing mock of the target UI, to try the Screen Assistant without a real game.

``python main.py --assistant-demo`` opens this window next to the assistant:
two quick bots play each other on a board drawn by
:mod:`quoridor.vision.synthetic`, and the assistant reads it off the screen
like any other game.
"""

from __future__ import annotations

import base64
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
