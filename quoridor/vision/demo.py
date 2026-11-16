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
from .synthetic import DARK, render_state


class DemoGame:
    def __init__(self, root: tk.Misc, delay_ms: int = 2500, x: int = 420, y: int = 60) -> None:
        self.window = tk.Toplevel(root, bg="#111318")
        self.window.title("Quoridor (demo game for the Screen Assistant)")
        self.window.geometry(f"+{x}+{y}")
        self.label = tk.Label(self.window, bd=0)
        self.label.pack()
        self.delay_ms = delay_ms
        self.bot = AlphaBetaBot("demo", depth=2, time_limit=0.5)
        self.bot.rng = random.Random()
        self._image = None
        self._new_game()
        self.window.after(delay_ms, self._step)

    def _new_game(self) -> None:
        self.board, self.turn, self.moves = Board(), 0, 0
        self._draw()

    def _draw(self) -> None:
        winner = next((p for p in (0, 1) if self.board.has_won(p)), None)
        state = self.board.snapshot(self.turn, self.moves, winner)
        image, _ = render_state(state, DARK, size=(1080, 720), origin=(70, 80))
        ok, png = cv2.imencode(".png", image)
        if ok:
            self._image = tk.PhotoImage(data=base64.b64encode(png.tobytes()))
            self.label.config(image=self._image)

    def _step(self) -> None:
        if not self.window.winfo_exists():
            return
        if any(self.board.has_won(p) for p in (0, 1)) or self.moves > 150:
            self._new_game()
        else:
            state = self.board.snapshot(self.turn, self.moves)
            move = self.bot.get_move(state)
            if self.moves < 2 and not isinstance(move, PawnMove):
                move = PawnMove(*state.valid_pawn_moves[0])
            self.board.apply_move(self.turn, move)
            self.turn, self.moves = 1 - self.turn, self.moves + 1
            self._draw()
        self.window.after(self.delay_ms, self._step)
