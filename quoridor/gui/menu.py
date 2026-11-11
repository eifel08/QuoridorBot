"""Start menu for the pygame GUI: game mode, bot depth slider, think time, side."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pygame

from ..bots import MAX_DEPTH
from . import theme as T

TIME_CHOICES: tuple[Optional[float], ...] = (1.0, 3.0, 5.0, 10.0, 30.0, None)


@dataclass
class MatchSettings:
    mode: str = "hvh"            # "hvh" (Human vs Human) or "hvb" (Human vs Bot)
    depth: int = 4               # bot search depth, 1..30
    time_limit: Optional[float] = 5.0  # seconds per bot move, None = no limit
    human_side: int = 0          # 0 = Red (moves first), 1 = Blue


def _time_label(t: Optional[float]) -> str:
    return "no limit" if t is None else f"{t:g} s"


class Menu:
    """Draws the menu on the GUI's window and returns the chosen settings."""

    def __init__(self, gui) -> None:
        self.gui = gui
        self.screen = gui.screen
        w, h = self.screen.get_size()
        self.cx = w / 2
        col_w = 520
        self.left = self.cx - col_w / 2
        self.col_w = col_w
        y = 150
        self.mode_rects = [pygame.Rect(self.left, y, col_w / 2 - 6, 52),
                           pygame.Rect(self.left + col_w / 2 + 6, y, col_w / 2 - 6, 52)]
        self.slider = pygame.Rect(self.left + 10, 285, col_w - 20, 8)
        bw = (col_w - 5 * 8) / 6
        self.time_rects = [pygame.Rect(self.left + i * (bw + 8), 360, bw, 40) for i in range(6)]
        self.side_rects = [pygame.Rect(self.left, 450, col_w / 2 - 6, 40),
                           pygame.Rect(self.left + col_w / 2 + 6, 450, col_w / 2 - 6, 40)]
        self.start_rect = pygame.Rect(self.cx - 120, h - 120, 240, 54)
        self.dragging_slider = False

    # ------------------------------------------------------------------ #
    def run(self, settings: MatchSettings) -> Optional[MatchSettings]:
        """Event loop; returns the settings on Start, None if the window closes."""
        s = MatchSettings(**vars(settings))
        clock = self.gui.clock
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        return s
                    if s.mode == "hvb" and event.key in (pygame.K_LEFT, pygame.K_DOWN):
                        s.depth = max(1, s.depth - 1)
                    if s.mode == "hvb" and event.key in (pygame.K_RIGHT, pygame.K_UP):
                        s.depth = min(MAX_DEPTH, s.depth + 1)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.start_rect.collidepoint(event.pos):
                        return s
                    for i, r in enumerate(self.mode_rects):
                        if r.collidepoint(event.pos):
                            s.mode = ("hvh", "hvb")[i]
                    if s.mode == "hvb":
                        if self.slider.inflate(20, 30).collidepoint(event.pos):
                            self.dragging_slider = True
                            s.depth = self._depth_at(event.pos[0])
                        for i, r in enumerate(self.time_rects):
                            if r.collidepoint(event.pos):
                                s.time_limit = TIME_CHOICES[i]
                        for i, r in enumerate(self.side_rects):
                            if r.collidepoint(event.pos):
                                s.human_side = i
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.dragging_slider = False
                if event.type == pygame.MOUSEMOTION and self.dragging_slider:
                    s.depth = self._depth_at(event.pos[0])
            self.draw(s)
            pygame.display.flip()
            clock.tick(T.FPS)

    def _depth_at(self, x: float) -> int:
        t = (x - self.slider.left) / self.slider.width
        return max(1, min(MAX_DEPTH, round(1 + t * (MAX_DEPTH - 1))))

    def _knob_x(self, depth: int) -> float:
        return self.slider.left + (depth - 1) / (MAX_DEPTH - 1) * self.slider.width

    # ------------------------------------------------------------------ #
    def _button(self, rect, label: str, selected: bool, color=None) -> None:
        mouse = pygame.mouse.get_pos()
        base = (58, 64, 88) if selected else T.PANEL
        if rect.collidepoint(mouse) and not selected:
            base = (44, 49, 69)
        pygame.draw.rect(self.screen, base, rect, border_radius=10)
        if selected:
            pygame.draw.rect(self.screen, color or T.PANEL_ACTIVE_EDGE, rect, width=2, border_radius=10)
        self.gui._text(label, rect.center, self.gui.font_small if rect.height < 50 else self.gui.font,
                       T.TEXT if selected else T.TEXT_DIM)

    def draw(self, s: MatchSettings) -> None:
        g, scr = self.gui, self.screen
        scr.fill(T.BACKGROUND)
        g._text("Quoridor", (self.cx, 70), g.font_big, T.TEXT)
        g._text("Choose a game mode", (self.cx, 118), g.font_small, T.TEXT_DIM)
        self._button(self.mode_rects[0], "Human vs Human", s.mode == "hvh")
        self._button(self.mode_rects[1], "Human vs Bot", s.mode == "hvb")

        if s.mode == "hvb":
            g._text(f"Search depth: {s.depth}", (self.left, 250), g.font, T.TEXT, anchor="midleft")
            g._text("1-30 plies (arrow keys work too)", (self.left + self.col_w, 250), g.font_small,
                    T.TEXT_DIM, anchor="midright")
            pygame.draw.rect(scr, (60, 66, 90), self.slider, border_radius=4)
            filled = self.slider.copy()
            filled.width = self._knob_x(s.depth) - self.slider.left
            pygame.draw.rect(scr, T.PANEL_ACTIVE_EDGE, filled, border_radius=4)
            pygame.draw.circle(scr, T.PANEL_ACTIVE_EDGE, (self._knob_x(s.depth), self.slider.centery), 12)
            g._text("faster", (self.slider.left, 312), g.font_small, T.TEXT_DIM, anchor="midleft")
            g._text("stronger", (self.slider.right, 312), g.font_small, T.TEXT_DIM, anchor="midright")

            g._text("Think time per bot move", (self.left, 342), g.font_small, T.TEXT_DIM, anchor="midleft")
            for t, r in zip(TIME_CHOICES, self.time_rects):
                self._button(r, _time_label(t), s.time_limit == t)

            g._text("You play", (self.left, 432), g.font_small, T.TEXT_DIM, anchor="midleft")
            self._button(self.side_rects[0], "Red (moves first)", s.human_side == 0, T.PAWN_COLORS[0])
            self._button(self.side_rects[1], "Blue", s.human_side == 1, T.PAWN_COLORS[1])
            g._text("The bot plays the best move from the deepest search it completes within the time.",
                    (self.cx, 525), g.font_small, T.TEXT_DIM)
        else:
            g._text("Two people take turns on this computer.", (self.cx, 260), g.font_small, T.TEXT_DIM)

        pygame.draw.rect(scr, T.PANEL_ACTIVE_EDGE, self.start_rect, border_radius=12)
        g._text("Start", self.start_rect.center, g.font, (30, 30, 30))
        g._text("Enter = start   Esc = quit", (self.cx, self.start_rect.bottom + 22), g.font_small, T.TEXT_DIM)
