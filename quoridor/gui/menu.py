"""Start menu for the pygame GUI.

Modes: Human vs Human, Human vs Bot (depth slider, think time, side) and
Bot vs Bot (independent depth + think time for each bot, move delay).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import pygame

from ..bots import MAX_DEPTH, BotSpec
from . import theme as T

TIME_CHOICES: tuple[Optional[float], ...] = (1.0, 3.0, 5.0, 10.0, 30.0, None)
DELAY_CHOICES: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0)
MODES = ("hvh", "hvb", "bvb")
MODE_LABELS = ("Human vs Human", "Human vs Bot", "Bot vs Bot")


@dataclass
class MatchSettings:
    mode: str = "hvh"            # "hvh", "hvb" (Human vs Bot) or "bvb" (Bot vs Bot)
    depth: int = 4               # Human vs Bot: search depth, 1..30
    time_limit: Optional[float] = 5.0  # Human vs Bot: seconds per bot move, None = no limit
    human_side: int = 0          # Human vs Bot: 0 = Red (moves first), 1 = Blue
    ml_model: Optional[str] = None  # Human vs Bot against a trained ML checkpoint instead of alpha-beta
    ml_sims: int = 64
    # Bot vs Bot: one spec per colour (Red, Blue) and a pause between moves so you can watch.
    bots: tuple[BotSpec, BotSpec] = field(default_factory=lambda: (BotSpec(depth=2), BotSpec(depth=4)))
    move_delay: float = 0.5


def _time_label(t: Optional[float]) -> str:
    return "no limit" if t is None else f"{t:g} s"


class Slider:
    """Horizontal 1..MAX_DEPTH slider."""

    def __init__(self, rect: pygame.Rect) -> None:
        self.rect = rect
        self.dragging = False

    def value_at(self, x: float) -> int:
        t = (x - self.rect.left) / self.rect.width
        return max(1, min(MAX_DEPTH, round(1 + t * (MAX_DEPTH - 1))))

    def knob_x(self, value: int) -> float:
        return self.rect.left + (value - 1) / (MAX_DEPTH - 1) * self.rect.width

    def hit(self, pos) -> bool:
        return self.rect.inflate(20, 30).collidepoint(pos)

    def draw(self, scr, value: int, color) -> None:
        pygame.draw.rect(scr, (60, 66, 90), self.rect, border_radius=4)
        filled = self.rect.copy()
        filled.width = max(1, self.knob_x(value) - self.rect.left)
        pygame.draw.rect(scr, color, filled, border_radius=4)
        pygame.draw.circle(scr, color, (self.knob_x(value), self.rect.centery), 12)


class Menu:
    """Draws the menu on the GUI's window and returns the chosen settings."""

    def __init__(self, gui) -> None:
        self.gui = gui
        self.screen = gui.screen
        w, h = self.screen.get_size()
        self.cx = w / 2
        col_w = 600
        self.left = self.cx - col_w / 2
        self.col_w = col_w
        mw = (col_w - 2 * 10) / 3
        self.mode_rects = [pygame.Rect(self.left + i * (mw + 10), 150, mw, 52) for i in range(3)]
        # Human vs Bot
        self.slider = Slider(pygame.Rect(self.left + 10, 285, col_w - 20, 8))
        bw = (col_w - 5 * 8) / 6
        self.time_rects = [pygame.Rect(self.left + i * (bw + 8), 360, bw, 40) for i in range(6)]
        self.side_rects = [pygame.Rect(self.left, 450, col_w / 2 - 6, 40),
                           pygame.Rect(self.left + col_w / 2 + 6, 450, col_w / 2 - 6, 40)]
        # Bot vs Bot: two columns
        half = (col_w - 30) / 2
        self.bot_cols = [self.left, self.left + half + 30]
        self.bot_sliders = [Slider(pygame.Rect(x + 10, 300, half - 20, 8)) for x in self.bot_cols]
        tw = (half - 5 * 4) / 6
        self.bot_time_rects = [[pygame.Rect(x + i * (tw + 4), 360, tw, 32) for i in range(6)]
                               for x in self.bot_cols]
        dw = (col_w - 4 * 8) / 5
        self.delay_rects = [pygame.Rect(self.left + i * (dw + 8), 450, dw, 36) for i in range(5)]
        self.start_rect = pygame.Rect(self.cx - 120, h - 120, 240, 54)

    # ------------------------------------------------------------------ #
    def run(self, settings: MatchSettings) -> Optional[MatchSettings]:
        """Event loop; returns the settings on Start, None if the window closes."""
        s = replace(settings)
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
                    s = self._click(s, event.pos)
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    for sl in (self.slider, *self.bot_sliders):
                        sl.dragging = False
                if event.type == pygame.MOUSEMOTION:
                    s = self._drag(s, event.pos[0])
            self.draw(s)
            pygame.display.flip()
            clock.tick(T.FPS)

    def _set_bot(self, s: MatchSettings, i: int, **changes) -> MatchSettings:
        bots = list(s.bots)
        bots[i] = replace(bots[i], **changes)
        return replace(s, bots=(bots[0], bots[1]))

    def _click(self, s: MatchSettings, pos) -> MatchSettings:
        for i, r in enumerate(self.mode_rects):
            if r.collidepoint(pos):
                s = replace(s, mode=MODES[i])
        if s.mode == "hvb":
            if self.slider.hit(pos):
                self.slider.dragging = True
                s = replace(s, depth=self.slider.value_at(pos[0]))
            for i, r in enumerate(self.time_rects):
                if r.collidepoint(pos):
                    s = replace(s, time_limit=TIME_CHOICES[i])
            for i, r in enumerate(self.side_rects):
                if r.collidepoint(pos):
                    s = replace(s, human_side=i)
        elif s.mode == "bvb":
            for b in (0, 1):
                if s.bots[b].kind != "ab":
                    continue
                if self.bot_sliders[b].hit(pos):
                    self.bot_sliders[b].dragging = True
                    s = self._set_bot(s, b, depth=self.bot_sliders[b].value_at(pos[0]))
                for i, r in enumerate(self.bot_time_rects[b]):
                    if r.collidepoint(pos):
                        s = self._set_bot(s, b, time_limit=TIME_CHOICES[i])
            for i, r in enumerate(self.delay_rects):
                if r.collidepoint(pos):
                    s = replace(s, move_delay=DELAY_CHOICES[i])
        return s

    def _drag(self, s: MatchSettings, x: float) -> MatchSettings:
        if self.slider.dragging:
            s = replace(s, depth=self.slider.value_at(x))
        for b, sl in enumerate(self.bot_sliders):
            if sl.dragging:
                s = self._set_bot(s, b, depth=sl.value_at(x))
        return s

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
        for mode, label, r in zip(MODES, MODE_LABELS, self.mode_rects):
            self._button(r, label, s.mode == mode)

        if s.mode == "hvb":
            self._draw_hvb(s)
        elif s.mode == "bvb":
            self._draw_bvb(s)
        else:
            g._text("Two people take turns on this computer.", (self.cx, 260), g.font_small, T.TEXT_DIM)

        pygame.draw.rect(scr, T.PANEL_ACTIVE_EDGE, self.start_rect, border_radius=12)
        g._text("Start", self.start_rect.center, g.font, (30, 30, 30))
        g._text("Enter = start   Esc = quit", (self.cx, self.start_rect.bottom + 22), g.font_small, T.TEXT_DIM)

    def _draw_hvb(self, s: MatchSettings) -> None:
        g = self.gui
        g._text(f"Search depth: {s.depth}", (self.left, 250), g.font, T.TEXT, anchor="midleft")
        g._text("1-30 plies (arrow keys work too)", (self.left + self.col_w, 250), g.font_small,
                T.TEXT_DIM, anchor="midright")
        self.slider.draw(self.screen, s.depth, T.PANEL_ACTIVE_EDGE)
        g._text("faster", (self.slider.rect.left, 312), g.font_small, T.TEXT_DIM, anchor="midleft")
        g._text("stronger", (self.slider.rect.right, 312), g.font_small, T.TEXT_DIM, anchor="midright")
        g._text("Think time per bot move", (self.left, 342), g.font_small, T.TEXT_DIM, anchor="midleft")
        for t, r in zip(TIME_CHOICES, self.time_rects):
            self._button(r, _time_label(t), s.time_limit == t)
        g._text("You play", (self.left, 432), g.font_small, T.TEXT_DIM, anchor="midleft")
        self._button(self.side_rects[0], "Red (moves first)", s.human_side == 0, T.PAWN_COLORS[0])
        self._button(self.side_rects[1], "Blue", s.human_side == 1, T.PAWN_COLORS[1])
        g._text("The bot plays the best move from the deepest search it completes within the time.",
                (self.cx, 525), g.font_small, T.TEXT_DIM)

    def _draw_bvb(self, s: MatchSettings) -> None:
        g = self.gui
        for b, x in enumerate(self.bot_cols):
            spec = s.bots[b]
            color = T.PAWN_COLORS[b]
            pygame.draw.circle(self.screen, color, (x + 10, 245), 9)
            g._text(f"Bot {b + 1} - {'Red, moves first' if b == 0 else 'Blue'}", (x + 26, 245),
                    g.font_small, T.TEXT, anchor="midleft")
            if spec.kind != "ab":
                g._text(spec.label(), (x, 290), g.font_small, T.TEXT, anchor="midleft")
                g._text("(set with --bot1/--bot2)", (x, 312), g.font_small, T.TEXT_DIM, anchor="midleft")
                continue
            g._text(f"Depth {spec.depth}", (x, 276), g.font, T.TEXT, anchor="midleft")
            self.bot_sliders[b].draw(self.screen, spec.depth, color)
            g._text("Think time", (x, 342), g.font_small, T.TEXT_DIM, anchor="midleft")
            for t, r in zip(TIME_CHOICES, self.bot_time_rects[b]):
                self._button(r, "none" if t is None else f"{t:g}s", spec.time_limit == t, color)
        g._text("Pause between moves (so you can follow the game)", (self.left, 432), g.font_small,
                T.TEXT_DIM, anchor="midleft")
        for d, r in zip(DELAY_CHOICES, self.delay_rects):
            self._button(r, "none" if d == 0 else f"{d:g} s", s.move_delay == d)
        g._text("While watching: Space = pause/resume, R = back to this menu.  "
                "For many fast games use: python -m quoridor.botmatch",
                (self.cx, 515), g.font_small, T.TEXT_DIM)
