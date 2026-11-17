"""Start menu for the pygame GUI.

Modes: Human vs Human (optional background eval), Human vs Bot and Bot vs Bot.
Wherever a bot is chosen there is the same ranked bot list (quoridor.bots.catalog:
hybrid, nnue, tuned, turbo, strongest first) with the selected bot's settings right
below it: sliders for numbers (search depth, the hybrid's phase thresholds), buttons
for choices (think time, walls searched, hash size) and on / off switches.

The menu is drawn in immediate mode: every frame registers its clickable areas
(``self._hits``) as it draws them, so the controls can change with the selected bot.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import pygame

from ..bots import MAX_DEPTH, BotSpec, engine_available
from ..bots.catalog import (RANKED_BOTS, Param, bot_choices, choice_index, default_spec, description, params_for,
                            title, value_of, with_value)
from ..bots.cpp_engine import DEFAULT_WALL_FILTER
from ..evaluation import EVAL_ENGINES
from . import theme as T

TIME_CHOICES: tuple[Optional[float], ...] = (1.0, 3.0, 5.0, 10.0, 30.0, None)
DELAY_CHOICES: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0)
MODES = ("hvh", "hvb", "bvb")
MODE_LABELS = ("Human vs Human", "Human vs Bot", "Bot vs Bot")

Action = Callable[["MatchSettings", tuple], "MatchSettings"]


@dataclass
class MatchSettings:
    mode: str = "hvh"            # "hvh", "hvb" (Human vs Bot), "bvb" (Bot vs Bot), "assistant" (Screen Assistant) or "autoplay"
    depth: int = 4               # legacy: plain alpha-beta bot from the command line (--depth)
    time_limit: Optional[float] = 5.0  # legacy: think time of that bot
    human_side: int = 0          # Human vs Bot: 0 = Red (moves first), 1 = Blue
    ml_model: Optional[str] = None  # Human vs Bot against a trained ML checkpoint instead
    ml_sims: int = 64
    wall_filter: int = DEFAULT_WALL_FILTER  # legacy: walls of that bot (2 = attack, 1 = attack + defense, 0 = all)
    show_eval: bool = True                  # Human vs Bot: show the bot's eval bar
    # Human vs Human: a bot evaluates every position in the background (1 s per move).
    hvh_eval: bool = False
    eval_engine: str = "both"               # "attack", "both", "all" (C++) or "py"
    bot_evals: tuple[bool, bool] = (True, True)  # Bot vs Bot: eval bar of each bot
    # Bot vs Bot: one spec per colour (Red, Blue) and a pause between moves so you can watch.
    bots: tuple[BotSpec, BotSpec] = field(
        default_factory=lambda: (default_spec("hybrid", 1.0), default_spec("nnue", 1.0)))
    move_delay: float = 0.5
    # Human vs Bot: the opponent (a ranked bot with its settings, or any spec). None = the legacy
    # alpha-beta bot made of depth / time_limit / wall_filter above.
    bot: Optional[BotSpec] = field(default_factory=lambda: default_spec("hybrid", 5.0))

    def opponent(self) -> BotSpec:
        """The Human vs Bot opponent as a spec."""
        if self.bot is not None:
            return self.bot
        return BotSpec(depth=self.depth, time_limit=self.time_limit).with_wall_filter(self.wall_filter)


def _time_label(t: Optional[float]) -> str:
    return "no limit" if t is None else f"{t:g} s"


class Slider:
    """Horizontal integer slider ``lo..hi`` (default 1..MAX_DEPTH)."""

    def __init__(self, rect: pygame.Rect, lo: int = 1, hi: int = MAX_DEPTH) -> None:
        self.rect = rect
        self.lo, self.hi = lo, hi
        self.dragging = False

    def value_at(self, x: float) -> int:
        t = (x - self.rect.left) / max(1, self.rect.width)
        return max(self.lo, min(self.hi, round(self.lo + t * (self.hi - self.lo))))

    def knob_x(self, value: int) -> float:
        return self.rect.left + (value - self.lo) / max(1, self.hi - self.lo) * self.rect.width

    def hit(self, pos) -> bool:
        return self.rect.inflate(20, 24).collidepoint(pos)

    def draw(self, scr, value: int, color) -> None:
        pygame.draw.rect(scr, (60, 66, 90), self.rect, border_radius=4)
        filled = self.rect.copy()
        filled.width = max(1, int(self.knob_x(value) - self.rect.left))
        pygame.draw.rect(scr, color, filled, border_radius=4)
        pygame.draw.circle(scr, color, (int(self.knob_x(value)), self.rect.centery), 10)


class Menu:
    """Draws the menu on the GUI's window and returns the chosen settings."""

    def __init__(self, gui) -> None:
        self.gui = gui
        self.screen = gui.screen
        w, h = self.screen.get_size()
        self.w, self.h = w, h
        self.cx = w / 2
        col_w = 600
        self.left = self.cx - col_w / 2
        self.col_w = col_w
        mw = (col_w - 2 * 10) / 3
        self.mode_rects = [pygame.Rect(self.left + i * (mw + 10), 150, mw, 52) for i in range(3)]
        # Human vs Human: eval on/off and the engine that evaluates
        self.hvh_eval_rects = [pygame.Rect(self.left + i * (col_w / 2 + 4), 318, col_w / 2 - 4, 36) for i in range(2)]
        ew = (col_w - 3 * 8) / 4
        self.engine_rects = [pygame.Rect(self.left + i * (ew + 8), 408, ew, 36) for i in range(4)]
        # Human vs Bot: bot column on the left, its settings on the right
        self.hvb_left = pygame.Rect(56, 226, 410, 330)
        self.hvb_right = pygame.Rect(500, 226, w - 556, 330)
        # Bot vs Bot: one column per bot
        half = (w - 2 * 40 - 40) / 2
        self.bot_cols = [pygame.Rect(40, 226, half, 330), pygame.Rect(40 + half + 40, 226, half, 330)]
        self.bot_eval_rects = [pygame.Rect(col.right - 104, col.top - 4, 104, 24) for col in self.bot_cols]
        L = self.hvb_left
        self.side_rects = [pygame.Rect(L.left + i * ((L.width - 8) / 2 + 8), L.top + 132, (L.width - 8) / 2, 36)
                           for i in range(2)]
        self.eval_rect = pygame.Rect(L.left, L.top + 204, (L.width - 8) / 2, 36)
        dw = 64
        self.delay_rects = [pygame.Rect(self.cx - 2.5 * dw - 8 + i * (dw + 4) + 90, 562, dw, 26) for i in range(5)]
        self.start_rect = pygame.Rect(self.cx - 120, h - 92, 240, 50)
        # Screen Assistant: closes this window and opens the screen-reading overlay (quoridor.vision)
        self.assistant_rect = pygame.Rect(w - 196, h - 52, 180, 36)
        # Auto-Play: the same, and the bot also enters the bottom player's moves with the mouse
        self.autoplay_rect = pygame.Rect(w - 196, h - 96, 180, 36)
        # immediate-mode state
        self._hits: list[tuple[pygame.Rect, Action]] = []
        self._sliders: dict[str, tuple[Slider, Callable]] = {}
        self._hints: list[tuple[pygame.Rect, str]] = []
        #: (bot key, setting key) -> the rects of its buttons, or [slider bar] (tests, keyboard help)
        self.controls: dict[tuple[str, str], list[pygame.Rect]] = {}
        #: bot key -> (selector rect, [rects of the folded-out list items])
        self.selectors: dict[str, tuple[pygame.Rect, list[pygame.Rect]]] = {}
        self._dropdown_draw: Optional[Callable] = None
        self.dragging: Optional[str] = None
        self.open: Optional[str] = None      # key of the bot list that is folded out
        self._paint = True

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
                        if self.open:
                            self.open = None
                            continue
                        return None
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        return s
                    if s.mode == "hvb" and s.bot is not None and event.key in (
                            pygame.K_LEFT, pygame.K_DOWN, pygame.K_RIGHT, pygame.K_UP):
                        step = 1 if event.key in (pygame.K_RIGHT, pygame.K_UP) else -1
                        s = replace(s, bot=replace(s.bot, depth=max(1, min(MAX_DEPTH, s.bot.depth + step))))
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if not self.open:
                        if self.start_rect.collidepoint(event.pos):
                            return s
                        if self.assistant_rect.collidepoint(event.pos):
                            return replace(s, mode="assistant")
                        if self.autoplay_rect.collidepoint(event.pos):
                            return replace(s, mode="autoplay")
                    s = self._click(s, event.pos)
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.dragging = None
                if event.type == pygame.MOUSEMOTION:
                    s = self._drag(s, event.pos[0])
            self.draw(s)
            pygame.display.flip()
            clock.tick(T.FPS)

    def _set_bot(self, s: MatchSettings, i: int, spec: BotSpec) -> MatchSettings:
        bots = list(s.bots)
        bots[i] = spec
        return replace(s, bots=(bots[0], bots[1]))

    def _layout(self, s: MatchSettings) -> None:
        """Register the clickable areas of ``s`` without drawing (for clicks before the first frame)."""
        self._paint = False
        try:
            self.draw(s)
        finally:
            self._paint = True

    def _click(self, s: MatchSettings, pos) -> MatchSettings:
        self._layout(s)
        if self.open:  # a folded-out bot list takes every click; outside it closes the list
            hits = [h for h in self._hits if h[0].width and getattr(h[1], "dropdown_item", False)]
            self.open = None
            for rect, action in hits:
                if rect.collidepoint(pos):
                    return action(s, pos)
            return s
        for rect, action in reversed(self._hits):
            if rect.collidepoint(pos):
                return action(s, pos)
        return s

    def _drag(self, s: MatchSettings, x: float) -> MatchSettings:
        if self.dragging and self.dragging in self._sliders:
            slider, apply = self._sliders[self.dragging]
            return apply(s, slider.value_at(x))
        return s

    # ------------------------------------------------------------------ #
    # drawing helpers (each also registers its click area)
    # ------------------------------------------------------------------ #
    def _text(self, text, pos, font=None, color=T.TEXT, anchor="center", max_width=None) -> None:
        if self._paint:
            self.gui._text(text, pos, font or self.gui.font_small, color, anchor=anchor, max_width=max_width)

    def _fit(self, text: str, width: float, font=None) -> str:
        font = font or self.gui.font_small
        if font.size(text)[0] <= width:
            return text
        for a, b in ((" s", "s"), ("no limit", "none"), (" MB", "M"), ("Att + def", "A+D")):
            text = text.replace(a, b)
        while font.size(text)[0] > width and len(text) > 2:
            text = text[:-2] + "…"
        return text

    def _button(self, rect, label: str, selected: bool, color=None, action: Optional[Action] = None,
                font=None) -> None:
        if action is not None:
            self._hits.append((pygame.Rect(rect), action))
        if not self._paint:
            return
        mouse = pygame.mouse.get_pos()
        base = (58, 64, 88) if selected else T.PANEL
        if rect.collidepoint(mouse) and not selected and not self.open:
            base = (44, 49, 69)
        pygame.draw.rect(self.screen, base, rect, border_radius=8)
        if selected:
            pygame.draw.rect(self.screen, color or T.PANEL_ACTIVE_EDGE, rect, width=2, border_radius=8)
        font = font or (self.gui.font_small if rect.height < 50 else self.gui.font)
        self.gui._text(self._fit(label, rect.width - 6, font), rect.center, font, T.TEXT if selected else T.TEXT_DIM)

    def _slider(self, key: str, rect: pygame.Rect, lo: int, hi: int, value: int, color,
                apply: Callable[[MatchSettings, int], MatchSettings]) -> None:
        slider = Slider(rect, lo, hi)
        self._sliders[key] = (slider, apply)

        def press(s, pos, key=key, slider=slider, apply=apply):
            self.dragging = key
            return apply(s, slider.value_at(pos[0]))

        self._hits.append((rect.inflate(20, 24), press))
        if self._paint:
            slider.draw(self.screen, value, color)

    def _segmented(self, rect: pygame.Rect, labels, current: int, color, pick: Callable[[MatchSettings, int], MatchSettings]):
        n = len(labels)
        gap = 3
        room = rect.width - gap * (n - 1)
        # widths in proportion to the labels, so "no limit" gets more room than "1 s"
        natural = [self.gui.font_small.size(label)[0] + 12 for label in labels]
        widths = [room * w / sum(natural) for w in natural]
        if min(natural) * n <= room:  # equal buttons when that fits every label
            widths = [room / n] * n if max(natural) <= room / n else widths
        x = rect.left
        for i, label in enumerate(labels):
            r = pygame.Rect(round(x), rect.top, round(widths[i]), rect.height)
            x += widths[i] + gap
            self._button(r, label, i == current, color, lambda s, _p, i=i: pick(s, i))

    def _bot_selector(self, key: str, rect: pygame.Rect, spec: BotSpec, color,
                      set_spec: Callable[[MatchSettings, BotSpec], MatchSettings], time_default: Optional[float]):
        """The ranked bot list (folds out below ``rect``)."""
        choices = bot_choices(spec, time_default)
        idx = choice_index(spec, choices)
        name = choices[idx][0].split(" - ")[0]

        def toggle(s, _p):
            self.open = None if self.open == key else key
            return s

        self._button(rect, f"{name}   v", True, color, toggle, self.gui.font)
        self.selectors[key] = (pygame.Rect(rect), [])
        if self.open != key:
            return
        item_h = 44
        items = []
        for i, (label, choice) in enumerate(choices):
            r = pygame.Rect(rect.left, rect.bottom + 4 + i * item_h, rect.width, item_h - 2)

            def pick(s, _p, choice=choice):
                return set_spec(s, choice)

            pick.dropdown_item = True  # type: ignore[attr-defined]
            self._hits.append((r, pick))
            self.selectors[key][1].append(r)
            items.append((r, label, i == idx))

        def draw_list():
            frame = pygame.Rect(rect.left - 2, rect.bottom + 2, rect.width + 4, len(items) * item_h + 4)
            pygame.draw.rect(self.screen, (16, 18, 26), frame, border_radius=8)
            pygame.draw.rect(self.screen, color, frame, width=1, border_radius=8)
            mouse = pygame.mouse.get_pos()
            for r, label, sel in items:
                if sel or r.collidepoint(mouse):
                    pygame.draw.rect(self.screen, (58, 64, 88) if sel else (44, 49, 69), r, border_radius=6)
                head, _, desc = label.partition(" - ")
                self.gui._text(head, (r.left + 12, r.top + 13), self.gui.font_small, T.TEXT, anchor="midleft")
                self.gui._text(self._fit(desc, r.width - 24), (r.left + 12, r.top + 31), self.gui.font_small,
                               T.TEXT_DIM, anchor="midleft")

        self._dropdown_draw = draw_list

    def _bot_settings(self, key: str, area: pygame.Rect, spec: BotSpec, color,
                      set_spec: Callable[[MatchSettings, BotSpec], MatchSettings], row_h: int,
                      label_w: int) -> None:
        """One row per setting of ``spec`` (see quoridor.bots.catalog.params_for)."""
        params = params_for(spec)
        if not params:
            self._text(spec.label(), (area.left, area.top + 12), anchor="midleft")
            self._text("(set with --bot1 / --bot2)", (area.left, area.top + 34), color=T.TEXT_DIM, anchor="midleft")
            return
        for i, p in enumerate(params):
            y = area.top + i * row_h
            value = value_of(spec, p)
            label = f"{p.label}: {p.text(value)}" if p.kind == "range" else p.label
            self._text(self._fit(label, label_w - 8), (area.left, y + row_h / 2 - 2), color=T.TEXT_DIM, anchor="midleft")
            self._hints.append((pygame.Rect(area.left, y, area.width, row_h), f"{p.label}: {p.hint}"))
            ctrl = pygame.Rect(area.left + label_w, y + 3, area.width - label_w, row_h - 8)
            first = len(self._hits)
            if p.kind == "range":
                bar = pygame.Rect(ctrl.left + 10, ctrl.centery - 4, ctrl.width - 20, 8)
                self._slider(f"{key}:{p.key}", bar, p.lo, p.hi, int(value), color,
                             lambda s, v, p=p: set_spec(s, with_value(self._spec_of(key, s), p, v)))
            elif p.kind == "toggle":
                self._segmented(ctrl, ("On", "Off"), 0 if value else 1, color,
                                lambda s, i, p=p: set_spec(s, with_value(self._spec_of(key, s), p, i == 0)))
            else:
                cur = p.values.index(value) if value in p.values else -1
                self._segmented(ctrl, p.labels, cur, color,
                                lambda s, i, p=p: set_spec(s, with_value(self._spec_of(key, s), p, p.values[i])))
            self.controls[(key, p.key)] = [h[0] for h in self._hits[first:]]

    def _spec_of(self, key: str, s: MatchSettings) -> BotSpec:
        if key == "hvb":
            return s.opponent()
        return s.bots[int(key[-1])]

    # ------------------------------------------------------------------ #
    def draw(self, s: MatchSettings) -> None:
        self._hits, self._sliders, self._hints, self._dropdown_draw = [], {}, [], None
        self.controls, self.selectors = {}, {}
        g, scr = self.gui, self.screen
        if self._paint:
            scr.fill(T.BACKGROUND)
        self._text("Quoridor", (self.cx, 70), g.font_big, T.TEXT)
        for i, (mode, label, r) in enumerate(zip(MODES, MODE_LABELS, self.mode_rects)):
            self._button(r, label, s.mode == mode, action=lambda st, _p, m=mode: replace(st, mode=m))

        if s.mode == "hvb":
            self._draw_hvb(s)
        elif s.mode == "bvb":
            self._draw_bvb(s)
        else:
            self._draw_hvh(s)

        # the help line under the title: what the setting under the mouse does
        hint = "Choose a game mode"
        if self._paint:
            mouse = pygame.mouse.get_pos()
            hint = next((h for r, h in self._hints if r.collidepoint(mouse)), hint)
        self._text(hint, (self.cx, 118), g.font_small, T.TEXT_DIM)
        if self._paint:
            pygame.draw.rect(scr, T.PANEL_ACTIVE_EDGE, self.start_rect, border_radius=12)
        self._text("Start", self.start_rect.center, g.font, (30, 30, 30))
        self._text("Enter = start   Esc = quit", (self.cx, self.start_rect.bottom + 18), g.font_small, T.TEXT_DIM)
        self._button(self.assistant_rect, "Screen Assistant", False)
        self._button(self.autoplay_rect, "Auto-Play Mode", False)
        if self._dropdown_draw is not None and self._paint:
            self._dropdown_draw()  # on top of everything

    def _draw_hvh(self, s: MatchSettings) -> None:
        g = self.gui
        self._text("Two people take turns on this computer.", (self.cx, 250), g.font_small, T.TEXT_DIM)
        self._text("Eval bar", (self.left, 300), g.font_small, T.TEXT_DIM, anchor="midleft")
        self._text("a bot evaluates every position in the background", (self.left + self.col_w, 300),
                   g.font_small, T.TEXT_DIM, anchor="midright")
        for on, label, r in zip((False, True), ("Off", "Show eval"), self.hvh_eval_rects):
            self._button(r, label, s.hvh_eval == on, action=lambda st, _p, on=on: replace(st, hvh_eval=on))
        self._text("Engine that evaluates (1 s per move)", (self.left, 390), g.font_small,
                   T.TEXT_DIM if s.hvh_eval else (90, 96, 112), anchor="midleft")
        for (key, label), r in zip(EVAL_ENGINES, self.engine_rects):
            short = label.replace("C++ ", "").replace("Attack + defense", "Att + def").replace("attack + defense",
                                                                                                  "att + def")
            self._button(r, short[:1].upper() + short[1:], s.hvh_eval and s.eval_engine == key,
                         action=lambda st, _p, key=key: replace(st, eval_engine=key, hvh_eval=True))
        note = ("The first three use the C++ engine with that wall mode." if engine_available()
                else "The C++ engine is not built, so the Python bot evaluates.")
        self._text(note, (self.cx, 466), g.font_small, T.TEXT_DIM)
        self._text("Positive = good for the colour shown; M3 = forced win in 3 moves, -M3 = forced loss.",
                   (self.cx, 490), g.font_small, T.TEXT_DIM)

    def _draw_hvb(self, s: MatchSettings) -> None:
        g = self.gui
        L, R = self.hvb_left, self.hvb_right
        bot_color = T.PAWN_COLORS[1 - s.human_side]
        spec = s.opponent()
        self._text("Bot (strongest first)", (L.left, L.top + 6), g.font_small, T.TEXT_DIM, anchor="midleft")
        sel = pygame.Rect(L.left, L.top + 20, L.width, 40)
        desc = description(spec.model) if spec.kind == "profile" else spec.label()
        if s.ml_model:
            desc = f"Playing the ML agent {s.ml_model} ({s.ml_sims} sims); pick a bot to switch."
        self._text(desc, (L.left, L.top + 76), g.font_small, T.TEXT_DIM, anchor="midleft", max_width=L.width)
        self._text("You play", (L.left, L.top + 118), g.font_small, T.TEXT_DIM, anchor="midleft")
        for side, label in ((0, "Red (moves first)"), (1, "Blue")):
            self._button(self.side_rects[side], label, s.human_side == side, T.PAWN_COLORS[side],
                         lambda st, _p, side=side: replace(st, human_side=side))
        self._text("Bot's eval bar", (L.left, L.top + 190), g.font_small, T.TEXT_DIM, anchor="midleft")
        self._button(self.eval_rect, "Eval: on" if s.show_eval else "Eval: off",
                     s.show_eval, bot_color, lambda st, _p: replace(st, show_eval=not st.show_eval))
        self._text("Ranking from 5 s/move round-robins: hybrid +222, nnue +209, tuned +36, turbo -20 Elo vs "
                   "the old engine. Hover a setting for help.", (L.left, L.top + 262), g.font_small, T.TEXT_DIM,
                   anchor="midleft", max_width=L.width)
        self._text("Bot settings", (R.left, R.top + 6), g.font_small, T.TEXT_DIM, anchor="midleft")
        self._bot_settings("hvb", pygame.Rect(R.left, R.top + 20, R.width, R.height - 20), spec, bot_color,
                           lambda st, sp: replace(st, bot=sp), 35, 190)
        self._bot_selector("hvb", sel, spec, bot_color, lambda st, sp: replace(st, bot=sp, ml_model=None),
                           spec.time_limit if spec.kind in ("ab", "profile") else 5.0)

    def _draw_bvb(self, s: MatchSettings) -> None:
        g = self.gui
        for b, col in enumerate(self.bot_cols):
            spec = s.bots[b]
            color = T.PAWN_COLORS[b]
            if self._paint:
                pygame.draw.circle(self.screen, color, (col.left + 9, col.top + 8), 8)
            self._text(f"Bot {b + 1} - {'Red, moves first' if b == 0 else 'Blue'}", (col.left + 24, col.top + 8),
                       g.font_small, T.TEXT, anchor="midleft")
            self._button(self.bot_eval_rects[b], "Eval: on" if s.bot_evals[b] else "Eval: off", s.bot_evals[b], color,
                         lambda st, _p, b=b: replace(st, bot_evals=tuple(not e if i == b else e
                                                                         for i, e in enumerate(st.bot_evals))))
            self._bot_settings(f"bvb{b}", pygame.Rect(col.left, col.top + 66, col.width, 270), spec, color,
                               lambda st, sp, b=b: self._set_bot(st, b, sp), 29, 170)
        for b, col in enumerate(self.bot_cols):  # the lists fold out over the settings
            sel = pygame.Rect(col.left, col.top + 24, col.width, 34)
            self._bot_selector(f"bvb{b}", sel, s.bots[b], T.PAWN_COLORS[b],
                               lambda st, sp, b=b: self._set_bot(st, b, sp),
                               s.bots[b].time_limit if s.bots[b].kind in ("ab", "profile") else 1.0)
        self._text("Pause between moves", (self.delay_rects[0].left - 10, self.delay_rects[0].centery), g.font_small,
                   T.TEXT_DIM, anchor="midright")
        for d, r in zip(DELAY_CHOICES, self.delay_rects):
            self._button(r, "none" if d == 0 else f"{d:g} s", s.move_delay == d,
                         action=lambda st, _p, d=d: replace(st, move_delay=d))


__all__ = ["MatchSettings", "Menu", "Slider", "RANKED_BOTS", "title", "TIME_CHOICES", "DELAY_CHOICES", "Param"]
