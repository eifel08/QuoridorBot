"""Pygame front-end.

:class:`QuoridorGUI` is a :class:`~quoridor.game.GameObserver`: it only ever
reads immutable :class:`~quoridor.state.BoardState` snapshots handed to it by
the engine, and it reports human decisions back as plain ``Move`` objects via
:class:`~quoridor.gui.player.GuiHumanPlayer`. It never touches the engine's
board and never flips coordinates: row 0 is always drawn at the top.
"""

from __future__ import annotations

import math
from typing import Optional

import pygame

from ..evaluation import BackgroundEvaluator, Eval
from ..game import Game, GameObserver
from ..moves import Move, Orientation, PawnMove, WallMove
from ..player import QuitGame
from ..state import BoardState
from . import theme as T
from .anim import Tween, ease_out_back, ease_out_cubic
from .layout import Layout, Rect

DRAG_THRESHOLD = 5


class RestartGame(QuitGame):
    """Raised from the GUI when the user asks for a new game."""


def _blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))  # type: ignore[return-value]


def _count(n: int) -> str:
    """776582 -> '777k', 2272464 -> '2.3M'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"{n / 1000:.0f}k"
    return f"{n:,}"


class QuoridorGUI(GameObserver):
    """Window, rendering and mouse interaction."""

    def __init__(self, layout: Optional[Layout] = None) -> None:
        pygame.display.init()
        pygame.font.init()
        self.layout = layout or Layout()
        self.screen = pygame.display.set_mode((self.layout.width, self.layout.height))
        pygame.display.set_caption("Quoridor")
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.Font(None, 44)
        self.font = pygame.font.Font(None, 30)
        self.font_small = pygame.font.Font(None, 22)

        self.state: Optional[BoardState] = None
        self.names: tuple[str, str] = ("Player 1", "Player 2")
        self.anims: list[Tween] = []
        self.message = ""
        self.message_until = 0
        self.human_turn = False
        self.restart_requested = False
        self.bot_thinking = False
        self.bot_info: Optional[dict] = None
        self.bots_only = False   # Bot vs Bot: show playback hints instead of human controls
        self.move_delay = 0.0    # Bot vs Bot: minimum seconds per move so the game can be followed
        self.paused = False
        self.quit_requested = False
        self.mouse: tuple[int, int] = (0, 0)
        self._origin: Optional[tuple[float, ...]] = None  # pixel start of the next move animation
        # Eval display. Slot 0 is the bar left of the board (Red's side), slot 1 the bar on the
        # right (Blue's). A bot's own eval goes in its slot; the Human vs Human analysis uses slot 0.
        self.eval_slots = [False, False]
        self.eval_bars: list[Optional[tuple[Eval, Optional[int]]]] = [None, None]  # (eval, owner or None)
        self.evaluator: Optional[BackgroundEvaluator] = None
        self._reset_interaction()

    # ================================================================== #
    # Eval display
    # ================================================================== #
    def set_eval_display(self, slots: tuple[bool, bool], evaluator: Optional[BackgroundEvaluator] = None) -> None:
        """Which eval bars to draw, and the Human vs Human analysis engine (if any)."""
        if self.evaluator is not None and self.evaluator is not evaluator:
            self.evaluator.close()
        self.evaluator = evaluator
        self.eval_slots = list(slots)
        self.eval_bars = [None, None]

    def set_bot_eval(self, player: int, ev: Optional[Eval]) -> None:
        """A bot's own search score after its move (shown in its colour on its side)."""
        if ev is not None and self.eval_slots[player]:
            self.eval_bars[player] = (ev, player)

    def _analysis_result(self, ev: Eval, move_number: int) -> None:
        # Called from the evaluator thread: only keep results for the position on the board.
        st = self.state
        if st is not None and st.winner is None and st.move_number == move_number:
            self.eval_bars[0] = (ev, None)

    # ================================================================== #
    # GameObserver hooks
    # ================================================================== #
    def on_game_start(self, game: Game, state: BoardState) -> None:
        self.names = game.names
        self.state = state
        self.bot_info = None
        self.eval_bars = [None, None]
        self.anims.clear()
        self.message = ""
        self._frame()

    def on_turn_start(self, game: Game, state: BoardState) -> None:
        self.state = state
        if self.evaluator is not None:
            self.evaluator.submit(state)
        self._reset_interaction()
        self._frame()

    def on_invalid_move(self, game: Game, player: int, move: Move, reason: str) -> None:
        self.flash(reason)

    def on_move(self, game: Game, player: int, move: Move, before: BoardState, after: BoardState) -> None:
        L = self.layout
        if isinstance(move, PawnMove):
            start = self._origin or L.square_center(*before.pawns[player])
            end = L.square_center(move.row, move.col)
            dist = math.dist(start, end) / L.pitch
            duration = T.PAWN_ANIM_MS * (1 + 0.25 * max(dist - 1, 0))
            self.anims.append(Tween(start, end, duration, "pawn", player))
        else:
            wall = (move.row, move.col, move.orientation)
            start = self._origin or L.stub_rect(player, before.walls_left[player] - 1)
            self.anims.append(Tween(start, L.wall_rect(wall), T.WALL_SNAP_MS, "wall", wall, ease_out_back))
        self._origin = None
        self.state = after
        self.human_turn = False
        self._reset_interaction()
        while self.anims:
            self._frame()

    def on_game_over(self, game: Game, state: BoardState) -> None:
        self.state = state
        self.human_turn = False
        if self.evaluator is not None and state.winner is not None:
            self.eval_bars[0] = (Eval.game_won(state.winner), None)
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.quit_requested = True
                    return
                if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1) or (
                    event.type == pygame.KEYDOWN and event.key in (pygame.K_r, pygame.K_RETURN, pygame.K_SPACE)
                ):
                    self.restart_requested = True
                    return
            self._frame(pump=False)

    # ================================================================== #
    # Human input
    # ================================================================== #
    def wait_for_bot(self, state: BoardState, thread) -> None:
        """Keep the window alive (and closable) until the bot thread finishes.

        In Bot vs Bot mode this also waits out ``move_delay`` (so the game can
        be followed) and holds while the game is paused.
        """
        self.state = state
        self.human_turn = False
        self.bot_thinking = True
        start = pygame.time.get_ticks()
        try:
            while thread.is_alive():
                self._frame()
            self.bot_thinking = False
            while self.paused or pygame.time.get_ticks() - start < self.move_delay * 1000:
                self._frame()
        finally:
            self.bot_thinking = False

    def wait_for_move(self, state: BoardState) -> Move:
        """Run the event loop until the active human picks a legal move."""
        self.state = state
        self.human_turn = True
        self._reset_interaction()
        while True:
            for event in pygame.event.get():
                move = self.handle_event(event)
                if move is not None:
                    return move
            self._frame(pump=False)

    def flash(self, text: str) -> None:
        self.message = text
        self.message_until = pygame.time.get_ticks() + T.MESSAGE_MS

    def _reset_interaction(self) -> None:
        self.selected = False
        self.press_pos: Optional[tuple[int, int]] = None
        self.pawn_drag = False
        self.wall_drag = False
        self.wall_preview: Optional[tuple[int, int, Orientation]] = None
        self.wall_preview_ok = False

    def _handle_global(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.quit_requested = True
            raise QuitGame()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            self.restart_requested = True
            raise RestartGame()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and self.bots_only:
            self.paused = not self.paused
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and \
                Layout.contains(self.layout.button_rect, *event.pos):
            self.restart_requested = True
            raise RestartGame()

    def handle_event(self, event: pygame.event.Event) -> Optional[Move]:
        """Process one event during a human turn. Returns a move when chosen."""
        self._handle_global(event)
        st, L = self.state, self.layout
        assert st is not None
        me = st.current_player

        if event.type == pygame.MOUSEMOTION:
            self.mouse = event.pos
            if self.press_pos and not self.pawn_drag and math.dist(event.pos, self.press_pos) > DRAG_THRESHOLD:
                self.pawn_drag = True
                self.selected = True
            if self.wall_drag:
                self._update_wall_preview()
            return None

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.mouse = event.pos
            if self._over_own_pawn(event.pos):
                self.press_pos = event.pos
                return None
            if st.walls_left[me] > 0 and Layout.contains(L.reserve_rect(me), *event.pos):
                self.selected = False
                self.wall_drag = True
                self._update_wall_preview()
                return None
            sq = L.square_at(*event.pos)
            if sq is not None and st.is_valid_pawn_move(sq):
                return PawnMove(*sq)
            if sq is not None and self.selected and sq != st.pawns[me]:
                self.flash("Your pawn can't move there.")
            self.selected = False
            return None

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.mouse = event.pos
            if self.pawn_drag:
                self.pawn_drag = False
                self.press_pos = None
                sq = L.square_at(*event.pos, forgiving=True)
                if sq is not None and st.is_valid_pawn_move(sq):
                    self._origin = tuple(event.pos)
                    return PawnMove(*sq)
                if sq is not None and sq != st.pawns[me]:
                    self.flash("Your pawn can't move there.")
                self.anims.append(Tween(event.pos, L.square_center(*st.pawns[me]),
                                        T.PAWN_ANIM_MS, "pawn", me))
                return None
            if self.press_pos:
                self.press_pos = None
                self.selected = not self.selected
                return None
            if self.wall_drag:
                return self._drop_wall()
            return None

        if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 3) or \
                (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            self._cancel_drags()
        return None

    # ------------------------------------------------------------------ #
    def _over_own_pawn(self, pos: tuple[int, int]) -> bool:
        st = self.state
        assert st is not None
        if self._animating("pawn", st.current_player):
            return False
        center = self.layout.square_center(*st.pawns[st.current_player])
        return math.dist(pos, center) <= self.layout.cell * 0.45

    def _update_wall_preview(self) -> None:
        st = self.state
        assert st is not None
        cands = self.layout.wall_candidates(*self.mouse)
        self.wall_preview, self.wall_preview_ok = None, False
        for wall in cands:
            if st.is_valid_wall(wall):
                self.wall_preview, self.wall_preview_ok = wall, True
                return
        if cands:
            self.wall_preview = cands[0]

    def _floating_wall_rect(self) -> Rect:
        L = self.layout
        if self.wall_preview is not None:
            _, _, w, h = L.wall_rect(self.wall_preview)
        else:
            w, h = L.gap, 2 * L.cell + L.gap
        mx, my = self.mouse
        return (mx - w / 2, my - h / 2, w, h)

    def _drop_wall(self) -> Optional[Move]:
        st = self.state
        assert st is not None
        me = st.current_player
        rect = self._floating_wall_rect()
        wall, ok = self.wall_preview, self.wall_preview_ok
        self.wall_drag = False
        self.wall_preview = None
        if wall is not None and ok:
            self._origin = rect
            return WallMove(wall[0], wall[1], wall[2])
        if wall is not None:
            # Ask a throw-away copy why the wall is illegal; the real board is untouched.
            reason = st.to_board().wall_error(me, wall) or "Illegal wall."
            self.flash(reason)
        target = self.layout.stub_rect(me, st.walls_left[me] - 1)
        self.anims.append(Tween(rect, target, T.WALL_RETURN_MS, "wall_return", me, ease_out_cubic))
        return None

    def _cancel_drags(self) -> None:
        st = self.state
        assert st is not None
        me = st.current_player
        if self.wall_drag:
            rect = self._floating_wall_rect()
            target = self.layout.stub_rect(me, st.walls_left[me] - 1)
            self.anims.append(Tween(rect, target, T.WALL_RETURN_MS, "wall_return", me))
        if self.pawn_drag:
            self.anims.append(Tween(self.mouse, self.layout.square_center(*st.pawns[me]),
                                    T.PAWN_ANIM_MS, "pawn", me))
        self._reset_interaction()

    def _animating(self, kind: str, data=None) -> bool:
        return any(a.kind == kind and (data is None or a.data == data) for a in self.anims)

    # ================================================================== #
    # Frame loop
    # ================================================================== #
    def _frame(self, pump: bool = True) -> None:
        dt = self.clock.tick(T.FPS)
        if pump:
            for event in pygame.event.get():
                self._handle_global(event)
        for a in self.anims:
            a.update(dt)
        self.draw()
        self.anims = [a for a in self.anims if not a.done]
        pygame.display.flip()

    # ================================================================== #
    # Drawing (read-only on self.state)
    # ================================================================== #
    def draw(self) -> None:
        st = self.state
        if st is None:
            return
        s = self.screen
        s.fill(T.BACKGROUND)
        self._draw_board(st)
        self._draw_eval_bars()
        self._draw_walls(st)
        if self.wall_drag and self.wall_preview is not None:
            color = T.PREVIEW_OK if self.wall_preview_ok else T.PREVIEW_BAD
            self._draw_wall_rect(self.layout.wall_rect(self.wall_preview), color,
                                 alpha=150 if self.wall_preview_ok else 190)
        if self.human_turn and (self.selected or self.pawn_drag) and not self.anims:
            self._draw_move_dots(st)
        self._draw_pawns(st)
        self._draw_panel(st)
        # Floating things on top.
        for a in self.anims:
            if a.kind in ("wall", "wall_return"):
                self._draw_wall_rect(a.value)
        if self.wall_drag and not self.wall_preview_ok:
            # Only float the wall under the cursor while it has no legal snap.
            self._draw_wall_rect(self._floating_wall_rect(), alpha=200)
        if self.pawn_drag:
            self._draw_pawn(self.mouse, st.current_player, lifted=True)
        if st.winner is not None:
            self._draw_game_over(st)

    # ------------------------------------------------------------------ #
    def _draw_board(self, st: BoardState) -> None:
        L, s = self.layout, self.screen
        pygame.draw.rect(s, T.FRAME, L.frame_rect, border_radius=14)
        pygame.draw.rect(s, T.FRAME_EDGE, L.frame_rect, width=3, border_radius=14)
        pygame.draw.rect(s, T.GROOVE, (L.ox - 4, L.oy - 4, L.board_px + 8, L.board_px + 8), border_radius=6)

        hover = None
        if self.human_turn and not self.wall_drag:
            hover = L.square_at(*self.mouse, forgiving=self.pawn_drag)
        goal_owner = {st.goal_rows[0]: 0, st.goal_rows[1]: 1}
        for r in range(st.size):
            for c in range(st.size):
                color = T.SQUARE
                if r in goal_owner:
                    color = _blend(T.SQUARE, T.PAWN_COLORS[goal_owner[r]], 0.30)
                if hover == (r, c) and st.is_valid_pawn_move((r, c)):
                    color = _blend(color, (255, 255, 255), 0.25)
                pygame.draw.rect(s, color, L.square_rect(r, c), border_radius=6)

        # Coordinates (same numbering as the CLI and the engine).
        for i in range(st.size):
            x, y = L.square_center(0, i)
            self._text(str(i), (x, L.margin / 2), self.font_small, T.TEXT_DIM)
            x, y = L.square_center(i, 0)
            self._text(str(i), (L.row_label_x, y), self.font_small, T.TEXT_DIM)

    def _draw_walls(self, st: BoardState) -> None:
        animating = {a.data for a in self.anims if a.kind == "wall"}
        for wall in st.walls:
            if wall not in animating:
                self._draw_wall_rect(self.layout.wall_rect(wall))

    def _draw_wall_rect(self, rect, color=T.WALL, alpha: int = 255) -> None:
        x, y, w, h = rect
        surf = pygame.Surface((max(int(w), 1) + 4, max(int(h), 1) + 4), pygame.SRCALPHA)
        dark = T.WALL_DARK if color == T.WALL else _blend(color, (0, 0, 0), 0.35)
        pygame.draw.rect(surf, (0, 0, 0, alpha // 3), (3, 3, int(w), int(h)), border_radius=4)
        pygame.draw.rect(surf, (*color, alpha), (0, 0, int(w), int(h)), border_radius=4)
        pygame.draw.rect(surf, (*dark, alpha), (0, 0, int(w), int(h)), width=2, border_radius=4)
        self.screen.blit(surf, (x, y))

    def _draw_move_dots(self, st: BoardState) -> None:
        surf = pygame.Surface((self.layout.width, self.layout.height), pygame.SRCALPHA)
        for r, c in st.valid_pawn_moves:
            pygame.draw.circle(surf, (*T.DOT, 200), self.layout.square_center(r, c), self.layout.cell * 0.14)
        self.screen.blit(surf, (0, 0))

    def _draw_pawns(self, st: BoardState) -> None:
        moving = {a.data: a.value for a in self.anims if a.kind == "pawn"}
        for p in (0, 1):
            if self.pawn_drag and p == st.current_player:
                continue
            pos = moving.get(p, self.layout.square_center(*st.pawns[p]))
            glow = self.human_turn and p == st.current_player and p not in moving
            self._draw_pawn(pos, p, glow=glow)

    def _draw_pawn(self, pos, player: int, glow: bool = False, lifted: bool = False) -> None:
        s, L = self.screen, self.layout
        x, y = pos
        radius = L.cell * (0.40 if lifted else 0.36)
        if glow:
            pulse = (math.sin(pygame.time.get_ticks() / 260) + 1) / 2
            ring = pygame.Surface((L.cell * 2, L.cell * 2), pygame.SRCALPHA)
            pygame.draw.circle(ring, (*T.PAWN_COLORS[player], int(60 + 70 * pulse)),
                               (L.cell, L.cell), radius + 4 + 3 * pulse, width=4)
            s.blit(ring, (x - L.cell, y - L.cell))
        shadow = pygame.Surface((L.cell * 2, L.cell * 2), pygame.SRCALPHA)
        offset = 7 if lifted else 3
        pygame.draw.circle(shadow, (0, 0, 0, 90), (L.cell, L.cell + offset), radius)
        s.blit(shadow, (x - L.cell, y - L.cell))
        pygame.draw.circle(s, T.PAWN_COLORS[player], (x, y), radius)
        pygame.draw.circle(s, T.PAWN_DARK[player], (x, y), radius, width=3)
        pygame.draw.circle(s, _blend(T.PAWN_COLORS[player], (255, 255, 255), 0.55),
                           (x - radius * 0.35, y - radius * 0.35), radius * 0.22)

    # ------------------------------------------------------------------ #
    def _draw_panel(self, st: BoardState) -> None:
        L, s = self.layout, self.screen
        for p in (0, 1):
            rect = L.card_rects[p]
            active = st.winner is None and st.current_player == p
            pygame.draw.rect(s, T.PANEL, rect, border_radius=12)
            if active:
                pygame.draw.rect(s, T.PANEL_ACTIVE_EDGE, rect, width=3, border_radius=12)
            x, y, w, h = rect
            pygame.draw.circle(s, T.PAWN_COLORS[p], (x + 26, y + 28), 12)
            self._text(self.names[p], (x + 46, y + 28), self.font, T.TEXT, anchor="midleft")
            goal = "top row" if st.goal_rows[p] == 0 else "bottom row"
            self._text(f"Goal: {goal} (row {st.goal_rows[p]})", (x + 20, y + 58),
                       self.font_small, T.TEXT_DIM, anchor="midleft")
            self._text(f"Walls left: {st.walls_left[p]}", (x + 20, y + 80),
                       self.font_small, T.TEXT, anchor="midleft")
            shown = st.walls_left[p]
            if p == st.current_player and (self.wall_drag or self._animating("wall_return", p)):
                shown -= 1
            for i in range(shown):
                self._draw_wall_rect(L.stub_rect(p, i))

        # Turn indicator, hints and messages between the cards.
        mid = L.height / 2
        cx = L.panel_x + L.panel_width / 2
        if st.winner is None:
            cur = st.current_player
            if self.paused:
                self._text("Paused", (cx, mid - 82), self.font, T.TEXT)
            elif self.bot_thinking:
                dots = "." * (1 + pygame.time.get_ticks() // 400 % 3)
                self._text(f"{self.names[cur]} is thinking{dots}", (cx, mid - 82), self.font, T.PAWN_COLORS[cur])
            else:
                self._text(f"{self.names[cur]}'s turn", (cx, mid - 82), self.font, T.PAWN_COLORS[cur])
            if self.bots_only:
                delay = f"{self.move_delay:g} s between moves" if self.move_delay else "no pause between moves"
                hints = ("Bot vs Bot", delay, "Space = pause/resume")
            else:
                hints = ("Click or drag your pawn to move",
                         "Drag a wall from your reserve",
                         "Right-click / Esc cancels a drag")
            for i, line in enumerate(hints):
                self._text(line, (cx, mid - 52 + 20 * i), self.font_small, T.TEXT_DIM)
        info_y = L.button_rect[1] + L.button_rect[3] + 14
        if self.bot_info and "nodes" in self.bot_info:  # alpha-beta search stats
            info = self.bot_info
            ev = Eval.from_info(info, 0)
            outcome = f"eval {ev.text()}" if ev is not None else f"forced {info.get('forced')}"
            self._text(f"Bot: depth {info['depth_reached']}, {_count(info['nodes'])} pos, "
                       f"{info['seconds']:.2f}s, {outcome}", (cx, info_y), self.font_small, T.TEXT_DIM)
        elif self.evaluator is not None:
            bar = self.eval_bars[0]
            if bar is not None and bar[0].mate != 0:
                ev = bar[0]
                detail = f"depth {ev.depth}, {_count(ev.nodes)} pos, {ev.seconds:.2f}s"
            else:
                detail = "thinking..." if st.winner is None else "game over"
            self._text(f"Eval ({self.evaluator.short_label}): {detail}", (cx, info_y), self.font_small, T.TEXT_DIM)
        if self.message and pygame.time.get_ticks() < self.message_until:
            self._text(self.message, (cx, mid + 14), self.font_small, T.PREVIEW_BAD, max_width=L.panel_width)

        bx, by, bw, bh = L.button_rect
        hovered = Layout.contains(L.button_rect, *self.mouse)
        pygame.draw.rect(s, _blend(T.PANEL, (255, 255, 255), 0.12 if hovered else 0.05), L.button_rect, border_radius=10)
        self._text("New game (R)", (bx + bw / 2, by + bh / 2), self.font_small, T.TEXT)

    def _draw_eval_bars(self) -> None:
        """Chess-style eval bars in Red (from the bottom) and Blue (from the top).

        A bot's bar shows its own search score, written in its colour: positive is
        good for that bot. The Human vs Human analysis bar writes the score in the
        colour of the side it favours, at that side's end of the bar.
        """
        L, s = self.layout, self.screen
        for slot in (0, 1):
            if not self.eval_slots[slot]:
                continue
            x, y, w, h = L.eval_bar_rects[slot]
            bar = self.eval_bars[slot]
            pygame.draw.rect(s, T.PANEL, (x - 3, y - 3, w + 6, h + 6), border_radius=6)
            if bar is None:
                share, label, color, pos = 0.5, "...", T.TEXT_DIM, L.eval_text_pos[slot]
                dim = 0.55
            else:
                ev, owner = bar
                share, dim = ev.red_share(), 0.0
                if owner is not None:
                    label, color, pos = ev.for_player(owner).text(), T.PAWN_COLORS[owner], L.eval_text_pos[slot]
                else:
                    lead = ev.leader()
                    label = ev.for_player(0 if lead is None else lead).text()
                    color = T.TEXT if lead is None else T.PAWN_COLORS[lead]
                    # Analysis readout at the favoured side's end: Red at the bottom, Blue at the top.
                    tx = x + w / 2
                    pos = (tx, L.height - L.margin / 2) if lead in (None, 0) else (tx, L.margin / 2)
            red_h = round(h * share)
            blue = _blend(T.PAWN_COLORS[1], T.PANEL, dim)
            red = _blend(T.PAWN_COLORS[0], T.PANEL, dim)
            pygame.draw.rect(s, blue, (x, y, w, h - red_h), border_top_left_radius=4, border_top_right_radius=4)
            pygame.draw.rect(s, red, (x, y + h - red_h, w, red_h), border_bottom_left_radius=4,
                             border_bottom_right_radius=4)
            pygame.draw.line(s, T.TEXT_DIM, (x - 3, y + h / 2), (x + w + 2, y + h / 2), 1)  # equal
            self._text(label, pos, self.font_small, color)

    def _draw_game_over(self, st: BoardState) -> None:
        L, s = self.layout, self.screen
        overlay = pygame.Surface((L.width, L.height), pygame.SRCALPHA)
        overlay.fill((10, 12, 20, 150))
        s.blit(overlay, (0, 0))
        winner = st.winner
        assert winner is not None
        box = (L.width / 2 - 260, L.height / 2 - 70, 520, 130)
        pygame.draw.rect(s, T.PANEL, box, border_radius=16)
        pygame.draw.rect(s, T.PAWN_COLORS[winner], box, width=3, border_radius=16)
        self._text(f"{self.names[winner]} wins!", (L.width / 2, L.height / 2 - 20),
                   self.font_big, T.PAWN_COLORS[winner])
        self._text("Click or press R for a new game  -  Esc to quit",
                   (L.width / 2, L.height / 2 + 24), self.font_small, T.TEXT)

    def _text(self, text: str, pos, font, color, anchor: str = "center", max_width: Optional[int] = None) -> None:
        if max_width:
            words, lines, line = text.split(), [], ""
            for word in words:
                trial = f"{line} {word}".strip()
                if font.size(trial)[0] > max_width - 10 and line:
                    lines.append(line)
                    line = word
                else:
                    line = trial
            lines.append(line)
            x, y = pos
            for i, ln in enumerate(lines):
                self._text(ln, (x, y + i * font.get_linesize()), font, color, anchor)
            return
        surf = font.render(text, True, color)
        rect = surf.get_rect(**{anchor: (int(pos[0]), int(pos[1]))})
        self.screen.blit(surf, rect)
