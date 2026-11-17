"""Auto-Play window: the Screen Assistant's panel plus the controls of the auto-player.

A mode of its own (``python main.py --autoplay``), separate from the passive
Screen Assistant: it starts switched off, and clicks only after *Start
auto-play* has been pressed. Esc (anywhere), the red STOP button, or the
mouse in a screen corner stop it at once.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional

from ..bots.cpp_engine import WALL_MODES, engine_available
from ..bots.factory import BotSpec, parse_bot_spec
from ..bots.presets import CUSTOM_LABEL, label_for, preset_choices
from ..moves import Orientation, WallMove
from .analysis import DEFAULT_SPEC
from .app import (BG, BLUE_HEX, CARD, DIM, GOOD, MARK, RED_HEX, TEXT, WALL_LABELS, WARN, AssistantApp,
                  enable_dpi_awareness)
from .autoplay import WALL_CLICKS, AutoPlayer, ClickProfile, EmergencyStop, PyAutoGuiMouse, StopSwitch, wall_points
from .parser import Profile

STATE_TEXT = {"off": "OFF", "waiting-start": "WAITING FOR A GAME", "opponent": "WAITING FOR OPPONENT",
              "thinking": "THINKING", "highlighting": "ABOUT TO PLAY", "moving": "CLICKING",
              "confirming": "CLICKED", "over": "GAME OVER"}
#: While the bot is about to click or clicking, a new bot only takes over after that move.
BUSY_STATES = ("highlighting", "moving", "confirming")


class AutoPlayApp(AssistantApp):
    AUTOPLAY = True

    def __init__(self, spec=DEFAULT_SPEC, monitor: int = 1, fps: float = 5.0, profile: Optional[Profile] = None,
                 debug: bool = False, click_profile: Optional[ClickProfile] = None, join_in_progress: bool = False,
                 mouse=None, grabber=None) -> None:
        # Before pyautogui is loaded: it would make the process only "system" DPI aware on
        # Windows, and clicks would then miss on scaled or mixed-DPI monitors.
        enable_dpi_awareness()
        self.stop = StopSwitch()
        self.click_profile = click_profile or ClickProfile()
        self.player = AutoPlayer(mouse or PyAutoGuiMouse(), self.click_profile, self.stop, join_in_progress)
        self.player.before_click = self._before_click
        self._hover_test = 0
        self._pending_spec: Optional[BotSpec] = None
        super().__init__(spec, monitor, fps, profile, debug, overlay=False, grabber=grabber)
        self.root.title("Quoridor Auto-Play")
        self.bar.config(height=120)              # the panel is taller than the assistant's: save some room
        self.watcher.keep_images = True          # the clock watcher compares the cards' pixels
        self.esc_how = self.stop.listen_for_escape()
        self.root.bind_all("<Escape>", lambda _e: self.stop.trigger("Esc pressed"))
        if not self.esc_how:
            self.stop_hint.config(text="Emergency stop: mouse into a screen corner, or Esc while this window is "
                                       "active (install pynput for Esc everywhere).")

    # -- widgets -----------------------------------------------------------------
    def _build_header(self) -> None:
        box = tk.Frame(self.root, bg=CARD)
        box.pack(fill="x", padx=12, pady=(10, 0))
        self._label(box, "AUTO-PLAY", DIM, 9, True).pack(fill="x", padx=10, pady=(8, 0))
        self.auto_state = self._label(box, "OFF", DIM, 15, True)
        self.auto_state.pack(fill="x", padx=10)
        self.auto_message = self._label(box, "Auto-play is off", TEXT, 9, wraplength=280)
        self.auto_message.pack(fill="x", padx=10)
        self.auto_identity = self._label(box, "Playing as: -", TEXT, 10, True, wraplength=280)
        self.auto_identity.pack(fill="x", padx=10, pady=(6, 0))
        self.auto_signals = self._label(box, "", DIM, 9, wraplength=280)
        self.auto_signals.pack(fill="x", padx=10)

        # Which bot plays: presets, trained ML checkpoints found on disk, or any bot spec.
        self.bot_choices = preset_choices(self.spec)
        bot_row = tk.Frame(box, bg=CARD)
        bot_row.pack(fill="x", padx=10, pady=(8, 0))
        self._label(bot_row, "Bot", DIM, 9).pack(side="left")
        self.bot_var = tk.StringVar(value=label_for(self.spec, self.bot_choices))
        self.bot_box = ttk.Combobox(bot_row, width=36, textvariable=self.bot_var, state="readonly",
                                    values=[label for label, _ in self.bot_choices] + [CUSTOM_LABEL])
        self.bot_box.pack(side="right", fill="x", expand=True, padx=(8, 0))
        self.bot_box.bind("<<ComboboxSelected>>", self._on_bot_selected)
        self.custom_row = tk.Frame(box, bg=CARD)
        self.custom_var = tk.StringVar(value=self.spec.to_string())
        entry = tk.Entry(self.custom_row, textvariable=self.custom_var, bg=BG, fg=TEXT, insertbackground=TEXT,
                         relief="flat", font=("Consolas", 9))
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda _e: self._apply_custom())
        tk.Button(self.custom_row, text="Use", command=self._apply_custom, bg=BG, fg=TEXT, activebackground=CARD,
                  activeforeground=TEXT, relief="flat").pack(side="right", padx=(6, 0))
        self._bot_row = bot_row
        self.bot_info = self._label(box, "", DIM, 9, wraplength=280)
        self.bot_info.pack(fill="x", padx=10)
        self.auto_button = tk.Button(box, text="Start auto-play", command=self._toggle_auto, bg="#2e7d46", fg="white",
                                     activebackground="#3c9a59", activeforeground="white", relief="flat",
                                     font=("Segoe UI", 11, "bold"))
        self.auto_button.pack(fill="x", padx=10, pady=(8, 4))
        self.stop_hint = self._label(box, "Emergency stop: Esc, or move the mouse into a screen corner.", WARN, 9,
                                     wraplength=280)
        self.stop_hint.pack(fill="x", padx=10)

        row = tk.Frame(box, bg=CARD)
        row.pack(fill="x", padx=10, pady=(6, 0))
        self._label(row, "Wall click point", DIM, 9).pack(side="left")
        self.wall_var = tk.StringVar(value=self.click_profile.wall_click)
        ttk.Combobox(row, width=8, textvariable=self.wall_var, state="readonly", values=WALL_CLICKS).pack(side="right")
        tk.Button(box, text="Test wall hover (no click)", command=self._test_hover, bg=BG, fg=TEXT,
                  activebackground=CARD, activeforeground=TEXT, relief="flat").pack(fill="x", padx=10, pady=(6, 0))
        self.auto_log = tk.Label(box, text="", fg=DIM, bg=CARD, font=("Consolas", 8), anchor="w", justify="left")
        self.auto_log.pack(fill="x", padx=10, pady=(6, 0))
        self.join_var = tk.BooleanVar(value=self.player.join_in_progress)
        tk.Checkbutton(box, text="Join a game in progress", variable=self.join_var, bg=CARD, fg=TEXT, selectcolor=BG,
                       activebackground=CARD, activeforeground=TEXT, anchor="w").pack(fill="x", padx=6, pady=(2, 6))

    # -- bot selection -----------------------------------------------------------------
    def _on_bot_selected(self, _event=None) -> None:
        label = self.bot_var.get()
        if label == CUSTOM_LABEL:
            self.custom_var.set(self.spec.to_string())
            self.custom_row.pack(fill="x", padx=10, pady=(4, 0), after=self._bot_row)
            self.bot_info.config(text="Type a bot spec, e.g. ab:depth=30,time=2,walls=attack or ml:model=best.pt, "
                                      "then press Enter.", fg=DIM)
            return
        self.custom_row.pack_forget()
        spec = next((s for lbl, s in self.bot_choices if lbl == label), None)
        if spec is not None:
            self.select_bot(spec)

    def _apply_custom(self) -> None:
        try:
            spec = parse_bot_spec(self.custom_var.get())
        except ValueError as exc:
            self.bot_info.config(text=str(exc), fg=RED_HEX)
            return
        self.select_bot(spec)

    def select_bot(self, spec: BotSpec) -> bool:
        """Make ``spec`` the bot that plays. Returns False (with the reason shown) if it cannot be used."""
        if spec.kind == "ab" and spec.engine == "cpp" and not engine_available():
            self.bot_info.config(text="The C++ engine is not built (python build_engine.py).", fg=RED_HEX)
            return False
        if spec.kind == "ml":
            try:
                import torch  # noqa: F401
            except Exception:
                self.bot_info.config(text="ML bots need PyTorch (pip install -r requirements-ml.txt).", fg=RED_HEX)
                return False
        if self.player.state in BUSY_STATES:
            self._pending_spec = spec  # never swap the engine under a move that is being clicked
            self.bot_info.config(text=f"Switching to {spec.label()} after this move...", fg=WARN)
            return True
        self._set_spec(spec)
        return True

    def _set_spec(self, spec: BotSpec) -> None:
        self._pending_spec = None
        self.spec = spec
        if spec.kind == "ab":
            # Keep the folded "Bot settings" in line: they are re-read on every tick.
            self.depth_var.set(str(spec.depth))
            self.time_var.set(f"{spec.time_limit:g}" if spec.time_limit else "no limit")
            wall_name = next((n for n, v, _ in WALL_MODES if v == spec.wall_filter), "both")
            self.walls_var.set(WALL_LABELS[wall_name])
            self.engine_var.set({"auto": "Auto", "cpp": "C++", "py": "Python"}[spec.engine])
        current = self.tracker.current
        with self._lock:
            self._analysis = None
            self._searching = current if (current and current.turn is not None and current.winner is None) else None
        self.analyzer.set_spec(spec, current)
        self.player.log(f"Bot: {spec.label()} ({spec.to_string()})")

    def _toggle_auto(self) -> None:
        if self.player.armed:
            self.stop.trigger("STOP button")
        else:
            self.player.arm()

    def _test_hover(self) -> None:
        """Move the cursor to where a wall would be clicked, without clicking, to check the site's preview."""
        frame = self.watcher.latest
        g = frame.result.geometry if frame else None
        if self.player.armed or g is None or frame is None:
            return
        self._hover_test += 1
        wall = WallMove(3, 4, (Orientation.HORIZONTAL, Orientation.VERTICAL)[self._hover_test % 2])
        point = wall_points(g, wall)[self.wall_var.get()]
        self.stop.clear()
        try:
            self.player.hand.glide(*frame.to_screen(*point))
        except EmergencyStop:
            pass
        kind = "horizontal wall e5h (above e5 and f5)" if wall.orientation is Orientation.HORIZONTAL \
            else "vertical wall e5v (right of e5 and e6)"
        self.auto_message.config(text=f"Cursor is where a {kind} would be clicked. If the site previews another "
                                      "wall, choose a different wall click point.")

    # -- the move marker -------------------------------------------------------------------
    def _show_overlay(self, frame, analysis) -> None:
        """Auto-Play marks only the move it is about to click, during the highlight before the click."""
        player = self.player
        res = frame.result if frame else None
        if (player.state != "highlighting" or player.planned is None or frame is None or res is None
                or res.geometry is None or not res.ok or player._before is None):
            self.overlay.hide()
            return
        from .position import move_notation

        self._mark_move(frame, player._before, player.planned, move_notation(player.planned))

    def _before_click(self) -> None:
        """Right before the mouse moves: take the marker off the screen so the click reaches the game."""
        self.overlay.hide(force=True)
        self.root.update_idletasks()

    # -- periodic update ---------------------------------------------------------------
    def _refresh(self) -> None:
        super()._refresh()
        self.watcher.keep_images = True
        player = self.player
        self.click_profile.wall_click = self.wall_var.get() if not player.state == "moving" else self.click_profile.wall_click
        player.join_in_progress = self.join_var.get()
        if self._pending_spec is not None and player.state not in BUSY_STATES:
            self._set_spec(self._pending_spec)
        if self._pending_spec is None:
            label = label_for(self.spec, self.bot_choices)
            if self.bot_var.get() != CUSTOM_LABEL and self.bot_var.get() != label:
                self.bot_var.set(label)  # changed from the folded settings
            self.bot_info.config(text=f"Playing with: {self.spec.to_string()}", fg=DIM)
        with self._lock:
            analysis = self._analysis
        was_testing = "Cursor is where" in self.auto_message.cget("text") and not player.armed
        player.tick(self.watcher.latest, self.tracker.current, analysis)
        self._show_overlay(self.watcher.latest, None)   # marker on / off right away, not on the next refresh
        if self.wall_var.get() != self.click_profile.wall_click:
            self.wall_var.set(self.click_profile.wall_click)   # learned from the site's preview
        # the exact move count is the best turn indicator there is: let the panel and the analysis use it
        self.tracker.counted_turn = None if player.ply is None else player.ply % 2

        colour = {"off": DIM, "waiting-start": WARN, "opponent": TEXT, "thinking": GOOD, "highlighting": MARK,
                  "moving": GOOD, "confirming": GOOD, "over": WARN}[player.state]
        if not player.armed and player.message.startswith(("STOPPED", "Stopped")):
            colour = RED_HEX
        self.auto_state.config(text=STATE_TEXT[player.state] if player.armed else "OFF", fg=colour)
        if not was_testing:
            self.auto_message.config(text=player.message)
        if player.identity is not None:
            me = player.me
            self.auto_identity.config(text=f"Playing as: {player.identity.text()}", fg=(RED_HEX, BLUE_HEX)[me])
            count = "move count unknown" if player.ply is None else f"move {player.ply}"
            self.auto_signals.config(text=f"Seen by: {', '.join(player.identity.evidence)}\n"
                                          f"{count} · turn signals: {player.signals.text()}"
                                          + (f"\nLast played: {player.last_move}" if player.last_move else ""))
        else:
            self.auto_identity.config(text="Playing as: - (the bottom player, locked at the start position)", fg=TEXT)
            self.auto_signals.config(text="")
        # the last steps, one line each (the console has them in full)
        lines = [line[9:] if len(line) > 9 else line for line in list(player.log_lines)[-4:]]
        self.auto_log.config(text="\n".join(line if len(line) <= 42 else line[:41] + "…" for line in lines))
        if player.armed:
            self.auto_button.config(text="STOP (Esc)", bg="#b3261e", activebackground="#d0342c")
        else:
            self.auto_button.config(text="Start auto-play", bg="#2e7d46", activebackground="#3c9a59")

    def close(self) -> None:
        self.stop.trigger("window closed")
        super().close()
