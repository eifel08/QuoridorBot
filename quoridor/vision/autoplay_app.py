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

from ..moves import Orientation, WallMove
from .analysis import DEFAULT_SPEC
from .app import BG, BLUE_HEX, CARD, DIM, GOOD, RED_HEX, TEXT, WARN, AssistantApp
from .autoplay import WALL_CLICKS, AutoPlayer, ClickProfile, EmergencyStop, PyAutoGuiMouse, StopSwitch, wall_points
from .parser import PLAYER_NAMES, Profile

STATE_TEXT = {"off": "OFF", "waiting-start": "WAITING FOR A GAME", "opponent": "WAITING FOR OPPONENT",
              "thinking": "THINKING", "moving": "MOVING", "confirming": "MOVING", "over": "GAME OVER"}


class AutoPlayApp(AssistantApp):
    def __init__(self, spec=DEFAULT_SPEC, monitor: int = 1, fps: float = 5.0, profile: Optional[Profile] = None,
                 debug: bool = False, click_profile: Optional[ClickProfile] = None, join_in_progress: bool = False,
                 mouse=None, grabber=None) -> None:
        self.stop = StopSwitch()
        self.click_profile = click_profile or ClickProfile()
        self.player = AutoPlayer(mouse or PyAutoGuiMouse(), self.click_profile, self.stop, join_in_progress)
        self._hover_test = 0
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
        self.join_var = tk.BooleanVar(value=self.player.join_in_progress)
        tk.Checkbutton(box, text="Join a game in progress", variable=self.join_var, bg=CARD, fg=TEXT, selectcolor=BG,
                       activebackground=CARD, activeforeground=TEXT, anchor="w").pack(fill="x", padx=6, pady=(2, 6))

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

    # -- periodic update ---------------------------------------------------------------
    def _refresh(self) -> None:
        super()._refresh()
        self.watcher.keep_images = True
        player = self.player
        self.click_profile.wall_click = self.wall_var.get() if not player.state == "moving" else self.click_profile.wall_click
        player.join_in_progress = self.join_var.get()
        with self._lock:
            analysis = self._analysis
        was_testing = "Cursor is where" in self.auto_message.cget("text") and not player.armed
        player.tick(self.watcher.latest, self.tracker.current, analysis)
        if self.wall_var.get() != self.click_profile.wall_click:
            self.wall_var.set(self.click_profile.wall_click)   # learned from the site's preview
        # the exact move count is the best turn indicator there is: let the panel and the analysis use it
        self.tracker.counted_turn = None if player.ply is None else player.ply % 2

        colour = {"off": DIM, "waiting-start": WARN, "opponent": TEXT, "thinking": GOOD, "moving": GOOD,
                  "confirming": GOOD, "over": WARN}[player.state]
        if not player.armed and player.message.startswith(("STOPPED", "Stopped")):
            colour = RED_HEX
        self.auto_state.config(text=STATE_TEXT[player.state] if player.armed else "OFF", fg=colour)
        if not was_testing:
            self.auto_message.config(text=player.message)
        if player.identity is not None:
            me = player.me
            self.auto_identity.config(text=f"Playing as: {PLAYER_NAMES[me]} (bottom)", fg=(RED_HEX, BLUE_HEX)[me])
            count = "move count unknown" if player.ply is None else f"move {player.ply}"
            self.auto_signals.config(text=f"Seen by: {', '.join(player.identity.evidence)}\n"
                                          f"{count} · turn signals: {player.signals.text()}"
                                          + (f"\nLast played: {player.last_move}" if player.last_move else ""))
        else:
            self.auto_identity.config(text="Playing as: - (the bottom player, locked at the start position)", fg=TEXT)
            self.auto_signals.config(text="")
        if player.armed:
            self.auto_button.config(text="STOP (Esc)", bg="#b3261e", activebackground="#d0342c")
        else:
            self.auto_button.config(text="Start auto-play", bg="#2e7d46", activebackground="#3c9a59")

    def close(self) -> None:
        self.stop.trigger("window closed")
        super().close()
