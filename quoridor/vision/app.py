"""Screen Assistant: analysis panel, on-board move marker and debug preview.

Three always-on-top windows (Tk, from the Python standard library):

* the **panel** with the detected state, the eval bar, the recommended move,
  alternatives and the bot settings;
* the **overlay** that marks the recommended move on the board itself. On
  Windows it is one transparent, click-through window over the monitor. On
  other systems, where Tk cannot make a window transparent to clicks, the
  marker is an outline built from four thin strip windows, so the squares
  stay clickable;
* the optional **debug preview** showing what the parser detected.

On Windows the three windows are excluded from screen capture, so the
assistant never reads its own drawings. Elsewhere the panel and the preview
are blanked out of every captured frame; keep them off the game.
"""

from __future__ import annotations

import base64
import sys
import threading
import tkinter as tk
from dataclasses import replace
from tkinter import ttk
from typing import Optional

import cv2

from ..bots.cpp_engine import WALL_MODES, engine_available, wall_filter_from_name
from ..bots.factory import BotSpec
from ..moves import PawnMove
from .analysis import DEFAULT_SPEC, Analysis, Analyzer
from .capture import Frame, ScreenWatcher
from .parser import BLUE, PLAYER_NAMES, RED, Profile, ScreenParser, annotate
from .position import Position, PositionTracker

BG, CARD, TEXT, DIM = "#181b26", "#222636", "#e8eaf0", "#969cac"
RED_HEX, BLUE_HEX = "#ff525c", "#3aa0ff"
GOOD, WARN = "#78e68c", "#ffc440"
MARK = "#ff3df2"           # move marker: magenta, a hue the parser reads neither as a pawn nor as the turn border
KEY = "#010203"            # transparent colour of the Windows overlay
TIME_CHOICES = ("0.25", "0.5", "1", "2", "3", "5", "10")
WALL_LABELS = {name: label for name, _, label in WALL_MODES}


# ---------------------------------------------------------------------- #
# Windows helpers (no-ops elsewhere)
# ---------------------------------------------------------------------- #
def enable_dpi_awareness() -> None:
    """Make Tk coordinates equal to captured pixels on scaled Windows displays."""
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _hwnd(window: tk.Misc) -> int:
    import ctypes

    window.update_idletasks()
    return ctypes.windll.user32.GetParent(window.winfo_id()) or window.winfo_id()


def exclude_from_capture(window: tk.Misc) -> bool:
    """Hide a window from screenshots (Windows 10 2004+). True if it worked."""
    if sys.platform != "win32":
        return False
    import ctypes

    try:
        return bool(ctypes.windll.user32.SetWindowDisplayAffinity(_hwnd(window), 0x11))
    except Exception:
        return False


def make_click_through(window: tk.Misc) -> bool:
    if sys.platform != "win32":
        return False
    import ctypes

    try:
        user32 = ctypes.windll.user32
        hwnd = _hwnd(window)
        style = user32.GetWindowLongW(hwnd, -20)
        # WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
        user32.SetWindowLongW(hwnd, -20, style | 0x80000 | 0x20 | 0x80 | 0x08000000)
        return True
    except Exception:
        return False


def _intersects(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# ---------------------------------------------------------------------- #
# Move marker on the board
# ---------------------------------------------------------------------- #
class Overlay:
    """Marks one rectangle of the screen (target square or wall slot)."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.transparent = sys.platform == "win32"
        self.excluded = False
        self._win: Optional[tk.Toplevel] = None
        self._canvas: Optional[tk.Canvas] = None
        self._strips: list[tk.Toplevel] = []
        self._shown: Optional[tuple] = None

    def _strip(self) -> tk.Toplevel:
        win = tk.Toplevel(self.root, bg=MARK)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        return win

    def show(self, area: tuple[int, int, int, int], rect: tuple[int, int, int, int], label: str,
             origin: Optional[tuple[int, int]] = None) -> None:
        """``area``: the captured monitor, ``rect``: what to mark, ``origin``: where a pawn comes from (all in
        screen coordinates)."""
        key = (area, rect, label, origin)
        if key == self._shown:
            return
        self._shown = key
        x1, y1, x2, y2 = rect
        if self.transparent:
            ax1, ay1, ax2, ay2 = area
            if self._win is None:
                self._win = tk.Toplevel(self.root, bg=KEY)
                self._win.overrideredirect(True)
                self._win.attributes("-topmost", True)
                self._win.attributes("-transparentcolor", KEY)
                self._canvas = tk.Canvas(self._win, bg=KEY, highlightthickness=0)
                self._canvas.pack(fill="both", expand=True)
                make_click_through(self._win)
                self.excluded = exclude_from_capture(self._win)
            assert self._canvas is not None
            self._win.geometry(f"{ax2 - ax1}x{ay2 - ay1}+{ax1}+{ay1}")
            self._win.deiconify()
            make_click_through(self._win)  # again after deiconify: clicks must go through to the game
            c = self._canvas
            c.delete("all")
            if self.excluded:  # invisible to our own screenshots: draw freely
                if origin is not None:
                    c.create_line(origin[0] - ax1, origin[1] - ay1, (x1 + x2) / 2 - ax1, (y1 + y2) / 2 - ay1,
                                  fill=MARK, width=5, arrow="last", arrowshape=(18, 22, 8))
                c.create_rectangle(x1 - ax1, y1 - ay1, x2 - ax1, y2 - ay1, outline=MARK, width=5)
                c.create_text(x1 - ax1 + 4, y1 - ay1 - 6, text=label, fill=MARK, anchor="sw",
                              font=("Segoe UI", 14, "bold"))
            else:  # older Windows: we capture this too, so only an outline that stays out of the grooves
                c.create_rectangle(x1 - ax1 + 2, y1 - ay1 + 2, x2 - ax1 - 2, y2 - ay1 - 2, outline=MARK, width=3)
            return
        t = 3
        while len(self._strips) < 4:
            self._strips.append(self._strip())
        boxes = ((x1, y1, x2 - x1, t), (x1, y2 - t, x2 - x1, t), (x1, y1, t, y2 - y1), (x2 - t, y1, t, y2 - y1))
        for win, (x, y, w, h) in zip(self._strips, boxes):
            win.geometry(f"{max(1, w)}x{max(1, h)}+{x}+{y}")
            win.deiconify()
            win.lift()

    def hide(self, force: bool = False) -> None:
        if self._shown is None and not force:
            return
        self._shown = None
        if self._win is not None:
            self._win.withdraw()
        for win in self._strips:
            win.withdraw()

    def windows(self) -> list[tk.Toplevel]:
        return ([self._win] if self._win is not None else []) + self._strips


# ---------------------------------------------------------------------- #
# Application
# ---------------------------------------------------------------------- #
class AssistantApp:
    #: Auto-Play reuses this window. It leaves out what does not apply there: the alternatives,
    #: "Show move for", "Mark the move on the board" (it marks the move it is about to click), and
    #: "Side to move" (its exact move count decides the turn; a manual side would make the engine
    #: analyse the opponent's move and the bot would never get a move of its own to play).
    AUTOPLAY = False

    def __init__(self, spec: BotSpec = DEFAULT_SPEC, monitor: int = 1, fps: float = 5.0,
                 profile: Optional[Profile] = None, debug: bool = False, overlay: bool = True,
                 grabber=None) -> None:
        enable_dpi_awareness()
        self.spec = spec
        self.root = tk.Tk()
        self.root.title("Quoridor Screen Assistant")
        self.root.configure(bg=BG)
        self.root.attributes("-topmost", True)
        self.root.geometry("+24+72")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self._lock = threading.Lock()
        self._analysis: Optional[Analysis] = None
        self._searching: Optional[Position] = None
        self.parser = ScreenParser(profile)
        self.tracker = PositionTracker()
        self.analyzer = Analyzer(spec, self._on_analysis)
        self.watcher = ScreenWatcher(self.parser, self.tracker, self._on_position, monitor, fps, grabber)
        self.overlay = Overlay(self.root)
        self._preview: Optional[tk.Toplevel] = None
        self._preview_label: Optional[tk.Label] = None
        self._preview_image = None
        self._preview_frame = -1
        self._placed_for = None
        self._closed = False

        self._build(debug, overlay)
        self.hidden = exclude_from_capture(self.root)
        self.watcher.start()
        self.root.after(100, self._tick)

    # -- callbacks from the worker threads ---------------------------------
    def _on_position(self, position: Position) -> None:
        with self._lock:
            self._searching = position if (position.turn is not None and position.winner is None) else None
            self._analysis = None
        self.analyzer.submit(position)

    def _on_analysis(self, analysis: Analysis) -> None:
        with self._lock:
            if analysis.position == self.tracker.current:
                self._analysis = analysis
                if analysis.complete:
                    self._searching = None

    # -- widgets ---------------------------------------------------------------
    def _label(self, parent, text="", fg=TEXT, size=10, bold=False, **kw) -> tk.Label:
        return tk.Label(parent, text=text, fg=fg, bg=parent["bg"], font=("Segoe UI", size, "bold" if bold else "normal"),
                        anchor="w", justify="left", **kw)

    def _build(self, debug: bool, overlay: bool) -> None:
        root = self.root
        pad = dict(padx=12, pady=(8, 0))
        self._build_header()
        self.status = self._label(root, "Looking for a Quoridor board...", WARN, 10, True, wraplength=300)
        self.status.pack(fill="x", **pad)

        state = tk.Frame(root, bg=CARD)
        state.pack(fill="x", **pad)
        self.turn_label = self._label(state, "Turn: -", size=11, bold=True)
        self.turn_label.pack(fill="x", padx=10, pady=(8, 0))
        self.blue_label = self._label(state, "Blue walls left: -", BLUE_HEX)
        self.blue_label.pack(fill="x", padx=10)
        self.red_label = self._label(state, "Red walls left: -", RED_HEX)
        self.red_label.pack(fill="x", padx=10)
        self.pos_label = self._label(state, "", DIM, 9, wraplength=280)
        self.pos_label.pack(fill="x", padx=10, pady=(0, 8))

        mid = tk.Frame(root, bg=BG)
        mid.pack(fill="x", **pad)
        self.bar = tk.Canvas(mid, width=30, height=190, bg=BG, highlightthickness=0)
        self.bar.pack(side="left")
        right = tk.Frame(mid, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))
        self._label(right, "Evaluation", DIM, 9).pack(fill="x")
        self.eval_label = self._label(right, "-", size=22, bold=True)
        self.eval_label.pack(fill="x")
        self.eval_sub = self._label(right, "", DIM, 9)
        self.eval_sub.pack(fill="x")
        self._label(right, "Best move", DIM, 9).pack(fill="x", pady=(8, 0))
        self.move_label = self._label(right, "-", MARK, 22, True)
        self.move_label.pack(fill="x")
        self.move_sub = self._label(right, "", DIM, 9)
        self.move_sub.pack(fill="x")

        self.alt_label = tk.Label(root, text="-", fg=TEXT, bg=BG, font=("Consolas", 10), anchor="w", justify="left")
        if not self.AUTOPLAY:
            self._label(root, "Alternatives", DIM, 9).pack(fill="x", **pad)
            self.alt_label.pack(fill="x", padx=12)
        self.info_label = self._label(root, "", DIM, 9, wraplength=300)
        self.info_label.pack(fill="x", **pad)
        self.warn_label = self._label(root, "", WARN, 9, wraplength=300)
        self.warn_label.pack(fill="x", padx=12)

        # bot settings
        box = tk.Frame(root, bg=CARD)
        if self.AUTOPLAY:  # folded away, so the panel fits on a 1080-pixel screen
            def toggle() -> None:
                if box.winfo_ismapped():
                    box.pack_forget()
                    fold.config(text="Bot settings ▸")
                else:
                    box.pack(fill="x", padx=12, pady=(0, 10))
                    fold.config(text="Bot settings ▾")

            fold = tk.Button(root, text="Bot settings ▸", command=toggle, bg=BG, fg=DIM, activebackground=CARD,
                             activeforeground=TEXT, relief="flat", anchor="w")
            fold.pack(fill="x", padx=12, pady=(8, 2))
        else:
            box.pack(fill="x", padx=12, pady=10)
        self.depth_var = tk.StringVar(value=str(self.spec.depth))
        self.time_var = tk.StringVar(value=f"{self.spec.time_limit:g}" if self.spec.time_limit else "no limit")
        wall_name = next((n for n, v, _ in WALL_MODES if v == self.spec.wall_filter), "both")
        self.walls_var = tk.StringVar(value=WALL_LABELS[wall_name])
        self.engine_var = tk.StringVar(value={"auto": "Auto", "cpp": "C++", "py": "Python"}[self.spec.engine])
        self.advise_var = tk.StringVar(value="Both sides")
        self.turn_var = tk.StringVar(value="Auto")
        self.side_var = tk.StringVar(value="Auto")
        self.overlay_var = tk.BooleanVar(value=overlay)
        self.debug_var = tk.BooleanVar(value=debug)
        rows = [("Depth", ttk.Spinbox(box, from_=1, to=30, width=8, textvariable=self.depth_var,
                                      command=self._apply_spec)),
                ("Think time (s)", ttk.Combobox(box, width=8, textvariable=self.time_var,
                                                values=(*TIME_CHOICES, "no limit"))),
                ("Walls searched", ttk.Combobox(box, width=16, textvariable=self.walls_var, state="readonly",
                                                values=[label for _, _, label in WALL_MODES])),
                ("Engine", ttk.Combobox(box, width=8, textvariable=self.engine_var, state="readonly",
                                        values=("Auto", "C++", "Python"))),
                ("Show move for", ttk.Combobox(box, width=10, textvariable=self.advise_var, state="readonly",
                                               values=("Both sides", "Red", "Blue"))),
                ("Side to move", ttk.Combobox(box, width=10, textvariable=self.turn_var, state="readonly",
                                              values=("Auto", "Red", "Blue"))),
                ("Red starts at", ttk.Combobox(box, width=10, textvariable=self.side_var, state="readonly",
                                               values=("Auto", "Bottom", "Top")))]
        if self.AUTOPLAY:
            rows = [row for row in rows if row[0] not in ("Show move for", "Side to move")]
        if self.spec.kind != "ab":
            rows = rows[4:]
            self._label(box, f"Bot: {self.spec.label()}", DIM, 9).grid(row=0, column=0, columnspan=2, sticky="w",
                                                                    padx=10, pady=(8, 0))
        for i, (text, widget) in enumerate(rows, start=1):
            self._label(box, text, DIM, 9).grid(row=i, column=0, sticky="w", padx=10, pady=3)
            widget.grid(row=i, column=1, sticky="e", padx=10, pady=3)
            widget.bind("<<ComboboxSelected>>", self._apply_spec)
            widget.bind("<Return>", self._apply_spec)
            widget.bind("<FocusOut>", self._apply_spec)
        box.columnconfigure(1, weight=1)
        n = len(rows) + 1
        checks = [("Mark the move on the board", self.overlay_var), ("Debug / calibration preview", self.debug_var)]
        for i, (text, var) in enumerate(checks[1:] if self.AUTOPLAY else checks):
            tk.Checkbutton(box, text=text, variable=var, bg=CARD, fg=TEXT, selectcolor=BG, activebackground=CARD,
                           activeforeground=TEXT, anchor="w", command=self._toggle_preview
                           ).grid(row=n + i, column=0, columnspan=2, sticky="w", padx=6)
        tk.Button(box, text="Re-scan screen", command=self._rescan, bg=BG, fg=TEXT, activebackground=CARD,
                  activeforeground=TEXT, relief="flat").grid(row=n + 2, column=0, columnspan=2, sticky="ew",
                                                             padx=10, pady=8)
        if debug:
            self._toggle_preview()

    def _build_header(self) -> None:
        """Extra widgets above the status line (used by Auto-Play)."""

    # -- settings -----------------------------------------------------------------
    def _apply_spec(self, _event=None) -> None:
        forced = {"Auto": None, "Red": RED, "Blue": BLUE}[self.turn_var.get()]
        if forced != self.tracker.forced_turn:
            self.tracker.forced_turn = forced
        self.tracker.forced_flipped = {"Auto": None, "Bottom": False, "Top": True}[self.side_var.get()]
        if self.spec.kind != "ab":
            return
        try:
            depth = max(1, min(30, int(self.depth_var.get())))
            text = self.time_var.get().strip().lower()
            limit = None if text in ("", "0", "no limit", "none") else max(0.05, float(text))
        except ValueError:
            return
        engine = {"Auto": "auto", "C++": "cpp", "Python": "py"}[self.engine_var.get()]
        if engine == "cpp" and not engine_available():
            engine = "auto"
            self.engine_var.set("Auto")
        wall_name = next(n for n, label in WALL_LABELS.items() if label == self.walls_var.get())
        spec = replace(self.spec, depth=depth, time_limit=limit, engine=engine)
        spec = spec.with_wall_filter(wall_filter_from_name(wall_name))
        if engine == "py":
            spec = replace(spec, options=())  # engine flags only exist in the C++ engine
        if spec != self.spec:
            self.spec = spec
            current = self.tracker.current
            with self._lock:
                self._analysis = None
                self._searching = current if (current and current.turn is not None
                                              and current.winner is None) else None
            self.analyzer.set_spec(spec, current)

    def _rescan(self) -> None:
        with self._lock:
            self._analysis, self._searching = None, None
        self._placed_for = None
        self.watcher.rescan()

    def _toggle_preview(self) -> None:
        on = self.debug_var.get()
        self.watcher.keep_images = on
        if on and self._preview is None:
            self._preview = tk.Toplevel(self.root, bg=BG)
            self._preview.title("Screen Assistant: what the parser sees")
            self._preview.attributes("-topmost", True)
            self._preview.geometry(f"+{self.root.winfo_rootx() + self.root.winfo_width() + 24}+72")
            self._preview.protocol("WM_DELETE_WINDOW", lambda: (self.debug_var.set(False), self._toggle_preview()))
            self._preview_label = tk.Label(self._preview, bg=BG, fg=DIM, text="waiting for a frame...")
            self._preview_label.pack()
            exclude_from_capture(self._preview)
        elif not on and self._preview is not None:
            self._preview.destroy()
            self._preview, self._preview_label, self._preview_image = None, None, None

    # -- periodic update ------------------------------------------------------------
    def _tick(self) -> None:
        if self._closed:
            return
        try:
            self._apply_spec()
            self._refresh()
        finally:
            self.root.after(120, self._tick)

    def _window_rect(self, win: tk.Misc) -> tuple[int, int, int, int]:
        x, y = win.winfo_rootx(), win.winfo_rooty()
        return x - 8, y - 40, x + win.winfo_width() + 8, y + win.winfo_height() + 8  # incl. title bar

    def _refresh(self) -> None:
        frame = self.watcher.latest
        if not self.hidden:  # our windows show up in screenshots: blank them out of the frames
            rects = [self._window_rect(self.root)]
            if self._preview is not None:
                rects.append(self._window_rect(self._preview))
            self.watcher.ignore_rects = rects
        with self._lock:
            analysis, searching = self._analysis, self._searching
        pos = self.tracker.current
        res = frame.result if frame else None

        # status line
        if self.watcher.error:
            self.status.config(text=self.watcher.error, fg=RED_HEX)
        elif res is None or res.geometry is None:
            self.status.config(text="Looking for a Quoridor board on the screen...", fg=WARN)
        elif not res.ok:
            covered = "" if self.hidden or not self._covering(frame) else \
                "  (an assistant window covers the game: move it away)"
            self.status.config(text="Board found, not readable: " + res.reason + covered, fg=WARN)
        elif self.tracker.problem:
            self.status.config(text="Position rejected: " + self.tracker.problem, fg=WARN)
        elif pos is not None and pos.winner is not None:
            self.status.config(text=f"Game over: {PLAYER_NAMES[pos.winner]} reached the goal", fg=GOOD)
        elif pos is not None and pos.turn is None:
            self.status.config(text="Board locked, but whose turn? Choose it under 'Side to move'.", fg=WARN)
        else:
            self.status.config(text=f"Board locked ({frame.parse_ms:.0f} ms per frame)", fg=GOOD)

        # detected state
        if pos is not None:
            who = PLAYER_NAMES[pos.turn] if pos.turn is not None else "unknown"
            src = f"  (from {self.tracker.turn_source})" if self.tracker.turn_source and pos.turn is not None else ""
            self.turn_label.config(text=f"Turn: {who}{src}",
                                   fg=(RED_HEX, BLUE_HEX)[pos.turn] if pos.turn is not None else WARN)
            self.blue_label.config(text=f"Blue walls left: {pos.walls_left[BLUE]}/10")
            self.red_label.config(text=f"Red walls left: {pos.walls_left[RED]}/10")
            parts = pos.summary().split(", ")
            side = f"Red starts at the {'top' if pos.flipped else 'bottom'} (from {self.tracker.orientation_source})"
            self.pos_label.config(text=f"{parts[0]} · {parts[1]} · {len(pos.walls)} walls on board\n{side}")
        else:
            self.turn_label.config(text="Turn: -", fg=TEXT)
            self.blue_label.config(text="Blue walls left: -")
            self.red_label.config(text="Red walls left: -")
            self.pos_label.config(text="")
        warnings = list(res.warnings) if res is not None and res.ok else []
        if frame is not None and not self.hidden and res is not None and res.geometry is None:
            warnings.append("Keep this panel off the game: the area under it is not scanned.")
        self.warn_label.config(text="\n".join("! " + w for w in warnings))

        self._show_analysis(analysis if (analysis and analysis.position == pos) else None, searching, pos)
        self._show_overlay(frame, analysis if (analysis and analysis.position == pos) else None)
        self._show_preview(frame, pos)
        self._keep_clear(frame)

    def _show_analysis(self, analysis: Optional[Analysis], searching: Optional[Position],
                       pos: Optional[Position]) -> None:
        ev = analysis.eval if analysis else None
        self._draw_bar(ev.red_share() if ev else None, bool(pos and pos.flipped))
        if analysis is None or analysis.best is None:
            thinking = searching is not None and pos is not None
            self.eval_label.config(text="..." if thinking else "-", fg=TEXT)
            self.eval_sub.config(text="thinking" if thinking else "")
            self.move_label.config(text="..." if thinking else "-")
            self.move_sub.config(text="")
            self.alt_label.config(text="-")
            self.info_label.config(text=analysis.error if analysis and analysis.error else "")
            return
        best, me = analysis.best, analysis.position.turn
        if ev is not None:
            leader = ev.leader()
            shown = ev.for_player(leader if leader is not None else RED)
            self.eval_label.config(text=shown.text(), fg=TEXT if leader is None else (RED_HEX, BLUE_HEX)[leader])
            side = "equal" if leader is None else f"better for {PLAYER_NAMES[leader]}"
            if ev.mate is not None:
                side = f"{PLAYER_NAMES[ev.winner]} wins by force"
            self.eval_sub.config(text=f"{side}  ·  {PLAYER_NAMES[me]} to move: {ev.for_player(me).text()}")
        else:
            self.eval_label.config(text="-", fg=TEXT)
            self.eval_sub.config(text="this bot reports no score")
        self.move_label.config(text=best.notation)
        kind = "pawn to " + best.notation if isinstance(best.move, PawnMove) else (
            ("horizontal" if best.notation.endswith("h") else "vertical") + " wall " + best.notation)
        self.move_sub.config(text=f"{PLAYER_NAMES[me]}: {kind}" + (f", then {best.reply}" if best.reply else ""))
        rows = []
        for i, line in enumerate(analysis.lines[1:], start=2):
            score = line.eval.for_player(me).text() if line.eval else "?"
            rows.append(f"{i}. {line.notation:<4} {score:>5}" + (f"   then {line.reply}" if line.reply else ""))
        self.alt_label.config(text="\n".join(rows) if rows else ("..." if not analysis.complete else "-"))
        nodes = f"{analysis.nodes / 1e6:.1f}M" if analysis.nodes >= 1e6 else f"{analysis.nodes:,}"
        self.info_label.config(text=f"{analysis.engine} · depth {analysis.depth} · {nodes} positions · "
                                    f"{analysis.seconds:.2f} s")

    def _draw_bar(self, red_share: Optional[float], flipped: bool = False) -> None:
        """Each colour fills from its own home side: Red from the bottom unless the board is flipped."""
        c = self.bar
        w, h = int(c["width"]), int(c["height"])
        c.delete("all")
        if red_share is None:
            c.create_rectangle(0, 0, w, h, fill=CARD, outline="")
            return
        top, bottom = (RED_HEX, BLUE_HEX) if flipped else (BLUE_HEX, RED_HEX)
        split = round(h * (red_share if flipped else 1 - red_share))
        c.create_rectangle(0, 0, w, split, fill=top, outline="")
        c.create_rectangle(0, split, w, h, fill=bottom, outline="")
        c.create_line(0, h // 2, w, h // 2, fill=BG, width=2)

    def _show_overlay(self, frame: Optional[Frame], analysis: Optional[Analysis]) -> None:
        g = frame.result.geometry if frame else None
        advise = {"Both sides": None, "Red": RED, "Blue": BLUE}[self.advise_var.get()]
        if (not self.overlay_var.get() or analysis is None or analysis.best is None or g is None or frame is None
                or not frame.result.ok or (advise is not None and analysis.position.turn != advise)):
            self.overlay.hide()
            return
        self._mark_move(frame, analysis.position, analysis.best.move, analysis.best.notation)

    def _mark_move(self, frame: Frame, position: Position, move, label: str) -> None:
        """Outline ``move`` (screen coordinates) on the board: the target square or the wall slot."""
        g = frame.result.geometry
        origin = None
        if isinstance(move, PawnMove):
            x1, y1, x2, y2 = g.cell_rect(move.row, move.col)
            if position.turn is not None:
                origin = frame.to_screen(*g.center(*position.pawns[position.turn]))
        else:
            x1, y1, x2, y2 = g.wall_rect(move.row, move.col, move.orientation.value, g.gap + 0.2 * g.side)
        sx1, sy1 = frame.to_screen(x1, y1)
        sx2, sy2 = frame.to_screen(x2, y2)
        area = (frame.left, frame.top, *frame.to_screen(frame.width, frame.height))
        self.overlay.show(area, (sx1, sy1, sx2, sy2), label, origin)

    def _show_preview(self, frame: Optional[Frame], pos: Optional[Position]) -> None:
        if self._preview_label is None or frame is None or frame.image is None:
            return
        if frame.number == self._preview_frame or frame.number % 2:
            return
        self._preview_frame = frame.number
        turn = pos.turn if (pos is not None and frame.result.ok) else None
        img = annotate(frame.image, frame.result, turn)
        limit = (900, 700) if self.hidden else (440, 400)  # small where it could cover the game
        scale = min(1.0, limit[0] / img.shape[1], limit[1] / img.shape[0])
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, png = cv2.imencode(".png", img)
        if ok:
            self._preview_image = tk.PhotoImage(data=base64.b64encode(png.tobytes()))
            self._preview_label.config(image=self._preview_image, text="")

    def _game_rect(self, frame: Optional[Frame]) -> Optional[tuple[int, int, int, int]]:
        """Screen rectangle of the board and the player cards."""
        rect = frame.result.game_rect() if frame else None
        if frame is None or rect is None:
            return None
        return (*frame.to_screen(rect[0], rect[1]), *frame.to_screen(rect[2], rect[3]))

    def _covering(self, frame: Optional[Frame]) -> bool:
        game = self._game_rect(frame)
        wins = [self.root] + ([self._preview] if self._preview is not None else [])
        return game is not None and any(_intersects(self._window_rect(w), game) for w in wins)

    def _keep_clear(self, frame: Optional[Frame]) -> None:
        """Once per located board: move our windows off the board and the player cards."""
        game = self._game_rect(frame)
        if game is None or self.hidden or frame is None:
            return
        wins = [self.root] + ([self._preview] if self._preview is not None else [])
        key = (game, tuple((w.winfo_width(), w.winfo_height()) for w in wins))
        if key == self._placed_for:
            return
        self._placed_for = key
        screen = (frame.left, frame.top, *frame.to_screen(frame.width, frame.height))
        gx1, gy1, gx2, gy2 = game
        placed: list[tuple[int, int, int, int]] = []
        for win in wins:
            rect = self._window_rect(win)
            w, h = rect[2] - rect[0], rect[3] - rect[1]
            spots = [(rect[0], rect[1]), (gx1 - w - 4, gy1), (gx2 + 4, gy1), (gx1, gy2 + 4), (gx1, gy1 - h - 4),
                     (screen[0], screen[1]), (screen[2] - w, screen[1])]
            spots += [(p[0], p[3] + 4) for p in placed] + [(p[2] + 4, p[1]) for p in placed]
            for x, y in spots:
                x, y = max(screen[0], x), max(screen[1], y)
                cand = (x, y, x + w, y + h)
                if cand[2] <= screen[2] and cand[3] <= screen[3] and not any(
                        _intersects(cand, other) for other in [game, *placed]):
                    if (x, y) != (rect[0], rect[1]):
                        win.geometry(f"+{x + 8}+{y + 40}")
                    placed.append(cand)
                    break
            else:
                placed.append(rect)  # no free spot: the user has to move it

    # -- lifecycle ----------------------------------------------------------------------
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.watcher.stop()
        self.analyzer.close()
        self.overlay.hide()
        self.root.destroy()

    def run(self) -> None:
        try:
            self.root.mainloop()
        finally:
            if not self._closed:
                self.watcher.stop()
                self.analyzer.close()
