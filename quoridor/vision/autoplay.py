"""Auto-Play: the bot plays the bottom player's moves with the mouse.

Built on the Screen Assistant's parser, tracker and analyser. This module
holds everything that does not need a window:

* :class:`ClickProfile`: how a move is entered on the site (where to click
  for a wall, pauses, retries). Loaded from JSON, so it is easy to adjust.
* :func:`pawn_point` / :func:`wall_points`: board geometry -> click points.
* :class:`PyAutoGuiMouse` / :class:`FakeMouse`: the mouse (real / for tests).
* :class:`ClockWatcher`: which player's clock is running, from the pixels of
  the two player cards changing over time, and which card is highlighted.
* :class:`AutoPlayer`: the state machine. It locks the bot's colour at the
  start position (the bottom player), counts the moves from move 0, decides
  from several signals whether it is really its turn, performs the engine's
  move, and waits until the board shows that move before doing anything else.

With *Continuous Play* on, the bot does not stop when a game ends. It cycles
``searching`` (no game: watch the screen, never click) -> in-game -> ``over``
(a short cool-down, never click) -> ``searching`` again, re-detecting its colour
at every new start position. With it off, auto-play switches itself off once
the game is over.

Safety: the bot never clicks unless it is armed, in a game, the move count says
it is its turn, and no other signal says it is the opponent's. :class:`StopSwitch`
(Esc, the panel's button, or the mouse in a screen corner) halts it at once,
also in the middle of a mouse movement.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field, fields, replace
from typing import Callable, Optional

import numpy as np

from ..moves import Move, PawnMove, WallMove
from .analysis import Analysis
from .capture import Frame
from .parser import BLUE, N, PLAYER_NAMES, RED, Geometry, ParseResult
from .position import Position, move_notation

WALL_CLICKS = ("first", "second", "center")


# ---------------------------------------------------------------------- #
# How moves are entered
# ---------------------------------------------------------------------- #
@dataclass
class ClickProfile:
    """How a move is entered on the site. Save changed values as JSON (``--click-profile``)."""

    #: Pawn move: "target" = click the target square; "pawn-then-target" = click the
    #: pawn first; "drag" = drag the pawn onto the target square.
    pawn_click: str = "target"
    #: Where a wall is clicked, in the gap between the squares: next to the "first"
    #: of the two squares it lies along (the left one of a horizontal wall, the
    #: upper one of a vertical wall), next to the "second", or at the "center"
    #: (the crossing point in the middle of the wall).
    wall_click: str = "first"
    #: "click" = hover over that point, then click. "drag" = drag from the bot's
    #: own wall reserve (its tick bar in the player card) to that point.
    wall_gesture: str = "click"
    #: Keys pressed before a horizontal / vertical wall is placed, for sites with
    #: a wall mode or an orientation toggle (e.g. "w", or "h" and "v"). "" = none.
    wall_key_h: str = ""
    wall_key_v: str = ""
    move_seconds: tuple = (0.1, 0.3)   # cursor travel time: shortest and longest (by distance)
    hover_pause: float = 0.2           # rest on the spot before pressing, so the site sees the hover
    press_seconds: float = 0.05        # how long the button stays down
    check_preview: bool = True         # look at the site's wall preview before clicking; skip a wrong slot
    confirm_timeout: float = 4.0       # seconds to wait for the board to show the move
    retries: int = 1                   # extra attempts if the board does not change
    think_delay: float = 0.0           # extra wait before every move, in seconds
    highlight_seconds: float = 1.0     # the move is shown on the board this long before the bot clicks

    @classmethod
    def load(cls, path: str) -> "ClickProfile":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown click-profile key(s): {', '.join(sorted(unknown))}")
        profile = cls(**data)
        profile.move_seconds = tuple(profile.move_seconds)
        profile.validate()
        return profile

    def validate(self) -> None:
        if self.pawn_click not in ("target", "pawn-then-target", "drag"):
            raise ValueError("pawn_click must be target, pawn-then-target or drag")
        if self.wall_click not in WALL_CLICKS:
            raise ValueError("wall_click must be first, second or center")
        if self.wall_gesture not in ("click", "drag"):
            raise ValueError("wall_gesture must be click or drag")

    def to_json(self) -> str:
        return json.dumps({f.name: getattr(self, f.name) for f in fields(self)}, indent=2)


def pawn_point(g: Geometry, move: PawnMove) -> tuple[float, float]:
    """Image pixel at the centre of the target square."""
    return g.center(move.row, move.col)


def wall_points(g: Geometry, move: WallMove) -> dict[str, tuple[float, float]]:
    """Image pixels in the groove where a wall can be clicked, by name (see ``ClickProfile.wall_click``)."""
    r, c = move.row, move.col
    if move.orientation.value == "h":   # in the gap under squares (r, c) and (r, c + 1)
        return {"first": g.center(r + 0.5, c), "second": g.center(r + 0.5, c + 1), "center": g.center(r + 0.5, c + 0.5)}
    return {"first": g.center(r, c + 0.5), "second": g.center(r + 1, c + 0.5), "center": g.center(r + 0.5, c + 0.5)}


def position_after(pos: Position, player: int, move: Move) -> Position:
    """The screen position after ``player`` makes ``move`` (screen coordinates); the other player to move."""
    if isinstance(move, PawnMove):
        pawns = list(pos.pawns)
        pawns[player] = (move.row, move.col)
        return replace(pos, pawns=(pawns[0], pawns[1]), turn=1 - player)
    left = list(pos.walls_left)
    left[player] -= 1
    walls = tuple(sorted([*pos.walls, (move.row, move.col, move.orientation.value)]))
    return replace(pos, walls=walls, walls_left=(left[0], left[1]), turn=1 - player)


def same_board(a: Optional[Position], b: Optional[Position]) -> bool:
    """Same pawns and walls (wall counters and turn are not compared: they may lag behind the board)."""
    return a is not None and b is not None and (a.pawns, a.walls) == (b.pawns, b.walls)


def legal_move_between(before: Position, after: Position, player: int) -> bool:
    """Is ``after``'s board reached from ``before`` by exactly one legal move of ``player``?"""
    from ..moves import Orientation

    board = before.to_board()  # the bots' board: mirrored when the screen is flipped
    if before.pawns[1 - player] != after.pawns[1 - player]:
        return False
    new = set(after.walls) - set(before.walls)
    if before.pawns[player] != after.pawns[player]:
        if before.walls != after.walls:
            return False
        target = before.screen_move(PawnMove(*after.pawns[player]))  # the mirror is its own inverse
        return (target.row, target.col) in board.valid_pawn_moves(player)
    if len(new) != 1 or not set(before.walls) <= set(after.walls) or board.walls_left[player] <= 0:
        return False
    r, c, o = next(iter(new))
    wall = before.screen_move(WallMove(r, c, Orientation(o)))
    return board.is_valid_wall(player, (wall.row, wall.col, wall.orientation))


# ---------------------------------------------------------------------- #
# Stop switch and mouse
# ---------------------------------------------------------------------- #
class EmergencyStop(Exception):
    """Raised inside a mouse action when the user stops the bot."""


class StopSwitch:
    """One flag every mouse step checks. Set by Esc, the panel button or the corner fail-safe."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self.reason = ""
        self._listener = None

    def trigger(self, reason: str = "stopped") -> None:
        if not self._event.is_set():
            self.reason = reason
            self._event.set()

    def clear(self) -> None:
        self.reason = ""
        self._event.clear()

    @property
    def stopped(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise EmergencyStop(self.reason)

    # -- global Esc key ----------------------------------------------------
    def listen_for_escape(self) -> str:
        """Watch the Esc key system-wide. Returns how ("windows", "pynput") or "" if that is not possible."""
        if sys.platform == "win32":
            import ctypes

            def poll() -> None:
                user32 = ctypes.windll.user32
                while True:
                    if user32.GetAsyncKeyState(0x1B) & 0x8000:
                        self.trigger("Esc pressed")
                    time.sleep(0.03)

            threading.Thread(target=poll, name="autoplay-esc", daemon=True).start()
            return "windows"
        try:
            from pynput import keyboard

            def on_press(key) -> None:
                if key == keyboard.Key.esc:
                    self.trigger("Esc pressed")

            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.daemon = True
            self._listener.start()
            return "pynput"
        except Exception:
            return ""


class PyAutoGuiMouse:
    """The real mouse and keyboard, through pyautogui (its corner fail-safe stays on)."""

    def __init__(self) -> None:
        import pyautogui

        pyautogui.FAILSAFE = True   # cursor in a screen corner -> FailSafeException
        pyautogui.PAUSE = 0
        self._gui = pyautogui

    def _call(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except self._gui.FailSafeException:
            raise EmergencyStop("mouse moved to a screen corner") from None

    def position(self) -> tuple[int, int]:
        p = self._call(self._gui.position)
        return int(p[0]), int(p[1])

    def move_to(self, x: float, y: float) -> None:
        self._call(self._gui.moveTo, round(x), round(y), _pause=False)

    def down(self) -> None:
        self._call(self._gui.mouseDown, _pause=False)

    def up(self) -> None:
        self._call(self._gui.mouseUp, _pause=False)

    def key(self, name: str) -> None:
        self._call(self._gui.press, name, _pause=False)


class FakeMouse:
    """Records what would be done (tests, dry runs)."""

    def __init__(self, start: tuple[int, int] = (5, 500)) -> None:
        self.pos = start
        self.events: list[tuple] = []

    def position(self) -> tuple[int, int]:
        return self.pos

    def move_to(self, x: float, y: float) -> None:
        self.pos = (round(x), round(y))

    def down(self) -> None:
        self.events.append(("down", *self.pos))

    def up(self) -> None:
        self.events.append(("up", *self.pos))

    def key(self, name: str) -> None:
        self.events.append(("key", name))

    @property
    def clicks(self) -> list[tuple[int, int]]:
        return [(e[1], e[2]) for e in self.events if e[0] == "down"]


class Hand:
    """Moves a mouse the way a site needs it: glide there, rest, press, release. Stops at once on request."""

    def __init__(self, mouse, profile: ClickProfile, stop: StopSwitch, sleep: Callable[[float], None] = time.sleep) -> None:
        self.mouse, self.profile, self.stop, self._sleep = mouse, profile, stop, sleep

    def wait(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while True:
            self.stop.check()
            left = end - time.monotonic()
            if left <= 0:
                return
            self._sleep(min(0.02, left))

    def glide(self, x: float, y: float) -> None:
        x0, y0 = self.mouse.position()
        lo, hi = self.profile.move_seconds
        distance = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5
        seconds = min(hi, max(lo, distance / 2500))
        steps = max(1, round(seconds / 0.015))
        for i in range(1, steps + 1):
            self.stop.check()
            t = i / steps
            t = t * t * (3 - 2 * t)  # ease in and out
            self.mouse.move_to(x0 + (x - x0) * t, y0 + (y - y0) * t)
            if i < steps:
                self._sleep(seconds / steps)

    def click(self, x: float, y: float) -> None:
        self.glide(x, y)
        self.wait(self.profile.hover_pause)
        self.press()

    def press(self) -> None:
        self.stop.check()
        self.mouse.down()
        try:
            self._sleep(self.profile.press_seconds)
        finally:
            self.mouse.up()

    def drag(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.glide(x1, y1)
        self.wait(self.profile.hover_pause)
        self.stop.check()
        self.mouse.down()
        try:
            self.glide(x2, y2)
            self.wait(self.profile.hover_pause)
        finally:
            self.mouse.up()


# ---------------------------------------------------------------------- #
# Clocks
# ---------------------------------------------------------------------- #
class ClockWatcher:
    """Which player's clock is running, without reading it.

    Only the active player's clock changes every second, so the pixels of that
    player's card keep changing while the other card stays the same. A card
    that changed at least twice in the last few seconds while the other did
    not change at all belongs to the player to move.

    The brightness of the two cards is remembered too: if one card turns out
    to be brighter whenever its player is to move (a highlighted clock box),
    :meth:`highlight_turn` uses that as a further signal.
    """

    WINDOW = 3.5      # seconds of history
    MIN_GAP = 0.5     # changes closer together than this count once (an animation, not a tick)

    def __init__(self) -> None:
        self._last: list[Optional[np.ndarray]] = [None, None]
        self._rects: list[Optional[tuple]] = [None, None]
        self._changes: list[list[float]] = [[], []]
        self._since: list[float] = [0.0, 0.0]
        self._brightness: Optional[float] = None       # Red card minus Blue card, latest frame
        self._samples: list[list[float]] = [[], []]    # that difference on Red's / Blue's turns

    def reset(self) -> None:
        self.__init__()

    @staticmethod
    def card_rect(res: ParseResult, player: int, shape) -> Optional[tuple[int, int, int, int]]:
        card, g = res.cards[player], res.geometry
        if card.dot is None or g is None or res.sidebar is None:
            return None
        parts = [card.dot, *card.ticks]
        x1 = max(0, round(card.dot[0] - 0.3 * g.pitch))
        x2 = min(shape[1], res.sidebar[2], round(card.dot[0] + 6 * g.pitch))
        y1 = max(0, round(min(p[1] for p in parts) - 0.4 * g.pitch))
        y2 = min(shape[0], round(max(p[3] for p in parts) + 0.4 * g.pitch))
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    def update(self, image: Optional[np.ndarray], res: ParseResult, now: float) -> None:
        if image is None or not res.ok:
            return
        means = []
        for p in (RED, BLUE):
            rect = self.card_rect(res, p, image.shape)
            if rect is None:
                self._last[p], self._rects[p] = None, None
                return
            crop = image[rect[1]:rect[3], rect[0]:rect[2]]
            if rect != self._rects[p] or self._last[p] is None or self._last[p].shape != crop.shape:
                self._rects[p], self._changes[p], self._since[p] = rect, [], now
            elif not np.array_equal(crop, self._last[p]):
                if not self._changes[p] or now - self._changes[p][-1] >= self.MIN_GAP:
                    self._changes[p].append(now)
            self._last[p] = crop.copy()
            self._changes[p] = [t for t in self._changes[p] if now - t <= self.WINDOW]
            means.append(float(crop.mean()))
        self._brightness = means[RED] - means[BLUE]

    def clock_turn(self, now: float) -> Optional[int]:
        """Player whose clock is ticking (None: no clocks, both still, or not watched long enough)."""
        if min(now - s for s in self._since) < self.WINDOW or None in self._rects:
            return None
        counts = [len([t for t in self._changes[p] if now - t <= self.WINDOW]) for p in (RED, BLUE)]
        for p in (RED, BLUE):
            if counts[p] >= 2 and counts[1 - p] == 0:
                return p
        return None

    def learn(self, turn: int) -> None:
        """Tell the watcher whose turn it certainly is now, so it can learn the highlight."""
        if self._brightness is not None:
            self._samples[turn] = (self._samples[turn] + [self._brightness])[-12:]

    def highlight_turn(self) -> Optional[int]:
        """Player whose card is highlighted, once the highlight has been seen to follow the turn."""
        if self._brightness is None or min(len(s) for s in self._samples) < 2:
            return None
        red, blue = (float(np.median(s)) for s in self._samples)
        spread = max(float(np.ptp(s)) for s in self._samples)
        if abs(red - blue) < max(2.5, 3 * spread):
            return None  # the cards look the same whoever is to move
        return RED if abs(self._brightness - red) < abs(self._brightness - blue) else BLUE


# ---------------------------------------------------------------------- #
# The player
# ---------------------------------------------------------------------- #
@dataclass
class Signals:
    """What each indicator says about the side to move in the current frame (None = says nothing)."""

    ply: Optional[int] = None
    border: Optional[int] = None
    move_list: Optional[int] = None
    outlines: Optional[int] = None
    clock: Optional[int] = None
    highlight: Optional[int] = None

    def text(self) -> str:
        parts = []
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None:
                parts.append(f"{f.name.replace('_', ' ')}: {PLAYER_NAMES[value]}")
        return ", ".join(parts) or "none"


@dataclass
class Identity:
    """Which colour the bot plays and what that was read from.

    The bot always plays the player who starts on the bottom row (rank 1), so its
    goal is always the top row (rank 9): it moves up the screen.
    """

    me: int
    evidence: list[str] = field(default_factory=list)

    def text(self) -> str:
        return f"{PLAYER_NAMES[self.me]} (bottom), goal: top row (rank 9), moving up"


def identify_bottom_player(res: ParseResult, require_start: bool = True) -> tuple[Optional[Identity], str]:
    """The colour of the bottom player, cross-checked. Returns ``(identity, problem)``.

    Three independent signs: the pawn standing on the bottom row (in the start
    position), the edge-row tint, and the colour of the lower player card. They
    must not contradict each other. The tint marks GOAL rows: a red bottom row is
    Red's goal, so Red starts at the top and the bottom player is Blue.
    """
    if not res.ok or res.pawns is None:
        return None, "board not readable"
    votes: dict[str, int] = {}
    # The pawns on e1 / e9 with no wall placed can only be a start position. The wall
    # counters are not required to read 10/10: one misread tick must not block the bot.
    at_start = len(res.walls) == 0 and {res.pawns[RED], res.pawns[BLUE]} == {(0, N // 2), (N - 1, N // 2)}
    if at_start:
        votes["pawn on the bottom row"] = RED if res.pawns[RED][0] == N - 1 else BLUE
    elif require_start:
        return None, "waiting for the start position of a new game"
    if res.flipped is not None:
        votes["edge-row tint (goal rows)"] = BLUE if res.flipped else RED
    dots = [res.cards[p].dot for p in (RED, BLUE)]
    if None not in dots and abs(dots[RED][1] - dots[BLUE][1]) > (res.geometry.pitch if res.geometry else 0):
        votes["bottom player card"] = RED if dots[RED][1] > dots[BLUE][1] else BLUE
    if not votes:
        return None, "cannot tell which colour is at the bottom (no tinted row, no player cards)"
    if len(set(votes.values())) > 1:
        detail = ", ".join(f"{name}: {PLAYER_NAMES[p]}" for name, p in votes.items())
        return None, f"signs disagree about the bottom player ({detail})"
    return Identity(next(iter(votes.values())), list(votes)), ""


class AutoPlayer:
    """Decides when to move and makes the move. Driven by :meth:`tick` a few times per second."""

    SOFT_OVERRIDE = 6.0     # seconds after which a lone, fallible signal (move list, highlight) is overruled
    OUTLINE_WAIT = 5.0      # seconds to wait for legal-move outlines once the site has shown them before
    CONFIRM_FRAMES = 2      # frames in a row that must show the expected board
    NEW_GAME_FRAMES = 2     # Continuous Play: frames in a row showing a start position before a game begins
    BOARD_GONE = 3.0        # Continuous Play: seconds without a readable board that end the game (modal, lobby)
    COOLDOWN = 2.0          # Continuous Play: seconds in "over" before searching for the next game
    #: States in which the bot is playing a game; the only ones in which it may click.
    IN_GAME = ("opponent", "thinking", "highlighting", "moving", "confirming")

    def __init__(self, mouse, profile: Optional[ClickProfile] = None, stop: Optional[StopSwitch] = None,
                 join_in_progress: bool = False, clock: Callable[[], float] = time.monotonic,
                 threaded: bool = True, continuous: bool = False) -> None:
        self.profile = profile or ClickProfile()
        self.stop = stop or StopSwitch()
        self.hand = Hand(mouse, self.profile, self.stop)
        self.clocks = ClockWatcher()
        self.join_in_progress = join_in_progress
        #: Continuous Play: keep going from one game to the next instead of stopping after one.
        self.continuous = continuous
        self._clock = clock
        self._threaded = threaded
        self.armed = False
        self.identity: Optional[Identity] = None
        self.ply: Optional[int] = None          # moves played so far, counted from the start position
        #: off, waiting-start (one game), searching (Continuous Play), opponent, thinking,
        #: highlighting, moving, confirming, over
        self.state = "off"
        self.message = "Auto-play is off"
        self.signals = Signals()
        self.last_move = ""                     # notation of the move last played by the bot
        self.moves_made = 0
        self._board: Optional[Position] = None  # last position accounted for in the move count
        self._frame_no = -1
        self._my_turn_since: Optional[float] = None
        self._outlines_seen = False
        self._expected: Optional[Position] = None
        self._before: Optional[Position] = None
        self._pending: Optional[Move] = None
        self._deadline = 0.0
        self._after_frame = 0
        self._good_frames = 0
        self._retries_left = 0
        self._worker: Optional[threading.Thread] = None
        self._worker_error: Optional[str] = None
        self._latest: Optional[Frame] = None
        self.planned: Optional[Move] = None     # the move being highlighted / played (screen coordinates)
        self._highlight_until = 0.0
        #: Called in the thread that calls tick(), right before the mouse starts moving,
        #: so the window can hide its move marker and the click lands on the game.
        self.before_click: Optional[Callable[[], None]] = None
        #: Called (in the tick() thread) when a game ends, with the result text, e.g. so
        #: the window can make the screen reader forget the old board.
        self.on_game_end: Optional[Callable[[str], None]] = None
        self.games: list[str] = []              # results of the games finished while armed
        self._last_readable: Optional[float] = None   # last time the frame showed a readable board
        self._start_frames = 0                  # consecutive new frames showing a start position
        self._finished: Optional[Position] = None     # final board of the last game: never joined again
        self._over_until = 0.0
        self.log_lines: deque = deque(maxlen=8)
        self._noted = ""

    # -- log -----------------------------------------------------------------
    def log(self, text: str) -> None:
        """One step of what the bot does, on the console and in the panel."""
        line = time.strftime("%H:%M:%S ") + text
        self.log_lines.append(line)
        print("[auto-play] " + line, flush=True)

    def _note(self, state: str, message: str) -> None:
        """Set the state and message; log the message when it changes (waiting reasons, not every tick)."""
        self.state, self.message = state, message
        if message != self._noted:
            self._noted = message
            self.log(message)

    # -- control -----------------------------------------------------------
    @property
    def me(self) -> Optional[int]:
        return self.identity.me if self.identity else None

    @property
    def in_game(self) -> bool:
        return self.armed and self.identity is not None and self.state in self.IN_GAME

    def _wait_message(self) -> str:
        return "Searching for new game..." if self.continuous else "Waiting for the start position of a game..."

    def _waiting_state(self) -> str:
        return "searching" if self.continuous else "waiting-start"

    def arm(self) -> None:
        self.stop.clear()
        self.armed = True
        self._my_turn_since = None
        self._last_readable = None
        self._start_frames = 0
        self.log("Auto-play switched on" + (" (Continuous Play)" if self.continuous else ""))
        if self.identity is not None and self.ply is not None:
            self._note("opponent", "Waiting for opponent...")  # carry on with the game being followed
        elif self.state in ("off", "over", "waiting-start", "searching"):
            self._note(self._waiting_state(), self._wait_message())

    def disarm(self, message: str = "Auto-play is off") -> None:
        self.armed = False
        self._note("off", message)
        self._expected = self._pending = self.planned = None

    def _halt(self, message: str) -> None:
        """Stop clicking because something is wrong; the user has to look and re-arm."""
        self.disarm("Stopped: " + message)

    # -- per frame ----------------------------------------------------------
    def tick(self, frame: Optional[Frame], current: Optional[Position], analysis: Optional[Analysis]) -> None:
        """Call regularly with the latest frame, the tracker's stable position and its analysis."""
        now = self._clock()
        self._latest = frame
        if frame is not None and frame.number != self._frame_no:
            self.clocks.update(frame.image, frame.result, now)
        if self.stop.stopped and self.armed:
            self.disarm(f"STOPPED ({self.stop.reason}). Press Start to continue.")
        if self._worker is not None:
            if self._worker.is_alive():
                return
            self._worker = None
            self._after_move(frame, now)
        if frame is None:
            return
        new_frame = frame.number != self._frame_no
        self._frame_no = frame.number
        res = frame.result
        live = self._live_position(res, current)

        if res.ok:
            self._last_readable = now
        elif self._last_readable is None:
            self._last_readable = now
        if self.armed and self.state == "over":
            # Game over / cool-down: never click, just wait, then search for the next game.
            if now < self._over_until:
                return
            self._note(self._waiting_state(), self._wait_message())
        if self.armed and self.state in ("waiting-start", "searching") and self.identity is None:
            if self._searching_step(res, live, new_frame):
                return
        self._follow_game(res, current, live)
        if not self.armed or self.identity is None:
            return
        if self.state in ("waiting-start", "searching", "off"):
            # armed with the colour already locked (e.g. joined while switched off): play on by the turn signals
            self._note("opponent", "Waiting for a sign that it is my turn...")
        if self._check_game_end(res, current, live, now):
            return
        if self.state == "confirming":
            if new_frame:
                self._confirm(frame, now)
            return
        if self.state == "highlighting":
            self._highlight_step(res, live, now)
            return
        if live is None or current is None:
            return  # frame unreadable or the board is still changing
        self._decide(res, current, analysis, now)

    # -- game lifecycle (Continuous Play) -----------------------------------------------
    def _searching_step(self, res: ParseResult, live: Optional[Position], new_frame: bool) -> bool:
        """While no game is followed. Returns True to skip the rest of the tick (no game yet)."""
        if live is not None and self._finished is not None and same_board(live, self._finished):
            self._start_frames = 0
            self._note(self._waiting_state(), self._wait_message() + " (the last game's final board is still shown)")
            return True
        if not self.continuous:
            return False
        identity, _ = identify_bottom_player(res, require_start=True) if live is not None else (None, "")
        if identity is None:
            self._start_frames = 0
            if not (self.join_in_progress and live is not None and live.winner is None):
                self._note("searching", self._wait_message())
                return True
            return False
        if new_frame:
            self._start_frames += 1
        if self._start_frames < self.NEW_GAME_FRAMES:
            self._note("searching", "New game found, checking that the board is stable...")
            return True
        return False

    def _check_game_end(self, res: ParseResult, current: Optional[Position], live: Optional[Position],
                        now: float) -> bool:
        """Has the game being played ended? Ends it and returns True if so."""
        if self.state == "moving":
            return False   # the mouse is busy: decided once the click is done
        board = live if live is not None else (current if current is not None and res.ok else None)
        if board is not None and board.winner is not None and same_board(board, self._board or board):
            self._end_game("the bot won" if board.winner == self.me else "the opponent won", board)
            return True
        if self.state == "confirming" and self._expected is not None and self._expected.winner == self.me \
                and not res.ok and self._last_readable is not None and now - self._last_readable > 0.5:
            # the winning move was clicked and the site covered the board with its result screen
            self._end_game("the bot won (its winning move was followed by a result screen)", self._expected)
            return True
        if self.continuous and self._last_readable is not None and now - self._last_readable >= self.BOARD_GONE:
            self._end_game("the board is no longer visible (result screen, menu or lobby)", None)
            return True
        return False

    def _end_game(self, result: str, final: Optional[Position]) -> None:
        """Clear everything about the finished game; search for the next one, or switch off.

        ``final``: the finished game's last board, if it is still on the screen. It is
        never taken for a new game (only a decided board is passed: never a start position).
        """
        moves = "?" if self.ply is None else str(self.ply)
        colour = PLAYER_NAMES[self.me] if self.me is not None else "?"
        text = f"Game over: {result} (played as {colour}, {moves} moves, {self.moves_made} by the bot)"
        self.games.append(result)
        self.log(text)
        self._finished = final
        self._reset_game()
        if self.on_game_end is not None:
            try:
                self.on_game_end(result)
            except Exception as exc:
                self.log(f"game-end hook failed: {exc}")
        if self.continuous:
            self._over_until = self._clock() + self.COOLDOWN
            self.state, self.message = "over", text + ". Searching for the next game shortly..."
            self._noted = self.message
        else:
            self.disarm(text + ". Auto-play stopped (Continuous Play is off).")

    def _reset_game(self) -> None:
        """Forget the game: move count, board, colour lock, pending move, turn timers, clock history."""
        self.identity, self.ply, self._board = None, None, None
        self._expected = self._pending = self._before = self.planned = None
        self._my_turn_since = None
        self._outlines_seen = False
        self._start_frames = 0
        self._retries_left = 0
        self.moves_made = 0
        self.last_move = ""
        self.signals = Signals()
        self.clocks.reset()

    @staticmethod
    def _live_position(res: ParseResult, current: Optional[Position]) -> Optional[Position]:
        """``current`` if the newest frame shows exactly that board, else None."""
        if current is None or not res.ok or (res.pawns, res.walls) != (current.pawns, current.walls):
            return None
        return current

    # -- identity and move count -----------------------------------------------
    def _follow_game(self, res: ParseResult, current: Optional[Position], live: Optional[Position]) -> None:
        if live is None or self.state in ("moving", "confirming"):
            return
        identity, problem = identify_bottom_player(res, require_start=True)
        if identity is not None and (self.ply != 0 or self.identity is None or not same_board(self._board, live)):
            # the start position: a (new) game begins, lock the colour for this game
            if live.goal_rows[identity.me] != 0:
                # The bottom player must head for the top row; anything else is a misread orientation.
                self.identity = None
                self.message = "Start position seen, but the board orientation is unclear: set 'Red starts at'."
                return
            new_game = self.identity is None or self.ply != 0 or self.identity.me != identity.me
            if new_game and self.identity is not None and self.armed:
                self.log("A new game started before the last one was seen to end")
                self.games.append("unknown (a new game started)")
                self._reset_game()
            if new_game:
                self.moves_made, self.last_move = 0, ""
            self._finished = None
            self._start_frames = 0
            self.identity, self.ply, self._board = identity, 0, live
            self._outlines_seen = False
            self.clocks.reset()
            if new_game:
                self.log(f"Start position: playing as {identity.text()} (from {', '.join(identity.evidence)})")
            if self.armed:
                self._note("opponent", "Waiting for opponent..." if identity.me == BLUE else "My turn (Red moves first)")
            return
        if self.identity is None:
            if self.armed and self.join_in_progress:
                identity, problem = identify_bottom_player(res, require_start=False)
                if identity is not None and live.goal_rows[identity.me] == 0 and live.winner is None \
                        and not same_board(live, self._finished):
                    self._finished = None
                    self.identity, self.ply, self._board = identity, None, live
                    self.log(f"Joined a game in progress: playing as {identity.text()}")
                    self._note("opponent", "Waiting for a sign that it is my turn...")
                    return
            if self.armed:
                self._note(self._waiting_state(), (self._wait_message() + " " if self.continuous else "Waiting: ")
                           + problem)
            return
        if self._board is None or same_board(live, self._board):
            return
        # the board changed: count the move
        if self.ply is not None:
            mover = self.ply % 2
            if legal_move_between(self._board, live, mover):
                self.ply += 1
            else:
                other = self._two_moves(self._board, live, mover)
                self.ply = self.ply + 2 if other else None
        self._board = live
        if self.ply is None and self.armed and not self.join_in_progress and live.winner is None:
            self._halt("lost track of the move count (moves happened that were not seen). "
                       "Tick 'Join a game in progress' to continue by the other turn signals.")

    @staticmethod
    def _two_moves(before: Position, after: Position, first: int) -> bool:
        """Two unseen moves, one by each player: only clear-cut cases (each side's change is one legal move)."""
        mid_pawns = list(before.pawns)
        mid_pawns[first] = after.pawns[first]
        first_walls = [w for w in set(after.walls) - set(before.walls)]
        if len(first_walls) > 2:
            return False
        candidates = [replace(before, pawns=(mid_pawns[0], mid_pawns[1]), turn=1 - first)]
        for w in first_walls:
            left = list(before.walls_left)
            left[first] -= 1
            candidates.append(replace(before, walls=tuple(sorted([*before.walls, w])),
                                      walls_left=(left[0], left[1]), turn=1 - first))
        for mid in candidates:
            try:
                if legal_move_between(before, mid, first) and legal_move_between(mid, after, 1 - first):
                    return True
            except Exception:
                continue
        return False

    # -- is it my turn? ----------------------------------------------------------
    def read_signals(self, res: ParseResult, now: float) -> Signals:
        return Signals(ply=None if self.ply is None else self.ply % 2, border=res.border_turn,
                       move_list=res.move_list_turn, outlines=res.hint_turn,
                       clock=self.clocks.clock_turn(now), highlight=self.clocks.highlight_turn())

    def _decide(self, res: ParseResult, current: Position, analysis: Optional[Analysis], now: float) -> None:
        me = self.me
        s = self.signals = self.read_signals(res, now)
        if s.outlines == me:
            self._outlines_seen = True
        # Hard: the site's own markings, which are never wrong about the turn. Soft: readings
        # that can be fooled (OCR, pixels that change for other reasons); they only delay.
        hard = {"border": s.border, "outlines": s.outlines}
        soft = {"move list": s.move_list, "clock": s.clock, "highlight": s.highlight}
        if s.ply is not None:
            mine = s.ply == me
            if mine and all(v in (None, me) for v in (*hard.values(), *soft.values())):
                self.clocks.learn(me)
            elif not mine and all(v in (None, 1 - me) for v in (*hard.values(), *soft.values())):
                self.clocks.learn(1 - me)
        else:
            # move count unknown (joined mid-game): a positive sign is needed
            mine = me in hard.values() or me in soft.values()
        if not mine:
            self._my_turn_since = None
            self._note("opponent", "Waiting for opponent...")
            return
        if self._my_turn_since is None:
            self._my_turn_since = now
        waited = now - self._my_turn_since
        against = [name for name, v in hard.items() if v == 1 - me]
        if against:
            self._note("opponent", f"Waiting: the move count says it is my turn, but the {' and '.join(against)} "
                                   f"say{'s' if len(against) == 1 else ''} it is the opponent's.")
            return
        doubts = [name for name, v in soft.items() if v == 1 - me]
        if doubts and waited < self.SOFT_OVERRIDE:
            self._note("opponent", f"Waiting for the {' and '.join(doubts)} to agree that it is my turn...")
            return
        if self._outlines_seen and s.outlines != me and waited < self.OUTLINE_WAIT:
            self._note("thinking", "My turn: waiting for the legal-move outlines...")
            return
        if waited < self.profile.think_delay:
            self._note("thinking", "Thinking...")
            return
        if analysis is None or analysis.best is None or not same_board(analysis.position, current) \
                or analysis.position.turn != me or analysis.position.flipped != current.flipped:
            self._note("thinking", "My turn: the engine is thinking...")
            return
        self._begin_highlight(current, analysis)

    # -- highlight, wait, click ------------------------------------------------------
    def _begin_highlight(self, current: Position, analysis: Analysis) -> None:
        """Show the move on the board first; it is clicked ``highlight_seconds`` later."""
        move = analysis.best.move
        self._before, self.planned = current, move
        self._highlight_until = self._clock() + max(0.0, self.profile.highlight_seconds)
        count = "move count unknown" if self.ply is None else f"move {self.ply}"
        score = analysis.eval.text() if analysis.eval else "?"
        self.log(f"Turn detected ({count}, {PLAYER_NAMES[self.me]} to move; signals: {self.signals.text()})")
        self.log(f"Best move {move_notation(move)} (eval {score}) -> highlighting it, "
                 f"clicking in {self.profile.highlight_seconds:g} s")
        self.state, self.message = "highlighting", f"Highlighting {move_notation(move)}..."
        self._noted = self.message

    def _highlight_step(self, res: ParseResult, live: Optional[Position], now: float) -> None:
        """While the move is shown: cancel if the board or the turn changes, else click when the time is up."""
        move, before = self.planned, self._before
        if move is None or before is None:
            self._note("thinking", "Thinking...")
            return
        if live is None or not same_board(live, before):
            if not res.ok and now < self._highlight_until + 2.0:
                return  # one unreadable frame (a redraw): keep waiting a little
            self.planned = None
            self._note("thinking", f"Board changed while {move_notation(move)} was highlighted: not clicking")
            return
        s = self.signals = self.read_signals(res, now)
        against = [name for name, v in (("border", s.border), ("outlines", s.outlines)) if v == 1 - self.me]
        if against:
            self.planned = None
            self._note("opponent", f"Not clicking: the {' and '.join(against)} now show the opponent to move")
            return
        left = self._highlight_until - now
        if left > 0:
            self.message = f"Highlighting {move_notation(move)}, clicking in {left:.1f} s..."
            return
        self._start_move(before, move, retries=self.profile.retries)

    # -- making the move -----------------------------------------------------------
    def _start_move(self, current: Position, move: Move, retries: int) -> None:
        frame = self._latest
        if frame is None or frame.result.geometry is None:
            return
        if not self.in_game or self.state not in ("highlighting", "confirming"):
            # Guardrail: never click while searching for a game, after a game or switched off.
            self.planned = None
            self.log(f"Not clicking {move_notation(move)}: no game is being played (state {self.state})")
            return
        self._before, self._pending, self._retries_left = current, move, retries
        self.planned = move
        self._expected = position_after(current, self.me, move)
        self._worker_error = None
        self.state, self.message = "moving", f"Clicking {move_notation(move)}..."
        self._noted = self.message
        self.last_move = move_notation(move)
        if self.before_click is not None:
            try:
                self.before_click()   # e.g. hide the move marker so it cannot catch the click
            except Exception as exc:
                self.log(f"could not hide the move marker: {exc}")
        if self._threaded:
            self._worker = threading.Thread(target=self._perform, args=(frame, current, move),
                                            name="autoplay-mouse", daemon=True)
            self._worker.start()
        else:
            self._perform(frame, current, move)
            self._after_move(self._latest, self._clock())

    def _perform(self, frame: Frame, current: Position, move: Move) -> None:
        try:
            self.perform(frame, current, move)
        except EmergencyStop as exc:
            self.stop.trigger(str(exc) or "stopped")
            self._worker_error = "stop"
        except Exception as exc:  # e.g. pyautogui cannot reach the display
            self._worker_error = f"mouse error: {exc}"

    def perform(self, frame: Frame, current: Position, move: Move) -> None:
        """Enter ``move`` (screen coordinates) with the mouse, then put the cursor back out of the way."""
        g, hand, prof = frame.result.geometry, self.hand, self.profile
        home = hand.mouse.position()
        bx1, by1 = frame.to_screen(*g.board_rect()[:2])
        bx2, by2 = frame.to_screen(*g.board_rect()[2:])
        if bx1 - 20 <= home[0] <= bx2 + 20 and by1 - 20 <= home[1] <= by2 + 20:
            home = ((bx1 + bx2) // 2, by2 + round(0.9 * g.pitch / frame.scale))  # never rest on the board
        try:
            if isinstance(move, PawnMove):
                target = frame.to_screen(*pawn_point(g, move))
                pawn = frame.to_screen(*g.center(*current.pawns[self.me]))
                self.log(f"Clicking {move_notation(move)}: pawn {prof.pawn_click} at {target}")
                if prof.pawn_click == "drag":
                    hand.drag(*pawn, *target)
                else:
                    if prof.pawn_click == "pawn-then-target":
                        hand.click(*pawn)
                        hand.wait(prof.hover_pause)
                    hand.click(*target)
            else:
                self._place_wall(frame, current, move)
        finally:
            if not self.stop.stopped:
                hand.wait(0.08)
                hand.glide(*home)

    def _place_wall(self, frame: Frame, current: Position, move: WallMove) -> None:
        g, hand, prof = frame.result.geometry, self.hand, self.profile
        key = prof.wall_key_h if move.orientation.value == "h" else prof.wall_key_v
        if key:
            hand.stop.check()
            hand.mouse.key(key)
            hand.wait(0.1)
        points = wall_points(g, move)
        target = (move.row, move.col, move.orientation.value)
        if prof.wall_gesture == "drag":
            ticks = frame.result.cards[self.me].ticks
            if not ticks:
                raise RuntimeError("cannot find the wall reserve (tick bar) to drag a wall from")
            t = ticks[-1]
            start = frame.to_screen((t[0] + t[2]) / 2, (t[1] + t[3]) / 2)
            end = frame.to_screen(*points[prof.wall_click])
            self.log(f"Clicking {move_notation(move)}: dragging a wall from {start} to {end}")
            hand.drag(*start, *end)
            return
        order = [prof.wall_click] + [w for w in WALL_CLICKS if w != prof.wall_click]
        for i, where in enumerate(order):
            hand.glide(*frame.to_screen(*points[where]))
            hand.wait(prof.hover_pause)
            seen = self._previewed_wall(current) if prof.check_preview else None
            if seen is None or seen == target:
                if seen == target and where != prof.wall_click:
                    prof.wall_click = where  # this is where this site wants the click: remember it
                    self.log(f"the site previews this wall at the '{where}' point: using that from now on")
                self.log(f"Clicking {move_notation(move)}: wall at the '{where}' point "
                         f"{frame.to_screen(*points[where])}")
                hand.press()
                return
            self.log(f"the site previews another wall at the '{where}' point: not clicking there")
            # The site previews a different wall here: do not click, try the next point.
        raise RuntimeError(f"the site previews another wall than {move_notation(move)} at every click point; "
                           "adjust 'wall_click' in the click profile")

    def _previewed_wall(self, current: Position) -> Optional[tuple]:
        """The wall the site shows as a preview under the cursor, if the parser can see one."""
        start = self._latest.number if self._latest else 0
        end = time.monotonic() + 0.8
        while time.monotonic() < end:
            self.stop.check()
            frame = self._latest
            if frame is not None and frame.number >= start + 2:
                res = frame.result
                extra = set(res.walls) - set(current.walls) if res.ok else set()
                if len(extra) == 1 and set(current.walls) <= set(res.walls):
                    return next(iter(extra))
                return None
            if not self._threaded:
                return None
            time.sleep(0.03)
        return None

    def _after_move(self, frame: Optional[Frame], now: float) -> None:
        if self._worker_error == "stop" or self.stop.stopped:
            self.disarm(f"STOPPED ({self.stop.reason}). Press Start to continue.")
            return
        if self._worker_error:
            self._halt(self._worker_error)
            return
        if not self.armed or self.state != "moving":
            return
        # State-change lock: nothing else happens until the board shows the move.
        self._note("confirming", f"Clicked {self.last_move}, waiting for the board to show it...")
        self._deadline = now + self.profile.confirm_timeout
        self._after_frame = (frame.number if frame else 0) + 1  # skip a frame captured during the click
        self._good_frames = 0

    def _confirm(self, frame: Frame, now: float) -> None:
        res, expected, before = frame.result, self._expected, self._before
        if frame.number <= self._after_frame or expected is None or before is None:
            return
        shown = (res.pawns, res.walls) if res.ok else None
        seen = Position(res.pawns, res.walls, res.walls_left, None, expected.flipped) if res.ok else None
        if shown == (expected.pawns, expected.walls):
            self._good_frames += 1
            if self._good_frames >= self.CONFIRM_FRAMES:
                self._moved(expected, 1)
            return
        if seen is not None and shown != (before.pawns, before.walls):
            # the opponent may have answered already
            try:
                answered = legal_move_between(expected, seen, 1 - self.me)
            except Exception:
                answered = False
            if answered:
                self._good_frames += 1
                if self._good_frames >= self.CONFIRM_FRAMES:
                    self._moved(seen, 2)
                return
        self._good_frames = 0
        if now < self._deadline:
            return
        if shown == (before.pawns, before.walls) and self._retries_left > 0 and self._pending is not None:
            self.log(f"{self.last_move} did not register, trying again")
            self._start_move(before, self._pending, self._retries_left - 1)
            return
        if shown is None and self.continuous:
            return  # board not readable: _check_game_end decides whether the game is over
        if shown == (before.pawns, before.walls):
            self._halt(f"the click for {self.last_move} did not change the board. Check the click profile "
                       "(wall_click / pawn_click) and that nothing covers the board.")
        else:
            self._halt(f"after playing {self.last_move} the board shows something unexpected.")

    def _moved(self, board: Position, plies: int) -> None:
        if self.ply is not None:
            self.ply += plies
        self._board = replace(board, turn=None)
        self.moves_made += 1
        self._expected = self._pending = self._before = self.planned = None
        self._my_turn_since = None
        self.log(f"{self.last_move} is on the board")
        self._note("opponent", "Waiting for opponent...")
