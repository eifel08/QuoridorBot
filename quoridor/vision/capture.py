"""Background screen-capture loop for the screen assistant.

:class:`ScreenWatcher` grabs the chosen monitor a few times per second with
``mss``, parses each frame and hands every *new* stable position to a
callback. While no board is locked it scans less often, because searching the
whole screen costs more than re-checking a known board.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .parser import ParseResult, ScreenParser
from .position import Position, PositionTracker


@dataclass
class Frame:
    """The latest captured frame and what was read from it."""

    result: ParseResult
    image: Optional[np.ndarray]     # BGR pixels (kept only while the debug preview is on)
    left: int                       # screen position of the captured area
    top: int
    scale: float                    # captured pixels per screen unit (2.0 on HiDPI macOS)
    number: int
    parse_ms: float
    width: int = 0                  # size of the captured image in pixels
    height: int = 0

    def to_screen(self, x: float, y: float) -> tuple[int, int]:
        """Image pixel -> screen coordinate (as window systems and Tk use them)."""
        return round(self.left + x / self.scale), round(self.top + y / self.scale)


Grabber = Callable[[], tuple[np.ndarray, int, int, float]]


def mss_grabber(monitor: int = 1) -> Grabber:
    """A function that captures ``monitor`` (1 = primary, 0 = all monitors together)."""
    import mss

    sct = (getattr(mss, "MSS", None) or mss.mss)()  # mss 10 renamed the factory
    if not 0 <= monitor < len(sct.monitors):
        raise ValueError(f"monitor {monitor} does not exist (found {len(sct.monitors) - 1})")
    area = sct.monitors[monitor]

    def grab() -> tuple[np.ndarray, int, int, float]:
        shot = sct.grab(area)
        frame = np.asarray(shot)[:, :, :3]  # BGRA -> BGR
        return np.ascontiguousarray(frame), area["left"], area["top"], shot.width / area["width"]

    return grab


class ScreenWatcher(threading.Thread):
    def __init__(self, parser: ScreenParser, tracker: PositionTracker,
                 on_position: Callable[[Position], None], monitor: int = 1, fps: float = 5.0,
                 grabber: Optional[Grabber] = None) -> None:
        super().__init__(name="assistant-capture", daemon=True)
        self.parser, self.tracker = parser, tracker
        self.on_position = on_position
        self.monitor = monitor
        self.interval = 1.0 / max(0.5, min(fps, 30.0))
        self.search_interval = max(self.interval, 0.6)  # full-screen search rate while no board is locked
        self._grabber = grabber
        self.keep_images = False
        self.ignore_rects: list[tuple[int, int, int, int]] = []  # screen rects of our own windows
        self.latest: Optional[Frame] = None
        self.error = ""
        self._stop_event = threading.Event()
        self._rescan = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def rescan(self) -> None:
        """Forget the board position and the game history, and search again."""
        self._rescan.set()

    def run(self) -> None:
        try:
            grab = self._grabber or mss_grabber(self.monitor)  # mss must be used in the thread that made it
        except Exception as exc:
            self.error = f"screen capture unavailable: {exc}"
            return
        number = 0
        while not self._stop_event.is_set():
            started = time.perf_counter()
            if self._rescan.is_set():
                self._rescan.clear()
                self.parser.reset()
                self.tracker.reset()
            try:
                image, left, top, scale = grab()
                for x1, y1, x2, y2 in list(self.ignore_rects):
                    px1, py1 = max(0, round((x1 - left) * scale)), max(0, round((y1 - top) * scale))
                    px2, py2 = round((x2 - left) * scale), round((y2 - top) * scale)
                    if px2 > px1 and py2 > py1:
                        image[py1:py2, px1:px2] = 0
                result = self.parser.parse(image)
                self.error = ""
            except Exception as exc:
                self.error = f"capture failed: {exc}"
                self._stop_event.wait(1.0)
                continue
            number += 1
            parse_ms = (time.perf_counter() - started) * 1000
            self.latest = Frame(result, image if self.keep_images else None, left, top, scale, number, parse_ms,
                                image.shape[1], image.shape[0])
            position = self.tracker.update(result)
            if position is not None:
                self.on_position(position)
            wait = self.interval if result.geometry is not None else self.search_interval
            self._stop_event.wait(max(0.01, wait - (time.perf_counter() - started)))
