"""Screen Assistant: read a Quoridor game from the screen and analyse it live.

``parser``    screenshot -> pawns, walls, walls left, side to move
``position``  stable positions from a stream of frames, move notation
``analysis``  best move, eval and alternatives from any configured bot
``capture``   background screen-capture loop (mss)
``app``       always-on-top panel, on-board move marker, debug preview (Tk)
``autoplay``  Auto-Play: the bot enters the bottom player's moves with the mouse

Needs ``mss``, ``opencv-python`` and ``numpy`` (``requirements-assistant.txt``).
"""

from __future__ import annotations

from typing import Optional

def missing_hint(exc: ImportError) -> str:
    """What to install when an import of the assistant failed."""
    if "pyautogui" in str(exc):
        return "Auto-Play needs the mouse library: pip install -r requirements-autoplay.txt"
    if "tkinter" in str(exc):
        return ("Tk is missing from this Python. Windows/macOS: reinstall Python from python.org with "
                "'tcl/tk' ticked; Linux: sudo apt install python3-tk")
    return "Install the Screen Assistant's dependencies with: pip install -r requirements-assistant.txt"


def run_assistant(spec=None, monitor: int = 1, fps: float = 5.0, profile_path: Optional[str] = None,
                  debug: bool = False, overlay: bool = True, demo: Optional[str] = None) -> None:
    """Open the Screen Assistant and run until its panel is closed.

    ``demo`` ("classic" or "timed") also opens a self-playing mock game in that layout for the
    assistant to watch.
    """
    from .analysis import DEFAULT_SPEC
    from .app import AssistantApp
    from .parser import Profile

    profile = Profile.load(profile_path) if profile_path else None
    app = AssistantApp(spec or DEFAULT_SPEC, monitor, fps, profile, debug, overlay)
    if demo:
        from .demo import DemoGame

        app.demo = DemoGame(app.root, layout=demo)
    app.run()


def run_autoplay(spec=None, monitor: int = 1, fps: float = 5.0, profile_path: Optional[str] = None,
                 debug: bool = False, click_profile_path: Optional[str] = None, join_in_progress: bool = False,
                 demo: Optional[str] = None, armed: bool = False, continuous: bool = False) -> None:
    """Open Auto-Play (the bot plays the bottom player with the mouse) and run until its panel is closed.

    It starts switched off; nothing is clicked before *Start auto-play* is pressed
    (``armed=True`` presses it at launch).
    ``continuous``: Continuous Play on from the start (keep playing game after game).
    ``demo``: "red" or "blue" also opens a clickable mock game with the bot on that colour.
    """
    from .analysis import DEFAULT_SPEC
    from .autoplay import ClickProfile
    from .autoplay_app import AutoPlayApp
    from .parser import Profile

    profile = Profile.load(profile_path) if profile_path else None
    clicks = ClickProfile.load(click_profile_path) if click_profile_path else None
    app = AutoPlayApp(spec or DEFAULT_SPEC, monitor, fps, profile, debug, clicks, join_in_progress,
                      continuous=continuous)
    if demo:
        from .demo import PlayableDemo

        app.demo = PlayableDemo(app.root, flipped=demo == "blue")
    if armed:
        app.player.arm()
    app.run()


def check_image(path: str, out_path: Optional[str] = None, profile_path: Optional[str] = None) -> bool:
    """Parse a saved screenshot, print what was detected and save an annotated copy.

    For calibrating without a live game. Returns True if the position was readable.
    """
    import cv2

    from .parser import PLAYER_NAMES, Profile, ScreenParser, annotate
    from .position import Position

    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"cannot read image '{path}'")
    res = ScreenParser(Profile.load(profile_path) if profile_path else None).parse(image)
    if res.geometry is None:
        print("No 9x9 board found.")
    else:
        x1, y1, x2, y2 = res.geometry.board_rect()
        print(f"Board at x {x1}-{x2}, y {y1}-{y2} (squares {res.geometry.side:.0f} px)")
    if res.ok:
        pos = Position(res.pawns, res.walls, res.walls_left, res.turn, bool(res.flipped))
        print(pos.summary())
        side = "no tinted edge rows seen" if res.flipped is None else (
            "Red starts at the top" if res.flipped else "Red starts at the bottom")
        print(f"Orientation: {side}")
        print(f"Walls left read from: {res.walls_source}; turn from: "
              + (res.turn_source or f"move parity (a guess: {PLAYER_NAMES[pos.parity_turn()]})"))
        if res.move_list_turn is not None:
            print(f"Move list: last row '{res.move_list_last}'")
        for p in (1, 0):
            card = res.cards[p]
            print(f"  {PLAYER_NAMES[p]} card: {'found' if card.dot else 'NOT found'}, ticks {card.tick_count}, "
                  f"text {card.ocr_count if card.ocr_count is not None else '-'}")
        if pos.error():
            print("Not a legal position: " + pos.error())
    elif res.geometry is not None:
        print("Not readable: " + res.reason)
    for warning in res.warnings:
        print("Warning: " + warning)
    out_path = out_path or path.rsplit(".", 1)[0] + ".parsed.png"
    cv2.imwrite(out_path, annotate(image, res))
    print(f"Annotated copy: {out_path}")
    return res.ok
