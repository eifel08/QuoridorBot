"""Named bot profiles for the bot-evolution tournament (docs/bot-evolution-plan.md).

Every profile is the C++ engine with a set of options, optionally tuned linear
weights, a trained network and an opening book. ``ProfileEngine`` wraps one
engine process for one profile; ``ProfileBot`` is the same thing as a
:class:`~quoridor.player.Player` for the GUI / CLI (bot spec ``profile:NAME``).

    baseline       the engine of main before this work (frozen source, engine/baseline/)
    turbo          + smart time management, PVS, aspiration windows, LMR, repetition, bigger TT
    tuned          turbo + Texel-tuned phase-tapered linear eval with path-shape features
    book           tuned + opening book
    nnue           turbo search + trained neural evaluation
    hybrid         book -> neural midgame -> tuned linear eval + exact deep endgame search
"""

from __future__ import annotations

import json
import random
import threading
from pathlib import Path
from typing import Mapping, Optional

from .alphabeta import FastBoard
from .cpp_engine import ENGINE_DIR, CppEngine, EngineError, fastboard_command, parse_move, to_imove

BUILD_DIR = ENGINE_DIR / "build"
NETS_DIR = ENGINE_DIR / "nets"
BOOK_PATH = ENGINE_DIR / "book" / "book.json"
LINEAR_WEIGHTS = NETS_DIR / "linear_weights.txt"
NET_PATH = NETS_DIR / "nnue.txt"

TURBO = {"smart_time": 1, "pvs": 1, "aspiration": 15, "lmr": 1, "repetition": 1, "hash_mb": 128}

PROFILES: dict[str, dict] = {
    "baseline": {"binary": "quoridor_engine_baseline", "options": {"hash_mb": 128}},
    "turbo": {"options": dict(TURBO)},
    "tuned": {"options": {**TURBO, "eval_mode": 1}, "linear": True},
    "book": {"options": {**TURBO, "eval_mode": 1}, "linear": True, "book": True},
    "nnue": {"options": {**TURBO, "eval_mode": 2}, "linear": True, "net": True},
    "hybrid": {"options": {**TURBO, "eval_mode": 3, "hybrid_walls": 6, "endgame_walls": 4},
               "linear": True, "net": True, "book": True},
}


def engine_binary(name: str) -> Path:
    import os
    exe = name + (".exe" if os.name == "nt" else "")
    return BUILD_DIR / exe


_REBUILD_LOCK = threading.Lock()
_rebuilt: dict[str, str] = {}


def rebuild_engine(stale: Path, reason: Exception) -> str:
    """Recompile the engine (``python build_engine.py``) because ``stale`` is out of date."""
    from .cpp_engine import build

    with _REBUILD_LOCK:  # the GUI and the analyser may both get here
        if str(stale) in _rebuilt:
            return _rebuilt[str(stale)]
        print(f"The C++ engine {stale} is older than this code ({reason}); rebuilding it...", flush=True)
        try:
            path = build(verbose=False)
        except EngineError as exc:
            raise EngineError(
                f"The C++ engine {stale} is out of date ({reason}) and could not be rebuilt "
                f"automatically. Rebuild it with:  python build_engine.py   (details: {exc})") from None
        print(f"Rebuilt the C++ engine: {path}", flush=True)
        _rebuilt[str(stale)] = path
        return path


def _fallback_binary(missing: Path) -> Path:
    """The main engine built by ``python build_engine.py`` (any build folder) if ``missing`` is it."""
    from .cpp_engine import EXE_NAME, find_engine

    if missing.name == EXE_NAME:
        found = find_engine()
        if found:
            return Path(found)
    raise EngineError(f"{missing} not built (python build_engine.py, or make -C engine && make -C engine baseline)")


# ---------------------------------------------------------------------- #
# Symmetry: book positions are stored canonically (side to move = player 0,
# then the smaller of the position and its left-right mirror).
# ---------------------------------------------------------------------- #
N, NW = 9, 8


def _sq(sq: int, flip: bool, mirror: bool) -> int:
    r, c = divmod(sq, N)
    return (N - 1 - r if flip else r) * N + (N - 1 - c if mirror else c)


def _an(a: int, flip: bool, mirror: bool) -> int:
    r, c = divmod(a, NW)
    return (NW - 1 - r if flip else r) * NW + (NW - 1 - c if mirror else c)


def _key(board: FastBoard, flip: bool, mirror: bool) -> str:
    me, opp = (1, 0) if flip else (0, 1)
    h = sorted(_an(a, flip, mirror) for a in board.h)
    v = sorted(_an(a, flip, mirror) for a in board.v)
    return (f"{_sq(board.pawns[me], flip, mirror)},{_sq(board.pawns[opp], flip, mirror)},"
            f"{board.walls_left[me]},{board.walls_left[opp]}|" + ".".join(map(str, h)) + "|" + ".".join(map(str, v)))


def canonical(board: FastBoard) -> tuple[str, bool, bool]:
    """(key, flip, mirror): the transform that maps the real board to the stored one."""
    flip = board.turn == 1
    k0, k1 = _key(board, flip, False), _key(board, flip, True)
    return (k0, flip, False) if k0 <= k1 else (k1, flip, True)


def transform_move(text: str, flip: bool, mirror: bool) -> str:
    """Engine move text through a transform (each transform is its own inverse)."""
    kind, r, c = text[0], int(text[1]), int(text[2])
    if kind == "m":
        sq = _sq(r * N + c, flip, mirror)
        return f"m{sq // N}{sq % N}"
    a = _an(r * NW + c, flip, mirror)
    return f"{kind}{a // NW}{a % NW}"


class OpeningBook:
    """canonical key -> list of [move, weight] (moves in canonical coordinates)."""

    def __init__(self, path: Path = BOOK_PATH) -> None:
        self.path = Path(path)
        self.entries: dict[str, list] = {}
        if self.path.is_file():
            data = json.loads(self.path.read_text())
            self.entries = data.get("positions", data)

    def __len__(self) -> int:
        return len(self.entries)

    def probe(self, board: FastBoard, rng: Optional[random.Random] = None) -> Optional[str]:
        key, flip, mirror = canonical(board)
        moves = self.entries.get(key)
        if not moves:
            return None
        if rng is None or len(moves) == 1:
            move = max(moves, key=lambda m: m[1])[0]
        else:
            move = rng.choices([m[0] for m in moves], weights=[m[1] for m in moves])[0]
        return transform_move(move, flip, mirror)


_BOOK: Optional[OpeningBook] = None


def shared_book() -> OpeningBook:
    global _BOOK
    if _BOOK is None:
        _BOOK = OpeningBook()
    return _BOOK


def profile_options(name: str, overrides: Optional[Mapping[str, int]] = None) -> dict[str, int]:
    """The engine options of a profile, with ``overrides`` (e.g. from the GUI) on top."""
    return {**PROFILES[name]["options"], **dict(overrides or {})}


def profile_uses_book(name: str) -> bool:
    return bool(PROFILES[name].get("book"))


class ProfileEngine:
    """One engine process configured as a profile. ``go(board, time)`` -> (move text, info).

    ``options`` override the profile's engine options; ``book`` turns the opening book on or
    off (None = the profile's default).
    """

    def __init__(self, name: str, seed: int = 0, options: Optional[Mapping[str, int]] = None,
                 book: Optional[bool] = None) -> None:
        if name not in PROFILES:
            raise EngineError(f"unknown profile '{name}' (known: {', '.join(PROFILES)})")
        self.name = name
        prof = PROFILES[name]
        binary = engine_binary(prof.get("binary", "quoridor_engine"))
        if not binary.is_file():
            binary = _fallback_binary(binary)
        try:
            self.engine = CppEngine(str(binary), profile_options(name, options))
        except EngineError as exc:
            if "unknown option" not in str(exc) or prof.get("binary"):
                raise
            # An executable built before the profile options existed (e.g. after a git pull):
            # rebuild it once from the current sources and try again.
            binary = Path(rebuild_engine(binary, exc))
            self.engine = CppEngine(str(binary), profile_options(name, options))
        if prof.get("linear"):
            if not LINEAR_WEIGHTS.is_file():
                raise EngineError(f"{LINEAR_WEIGHTS} missing (python -m tools.texel_tune)")
            self.engine.command("linweights " + " ".join(LINEAR_WEIGHTS.read_text().split()))
        if prof.get("net"):
            if not NET_PATH.is_file():
                raise EngineError(f"{NET_PATH} missing (python -m tools.train_nnue)")
            self.engine.command(f"loadnet {NET_PATH}")
        use_book = prof.get("book", False) if book is None else book
        self.book = shared_book() if use_book else None
        self.rng = random.Random(seed) if seed else None

    def new_game(self) -> None:
        self.engine.new_game()

    def go(self, board: FastBoard, time_limit: float, depth: int = 64) -> tuple[Optional[str], dict]:
        if self.book is not None:
            move = self.book.probe(board, self.rng)
            if move is not None:
                return move, {"book": True, "depth_reached": 0, "nodes": 0, "seconds": 0.0, "value": 0}
        self.engine.set_position(fastboard_command(board))
        move, info = self.engine.go(depth, time_limit)
        info["book"] = False
        return (info["move"] if move is not None else None), info

    def close(self) -> None:
        self.engine.close()


def imove(text: str):
    return to_imove(parse_move(text))


# ---------------------------------------------------------------------- #
# Player for the GUI / CLI / Auto-Play (bot spec "profile:name=hybrid,time=5")
# ---------------------------------------------------------------------- #
from .cpp_engine import CppBot  # noqa: E402


class ProfileBot(CppBot):
    """A profile as a :class:`~quoridor.player.Player` (book move if any, else the engine)."""

    def __init__(self, name: str, profile: str, time_limit: Optional[float] = 5.0, depth: int = 30,
                 options: Optional[Mapping[str, int]] = None, book: Optional[bool] = None) -> None:
        if profile not in PROFILES:
            raise EngineError(f"unknown profile '{profile}' (known: {', '.join(PROFILES)})")
        super().__init__(name, depth, time_limit, profile_options(profile, options))
        self.profile = profile
        self.overrides = dict(options or {})
        self.book_on = book
        self._pe: Optional[ProfileEngine] = None

    @property
    def engine(self) -> CppEngine:
        if self._pe is None:
            self._pe = ProfileEngine(self.profile, seed=random.getrandbits(31) | 1, options=self.overrides,
                                     book=self.book_on)
            self._engine = self._pe.engine
        return self._pe.engine

    def get_move(self, board_state):
        _ = self.engine
        if self._pe.book is not None:
            text = self._pe.book.probe(FastBoard(board_state), self._pe.rng)
            if text is not None:
                self.last_info = {"engine": "book", "book": True, "depth_reached": 0, "nodes": 0, "seconds": 0.0,
                                  "nps": 0, "value": 0, "forced": None, "plies": None, "move": text,
                                  "score": None, "eval_steps": None, "depth_requested": self.depth}
                return parse_move(text)
        return super().get_move(board_state)

    def close(self) -> None:
        super().close()
        self._pe = None
