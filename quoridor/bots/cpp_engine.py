"""Bridge to the C++ search engine (``engine/``).

The engine is a separate executable that speaks a small line-based text
protocol on stdin/stdout (see ``engine/src/main.cpp``). This module starts
it, translates positions and moves, and wraps it as a normal
:class:`~quoridor.player.Player`, so the GUI and CLI use it exactly like the
pure-Python :class:`~quoridor.bots.alphabeta.AlphaBetaBot`.

Build the engine first (see ``engine/README.md``)::

    python -m quoridor.bots.cpp_engine build

The executable is looked up in ``$QUORIDOR_ENGINE`` and then in
``engine/build``. If it is missing, :func:`make_search_bot` falls back to the
Python bot.
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import sys
import threading
import weakref
from pathlib import Path
from typing import Iterable, Mapping, Optional, Union

from ..moves import Move, Orientation, PawnMove, WallMove
from ..state import BoardState
from .alphabeta import MAX_DEPTH, N, NW, AlphaBetaBot, FastBoard

ENGINE_DIR = Path(__file__).resolve().parent.parent.parent / "engine"
EXE_NAME = "quoridor_engine.exe" if sys.platform == "win32" else "quoridor_engine"
WIN = 30000            # must match engine/src/search.h
WIN_THRESHOLD = WIN - 1000

IMove = tuple[str, int]
OptionValue = Union[int, bool]

#: Engine options and their defaults (``set <name> <value>``); see engine/README.md.
DEFAULT_OPTIONS: dict[str, int] = {
    "dist_weight": 10, "wall_weight": 16, "tempo_bonus": 5,
    "tt": 1, "tt_exact_depth": 0, "alphabeta": 1, "killers": 1, "history": 1,
    "wall_filter": 1, "decided": 1, "decided_margin": 1,
    "futility": 1, "wall_recovery": 5, "depth_margin": 5,
    "seed": 0, "hash_mb": 64,
}


class EngineError(RuntimeError):
    pass


def find_engine() -> Optional[str]:
    """Path of the engine executable, or ``None`` if it has not been built."""
    env = os.environ.get("QUORIDOR_ENGINE")
    if env:
        return env if os.path.isfile(env) else None
    for sub in ("build", "build/Release", "build/RelWithDebInfo", "build/Debug"):
        path = ENGINE_DIR / sub / EXE_NAME
        if path.is_file():
            return str(path)
    return None


def engine_available() -> bool:
    return find_engine() is not None


#: Which search "auto" bots use: "auto" (C++ if built, else Python), "cpp" or "py".
#: Set with ``main.py --engine`` or the QUORIDOR_BOT_ENGINE environment variable.
default_engine = os.environ.get("QUORIDOR_BOT_ENGINE", "auto").lower()


def resolve_engine(engine: str = "auto") -> str:
    """Turn "auto" into "cpp" or "py"."""
    if engine == "auto":
        engine = default_engine
    if engine == "auto":
        engine = "cpp" if engine_available() else "py"
    return engine


# ---------------------------------------------------------------------- #
# Text encodings
# ---------------------------------------------------------------------- #
def position_command(turn: int, pawns, walls_left, walls: Iterable[tuple[int, int, str]]) -> str:
    """``position`` line; ``pawns`` are (row, col) pairs, walls (row, col, "h"|"v")."""
    (r0, c0), (r1, c1) = pawns
    parts = ["position", turn, r0, c0, r1, c1, walls_left[0], walls_left[1]]
    parts += [f"{o}{r}{c}" for r, c, o in walls]
    return " ".join(map(str, parts))


def state_command(state: BoardState) -> str:
    return position_command(state.current_player, state.pawns, state.walls_left,
                            ((r, c, Orientation(o).value) for r, c, o in state.walls))


def fastboard_command(board: FastBoard) -> str:
    walls = [(*divmod(a, NW), "h") for a in sorted(board.h)] + [(*divmod(a, NW), "v") for a in sorted(board.v)]
    return position_command(board.turn, [divmod(p, N) for p in board.pawns], board.walls_left, walls)


def parse_move(text: str) -> Optional[Move]:
    """``m74`` / ``h34`` / ``v07`` -> :class:`PawnMove` / :class:`WallMove` (``none`` -> None)."""
    if text == "none":
        return None
    kind, r, c = text[0], int(text[1]), int(text[2])
    return PawnMove(r, c) if kind == "m" else WallMove(r, c, Orientation(kind))


def format_move(move: Move) -> str:
    if isinstance(move, PawnMove):
        return f"m{move.row}{move.col}"
    return f"{Orientation(move.orientation).value}{move.row}{move.col}"


def to_imove(move: Move) -> IMove:
    """Internal move of the Python bot / ML code: ("p", square) or ("h"|"v", anchor)."""
    if isinstance(move, PawnMove):
        return ("p", move.row * N + move.col)
    return (Orientation(move.orientation).value, move.row * NW + move.col)


# ---------------------------------------------------------------------- #
# Engine process
# ---------------------------------------------------------------------- #
_running: "weakref.WeakSet[CppEngine]" = weakref.WeakSet()


@atexit.register
def _stop_all() -> None:
    """Never leave an engine process behind (e.g. window closed mid-search)."""
    for engine in list(_running):
        engine.close()


class CppEngine:
    """One running engine process. Safe to use from several threads (one command at a time)."""

    def __init__(self, path: Optional[str] = None, options: Optional[Mapping[str, OptionValue]] = None) -> None:
        path = path or find_engine()
        if path is None:
            raise EngineError("C++ engine not built. Run: python -m quoridor.bots.cpp_engine build")
        self.path = path
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        _running.add(self)
        self.info_lines: list[str] = []
        for name, value in (options or {}).items():
            self.set_option(name, value)

    # -- low level -------------------------------------------------------
    def command(self, line: str) -> str:
        """Send one command and return its final answer line."""
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                raise EngineError("engine process is not running")
            assert proc.stdin is not None and proc.stdout is not None
            try:
                proc.stdin.write(line + "\n")
                proc.stdin.flush()
                self.info_lines = []
                while True:
                    answer = proc.stdout.readline()
                    if not answer:
                        raise EngineError(f"engine exited while handling '{line}'")
                    answer = answer.rstrip("\n")
                    if not answer.startswith("info "):
                        break
                    self.info_lines.append(answer)
            except (BrokenPipeError, OSError) as exc:
                raise EngineError(f"lost the engine process: {exc}") from None
        if answer.startswith("error"):
            raise EngineError(f"{answer[6:]} (command: {line})")
        return answer

    def close(self) -> None:
        proc, self._proc = self._proc, None
        _running.discard(self)
        if proc is None:
            return
        try:
            proc.kill()  # also stops a search that is still running
            proc.wait(timeout=2)
        except Exception:
            pass
        for stream in (proc.stdin, proc.stdout):
            try:
                if stream:
                    stream.close()
            except Exception:
                pass

    def __enter__(self) -> "CppEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- commands --------------------------------------------------------
    def set_option(self, name: str, value: OptionValue) -> None:
        self.command(f"set {name} {int(value)}")

    def new_game(self) -> None:
        self.command("newgame")

    def set_position(self, position: Union[BoardState, FastBoard, str]) -> None:
        if isinstance(position, BoardState):
            position = state_command(position)
        elif isinstance(position, FastBoard):
            position = fastboard_command(position)
        self.command(position)

    def play(self, move: Move) -> None:
        self.command("play " + format_move(move))

    def legal_moves(self) -> list[Move]:
        """Every legal move of the side to move (no wall filtering)."""
        return [parse_move(t) for t in self.command("moves").split()[1:]]  # type: ignore[misc]

    def search_order(self) -> list[Move]:
        """The moves the search would try, in order (wall filter applied)."""
        return [parse_move(t) for t in self.command("genmoves").split()[1:]]  # type: ignore[misc]

    def distances(self) -> tuple[int, int]:
        _, d0, d1 = self.command("dist").split()
        return int(d0), int(d1)

    def evaluate(self) -> int:
        return int(self.command("eval").split()[1])

    def go(self, depth: Optional[int] = None, time_limit: Optional[float] = None) -> tuple[Optional[Move], dict]:
        """Search the current position. Returns (move, info); move is None if there is no legal move."""
        cmd = "go"
        if depth:
            cmd += f" depth {int(depth)}"
        if time_limit:
            cmd += f" time {time_limit:.4f}"
        tok = self.command(cmd).split()
        if tok[0] != "bestmove":
            raise EngineError(f"unexpected answer: {' '.join(tok)}")
        raw = dict(zip(tok[2::2], tok[3::2]))
        value, nodes, seconds = int(raw["score"]), int(raw["nodes"]), float(raw["time"])
        decided = abs(value) >= WIN_THRESHOLD
        info = {
            "engine": "cpp",
            "depth_requested": depth or 64,
            "depth_reached": int(raw["depth"]),
            "nodes": nodes,
            "seconds": seconds,
            "nps": int(nodes / seconds) if seconds > 0 else 0,
            "value": value,  # engine units, for the side to move (higher is better)
            "forced": None if not decided else ("win" if value > 0 else "loss"),
            "move": tok[1],
        }
        return parse_move(tok[1]), info


# ---------------------------------------------------------------------- #
# Player
# ---------------------------------------------------------------------- #
class CppBot(AlphaBetaBot):
    """The alpha-beta bot, searched by the C++ engine.

    Drop-in replacement for :class:`AlphaBetaBot` (same constructor arguments,
    ``depth`` / ``time_limit`` / ``last_info`` attributes). ``options`` are
    engine flags such as ``{"futility": 0, "wall_filter": 0}``.

    The engine process is started on the first move and keeps its
    transposition table for the rest of the game; it is told the complete
    position on every move, so the bot still does not rely on any memory of
    the game.
    """

    def __init__(self, name: str = "Bot", depth: int = 4, time_limit: Optional[float] = 5.0,
                 options: Optional[Mapping[str, OptionValue]] = None, engine_path: Optional[str] = None) -> None:
        super().__init__(name, depth, time_limit)
        self.options = dict(options or {})
        self.engine_path = engine_path
        self._engine: Optional[CppEngine] = None

    @property
    def engine(self) -> CppEngine:
        if self._engine is None:
            self._engine = CppEngine(self.engine_path, self.options)
            if self.rng is not None and "seed" not in self.options:
                self._engine.set_option("seed", self.rng.getrandbits(31) | 1)
        return self._engine

    def get_move(self, board_state: BoardState) -> Move:
        engine = self.engine
        if board_state.move_number <= 1:
            engine.new_game()
        engine.set_position(board_state)
        move, info = engine.go(self.depth, self.time_limit)
        if move is None:
            raise EngineError("the engine found no legal move in this position")
        info["depth_requested"] = self.depth
        if not info["forced"]:
            # Same convention as the Python bot: own steps - opponent steps, lower is better.
            scale = int(self.options.get("dist_weight", DEFAULT_OPTIONS["dist_weight"])) or 1
            info["score"] = -round(info["value"] / scale)
        else:
            info["score"] = None
        self.last_info = info
        return move

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None

    def __del__(self) -> None:  # e.g. the GUI started a new game with new players
        try:
            self.close()
        except Exception:
            pass


def make_search_bot(name: str = "Bot", depth: int = 4, time_limit: Optional[float] = 5.0,
                    options: Optional[Mapping[str, OptionValue]] = None, engine: str = "auto") -> AlphaBetaBot:
    """The alpha-beta bot on the fastest available engine.

    ``engine``: ``"cpp"`` (error if not built), ``"py"`` (pure Python), or
    ``"auto"`` (C++ if it has been built, otherwise Python).
    """
    if resolve_engine(engine) == "cpp":
        return CppBot(name, min(depth, MAX_DEPTH), time_limit, options)
    return AlphaBetaBot(name, depth, time_limit)


# ---------------------------------------------------------------------- #
# Building
# ---------------------------------------------------------------------- #
def build(verbose: bool = True) -> str:
    """Compile the engine with whatever toolchain is installed; returns the executable path."""
    out_dir = ENGINE_DIR / "build"
    out_dir.mkdir(exist_ok=True)
    sources = [str(ENGINE_DIR / "src" / f) for f in ("board.cpp", "search.cpp", "main.cpp")]
    target = out_dir / EXE_NAME
    attempts: list[list[str]] = []
    for cxx in filter(None, [os.environ.get("CXX"), "g++", "clang++", "c++"]):
        if shutil.which(cxx):
            attempts.append([cxx, "-O3", "-std=c++20", "-DNDEBUG", *sources, "-o", str(target)])
    if shutil.which("cl"):
        attempts.append(["cl", "/nologo", "/O2", "/std:c++20", "/EHsc", "/DNDEBUG", *sources,
                         f"/Fe:{target}", f"/Fo:{out_dir}{os.sep}"])
    if not attempts:
        raise EngineError("no C++ compiler found (install g++, clang++ or MSVC, or use CMake: see engine/README.md)")
    last = ""
    for cmd in attempts:
        if verbose:
            print(" ".join(cmd))
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode == 0 and target.is_file():
            return str(target)
        last = done.stdout + done.stderr
    raise EngineError("could not build the engine:\n" + last)


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        print("built", build())
    else:
        path = find_engine()
        print(path or "engine not built; run: python -m quoridor.bots.cpp_engine build")
