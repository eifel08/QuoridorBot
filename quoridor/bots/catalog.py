"""The bots offered by every bot selector, strongest first, and their tunable settings.

The order comes from the round-robin tournaments in docs/bot-evolution-report.md
(5 s per move: hybrid +222, nnue +209, tuned +36, turbo -20 Elo against the old engine).
Hybrid and NNUE now use the generation-2 network (docs/nnue-gen2-report.md); with it NNUE
edges Hybrid (56-44 at 0.5 s, 9-7 at 5 s), which is within noise, so the order is unchanged.

Every bot is a :class:`~quoridor.bots.factory.BotSpec` of kind ``"profile"``; the GUI
menus and the Screen Assistant / Auto-Play panel build their controls from
:func:`params_for`, and change a spec with :func:`with_value`, so a setting means the same
thing everywhere (and in ``profile:...`` spec strings on the command line).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

from .alphabeta import MAX_DEPTH
from .cpp_engine import DEFAULT_OPTIONS, DEFAULT_WALL_FILTER
from .factory import BotSpec
from .profiles import PROFILES, profile_uses_book

#: (profile, title, one-line description), strongest first.
RANKED_BOTS: tuple[tuple[str, str, str], ...] = (
    ("hybrid", "Hybrid", "strongest: opening book, neural midgame, exact endgame"),
    ("nnue", "NNUE", "neural-network evaluation, fast search"),
    ("tuned", "Tuned", "Texel-tuned evaluation, fast search"),
    ("turbo", "Turbo", "fast search (PVS, LMR, smart time), classic evaluation"),
)
_TITLES = {name: t for name, t, _ in RANKED_BOTS} | {"book": "Book", "baseline": "Baseline"}


def title(profile: Optional[str]) -> str:
    return _TITLES.get(profile or "", str(profile))


def description(profile: str) -> str:
    return next((d for name, _, d in RANKED_BOTS if name == profile), "")


@dataclass(frozen=True)
class Param:
    """One setting of a bot.

    ``kind``: ``"range"`` (integers ``lo..hi``: a slider), ``"choice"`` (one of ``values``,
    shown as ``labels``: buttons or a list) or ``"toggle"`` (on / off).
    ``key``: ``depth``, ``time``, ``book``, ``walls`` (engine flag wall_filter) or an engine flag.
    """

    key: str
    label: str
    kind: str
    lo: int = 0
    hi: int = 0
    values: tuple = ()
    labels: tuple = ()
    hint: str = ""

    def text(self, value: Any) -> str:
        """``value`` as shown next to the control."""
        if self.kind == "toggle":
            return "on" if value else "off"
        if self.kind == "choice":
            return self.labels[self.values.index(value)] if value in self.values else str(value)
        if self.key == "endgame_walls" and value < 0:
            return "off"
        return str(value)


TIME_VALUES: tuple[Optional[float], ...] = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, None)
DEPTH = Param("depth", "Search depth", "range", 1, MAX_DEPTH,
              hint="maximum plies; the think time usually stops the search first")
TIME = Param("time", "Think time", "choice", values=TIME_VALUES,
             labels=tuple("no limit" if t is None else f"{t:g} s" for t in TIME_VALUES),
             hint="seconds per move")
WALLS = Param("walls", "Walls searched", "choice", values=(2, 1, 0), labels=("Attack", "Att + def", "All"),
              hint="attack = only walls on the opponent's paths (deepest); all = every legal wall")
BOOK = Param("book", "Opening book", "toggle", hint="play stored opening moves instantly")
LMR = Param("lmr", "Late-move reductions", "toggle", hint="search late, quiet moves shallower (deeper overall)")
FUTILITY = Param("futility", "Futility pruning", "toggle", hint="skip hopeless moves near the leaves")
HASH = Param("hash_mb", "Hash table", "choice", values=(32, 64, 128, 256), labels=("32 MB", "64 MB", "128 MB", "256 MB"),
             hint="transposition table size")
HYBRID_WALLS = Param("hybrid_walls", "Neural eval from", "range", 0, 20,
                     hint="walls left in total at which the neural eval is used (below: tuned eval)")
ENDGAME_WALLS = Param("endgame_walls", "Exact endgame at", "range", -1, 20,
                      hint="walls left in total at or below which pruning is switched off (-1 = never)")

_COMMON = (DEPTH, TIME, WALLS, BOOK, LMR, FUTILITY, HASH)
PARAMS: dict[str, tuple[Param, ...]] = {
    "hybrid": (*_COMMON, HYBRID_WALLS, ENDGAME_WALLS),
    "nnue": _COMMON,
    "tuned": _COMMON,
    "turbo": _COMMON,
    "book": _COMMON,
    "baseline": (DEPTH, TIME, WALLS, FUTILITY, HASH),
}
ENGINE = Param("engine", "Engine", "choice", values=("auto", "cpp", "py"), labels=("Auto", "C++", "Python"),
               hint="search engine of a plain alpha-beta bot (auto = C++ if built)")
#: Plain alpha-beta specs (from the command line or a custom spec).
AB_PARAMS: tuple[Param, ...] = (DEPTH, TIME, WALLS, ENGINE)


def params_for(spec: BotSpec) -> tuple[Param, ...]:
    if spec.kind == "profile":
        return PARAMS.get(spec.model or "", _COMMON)
    if spec.kind == "ab":
        return AB_PARAMS
    return ()


def _option_key(param: Param) -> str:
    return "wall_filter" if param.key == "walls" else param.key


def _default_option(spec: BotSpec, key: str) -> int:
    if spec.kind == "profile":
        return int(PROFILES[spec.model]["options"].get(key, DEFAULT_OPTIONS[key]))  # type: ignore[index]
    return DEFAULT_WALL_FILTER if key == "wall_filter" else DEFAULT_OPTIONS[key]


def value_of(spec: BotSpec, param: Param) -> Any:
    if param.key == "depth":
        return spec.depth
    if param.key == "time":
        return spec.time_limit
    if param.key == "engine":
        return spec.engine
    if param.key == "book":
        return profile_uses_book(spec.model) if spec.book is None else spec.book  # type: ignore[arg-type]
    key = _option_key(param)
    value = dict(spec.options).get(key, _default_option(spec, key))
    return bool(value) if param.kind == "toggle" else value


def with_value(spec: BotSpec, param: Param, value: Any) -> BotSpec:
    """``spec`` with one setting changed; a value equal to the bot's default is dropped from the spec."""
    if param.key == "depth":
        return replace(spec, depth=max(1, min(MAX_DEPTH, int(value))))
    if param.key == "time":
        return replace(spec, time_limit=None if value is None else float(value))
    if param.key == "engine":
        # engine flags only exist in the C++ engine
        return replace(spec, engine=value, options=() if value == "py" else spec.options)
    if param.key == "book":
        on = bool(value)
        return replace(spec, book=None if on == profile_uses_book(spec.model) else on)  # type: ignore[arg-type]
    key = _option_key(param)
    value = int(value)
    if param.kind == "range":
        value = max(param.lo, min(param.hi, value))
    return spec.with_option(key, None if value == _default_option(spec, key) else value)


def default_spec(profile: str = "hybrid", time_limit: Optional[float] = 5.0) -> BotSpec:
    return BotSpec("profile", depth=MAX_DEPTH, time_limit=time_limit, model=profile)


def bot_choices(current: Optional[BotSpec] = None, time_limit: Optional[float] = 5.0) -> list[tuple[str, BotSpec]]:
    """(label, spec) for every ranked bot, strongest first; ``current`` is kept for its own bot
    (with its settings), and appended if it is not one of the ranked bots."""
    out = []
    for i, (name, t, desc) in enumerate(RANKED_BOTS, 1):
        spec = default_spec(name, time_limit)
        if current is not None and current.kind == "profile" and current.model == name:
            spec = current
        out.append((f"{i}. {t} - {desc}", spec))
    if current is not None and not (current.kind == "profile" and current.model in dict(
            (n, 1) for n, _, _ in RANKED_BOTS)):
        out.append((f"Other: {current.label()}", current))
    return out


def choice_index(spec: BotSpec, choices: list[tuple[str, BotSpec]]) -> int:
    """Which entry of :func:`bot_choices` ``spec`` belongs to."""
    for i, (_, s) in enumerate(choices):
        if s.kind == "profile" and spec.kind == "profile" and s.model == spec.model:
            return i
        if s == spec:
            return i
    return 0
