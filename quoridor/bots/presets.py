"""Named bot configurations for pickers (Auto-Play's bot selector).

Each preset is a label plus a bot spec string (see :mod:`quoridor.bots.factory`),
so anything a preset can do, ``--autoplay SPEC`` / ``--bot1 SPEC`` can do too.
Trained ML checkpoints found on disk are offered as extra presets.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

from .factory import BotSpec, parse_bot_spec

#: (label, spec). The first one is the default of the Screen Assistant and Auto-Play.
BOT_PRESETS: tuple[tuple[str, str], ...] = (
    ("Standard: C++ alpha-beta, 1 s", "ab:depth=30,time=1"),
    ("Strong: C++ alpha-beta, 3 s", "ab:depth=30,time=3"),
    ("Fast: C++ alpha-beta, 0.3 s", "ab:depth=30,time=0.3"),
    ("Attack walls only: C++, 1 s", "ab:depth=30,time=1,walls=attack"),
    ("All walls: C++, 2 s", "ab:depth=30,time=2,walls=all"),
    ("Classic eval (flat wall value): C++, 1 s", "ab:depth=30,time=1,smart_walls=0,def_global=0,lock_decided=0"),
    ("Python alpha-beta, depth 4", "ab:depth=4,engine=py"),
    ("Hybrid 3-phase (book + NNUE + endgame), 5 s", "profile:name=hybrid,time=5"),
    ("Book + tuned eval + turbo search, 5 s", "profile:name=book,time=5"),
    ("NNUE eval + turbo search, 5 s", "profile:name=nnue,time=5"),
    ("Turbo search (classic eval), 5 s", "profile:name=turbo,time=5"),
)

CUSTOM_LABEL = "Custom spec..."

#: Where trained checkpoints usually live (``python -m quoridor.ml.train --out runs/NAME``).
ML_SEARCH_DIRS = ("runs", "checkpoints", "models", ".")


def find_checkpoints(dirs: Iterable[str] = ML_SEARCH_DIRS, root: Optional[Path] = None, limit: int = 6) -> list[Path]:
    """``best.pt`` / ``*.pt`` files in the usual places (newest first, at most ``limit``)."""
    root = root or Path.cwd()
    found: dict[Path, float] = {}
    for d in dirs:
        base = root / d
        if not base.is_dir():
            continue
        pattern = "*.pt" if d == "." else "**/*.pt"
        try:
            for p in base.glob(pattern):
                if p.is_file():
                    found[p.resolve()] = p.stat().st_mtime
        except OSError:
            continue
    return sorted(found, key=lambda p: -found[p])[:limit]


def ml_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def preset_choices(current: Optional[BotSpec] = None, include_ml: bool = True) -> list[tuple[str, BotSpec]]:
    """Every selectable bot as (label, spec), with ``current`` first if it is not a preset."""
    choices = [(label, parse_bot_spec(spec)) for label, spec in BOT_PRESETS]
    if include_ml and ml_available():
        for path in find_checkpoints():
            try:
                rel = os.path.relpath(path)
            except ValueError:  # another drive on Windows
                rel = str(path)
            choices.append((f"ML agent: {rel} (64 sims)", BotSpec("ml", model=str(path), sims=64)))
    if current is not None and all(spec != current for _, spec in choices):
        choices.insert(0, (f"From command line: {current.label()}", current))
    return choices


def label_for(spec: BotSpec, choices: list[tuple[str, BotSpec]]) -> str:
    return next((label for label, s in choices if s == spec), f"Custom: {spec.label()}")
