"""Tiny tweening helpers for GUI animations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


def ease_out_cubic(t: float) -> float:
    return 1 - (1 - t) ** 3


def ease_out_back(t: float) -> float:
    c1 = 1.4
    c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def lerp_seq(a: Sequence[float], b: Sequence[float], t: float) -> tuple[float, ...]:
    return tuple(x + (y - x) * t for x, y in zip(a, b))


@dataclass
class Tween:
    """Interpolates between two tuples over ``duration`` milliseconds."""

    start: Sequence[float]
    end: Sequence[float]
    duration: float
    kind: str  # what is being animated, e.g. "pawn", "wall", "wall_return"
    data: Any = None  # payload (player index, wall, ...)
    easing: Callable[[float], float] = ease_out_cubic
    elapsed: float = field(default=0.0)

    def update(self, dt_ms: float) -> None:
        self.elapsed = min(self.elapsed + dt_ms, self.duration)

    @property
    def done(self) -> bool:
        return self.elapsed >= self.duration

    @property
    def value(self) -> tuple[float, ...]:
        t = 1.0 if self.duration <= 0 else self.elapsed / self.duration
        return lerp_seq(self.start, self.end, self.easing(t))
