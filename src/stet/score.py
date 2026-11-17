"""Gain and the data-efficiency multiplier.

Gain is measured directly: baseline exam loss minus the editor's, same round, same initial
weights. The multiplier translates a gain into "how many times more unedited data would
the student have needed?", by reading it off a reference curve of baseline loss against
training tokens that is measured once per season and published with it.
"""

from __future__ import annotations

import math

from .spec import Season


def reference_curve(season: Season, scale: str) -> list[tuple[float, float]] | None:
    """[(multiple of the budget, mean baseline exam loss), ...] sorted by multiple."""
    pts = (season.raw.get("reference_curves") or {}).get(scale)
    return sorted((float(m), float(v)) for m, v in pts) if pts else None


def multiplier(season: Season, scale: str, gain: float) -> tuple[float, bool] | None:
    """(multiplier, extrapolated). None if the season has no curve for this scale."""
    curve = reference_curve(season, scale)
    if not curve or len(curve) < 2:
        return None
    base = dict(curve).get(1.0)
    if base is None:
        return None
    target = base - gain
    xs = [math.log2(m) for m, _ in curve]
    ys = [v for _, v in curve]  # decreasing in m
    for i in range(len(xs) - 1):
        hi, lo = ys[i], ys[i + 1]
        if lo <= target <= hi and hi != lo:
            x = xs[i] + (hi - target) / (hi - lo) * (xs[i + 1] - xs[i])
            return 2**x, False
    # Outside the measured range: continue the nearest segment's slope, and say so.
    i = 0 if target > ys[0] else len(xs) - 2
    slope = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])
    if slope >= 0:
        return None
    x = xs[i] + (target - ys[i]) / slope
    return 2 ** max(-10.0, min(10.0, x)), True


def format_multiplier(m: tuple[float, bool] | None) -> str:
    if m is None:
        return "n/a"
    return f"{m[0]:.2f}x" + (" (extrapolated)" if m[1] else "")
