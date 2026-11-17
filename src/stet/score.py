"""Gain and the data-efficiency multiplier.

Gain is measured directly: baseline exam loss minus the editor's, same round, same initial
weights. Gain, in nats, is what ranks editors.

The multiplier is a translation for humans: "how many times more unedited data would the
student have needed to do this well?" It is read off a reference curve of baseline loss
against training tokens, measured once per season and published with it. It is never
extrapolated: beyond the end of the measured curve it is reported as "more than" the last
measured point.
"""

from __future__ import annotations

import math

from .spec import Season


def reference_curve(season: Season, kind: str, scale: str) -> list[tuple[float, float]] | None:
    """[(multiple of the budget, mean baseline exam loss), ...] sorted by multiple."""
    pts = (season.raw.get("reference_curves") or {}).get(f"{kind}/{scale}")
    return sorted((float(m), float(v)) for m, v in pts) if pts else None


def multiplier(season: Season, kind: str, scale: str, gain: float) -> tuple[float, bool] | None:
    """(multiplier, capped). None if the season has no curve for this kind of round and scale.

    `capped` means the gain runs off the end of the measured curve and the true multiplier
    is beyond the value returned.
    """
    curve = reference_curve(season, kind, scale)
    if not curve or len(curve) < 2:
        return None
    base = dict(curve).get(1.0)
    if base is None:
        return None
    target = base - gain
    xs = [math.log2(m) for m, _ in curve]
    ys = [v for _, v in curve]  # decreasing in m
    if target >= ys[0]:
        return curve[0][0], target > ys[0]
    if target <= ys[-1]:
        return curve[-1][0], target < ys[-1]
    for i in range(len(xs) - 1):
        hi, lo = ys[i], ys[i + 1]
        if lo <= target <= hi and hi != lo:
            x = xs[i] + (hi - target) / (hi - lo) * (xs[i + 1] - xs[i])
            return 2**x, False
    return None


def format_multiplier(m: tuple[float, bool] | None) -> str:
    if m is None:
        return "n/a"
    value, capped = m
    if capped:
        return f"{'>' if value >= 1 else '<'}{value:g}x"
    return f"{value:.2f}x"
