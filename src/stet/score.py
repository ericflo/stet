"""Gain and the data-efficiency multiplier.

Gain is measured directly: how much better the curator's student does on the exam than the
unedited corpus's, same round, same initial weights. Gain, in bits per byte, is what ranks
curators.

The multiplier is a translation for humans: "how many times more unedited data would the
student have needed to do this well?" It is read off a reference curve of the unedited
student's exam score, as a loss (lower is better: minus the score), against training tokens,
measured once per season and published with it. Until a season's curve is measured there is
no multiplier. It is never
extrapolated: beyond the end of the measured curve it is reported as "more than" the last
measured point.

Only differences along the curve are read, so the level it is written at is free. Season 2's
curves are written as minus the gain of each unedited student over the one trained on the
budget itself (`stet curve`, exam/measure.py): 0 at 1x, positive below it, negative above. A
curator's gain is then looked up among gains measured the way it was, by the season's score.
"""

from __future__ import annotations

import math

from .spec import Season


def reference_curve(season: Season, kind: str, scale: str) -> list[tuple[float, float]] | None:
    """[(multiple of the budget, the unedited student's exam score as a loss), ...] sorted by multiple."""
    pts = (season.raw.get("reference_curves") or {}).get(f"{kind}/{scale}")
    return sorted((float(m), float(v)) for m, v in pts) if pts else None


def multiplier(season: Season, kind: str, scale: str, gain: float) -> tuple[float, bool] | None:
    """(multiplier, capped). None if the season has no curve for this kind of round and scale.

    `capped` means the gain runs off the end of the measured curve and the true multiplier
    is beyond the value returned.
    """
    return read_off(reference_curve(season, kind, scale), gain)


def read_off(curve: list[tuple[float, float]] | None, gain: float) -> tuple[float, bool] | None:
    """A gain read off a curve of (multiple of the budget, loss) points, as `multiplier` does:
    for a curve that is not the season's own, such as the one a season's finals measure."""
    curve = sorted((float(m), float(v)) for m, v in curve or [])
    if len(curve) < 2:
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
