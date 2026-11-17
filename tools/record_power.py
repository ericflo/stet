#!/usr/bin/env python3
"""How often the record rule hands the record to a challenger with a given true lead.

Simulates the rule as the server applies it (server/rating.go, checkRecord): at each look,
a one-sided paired t-test of "the challenger's mean advantage over the holder is no more
than the margin", at alpha split evenly across the looks.

    uv run --no-project --with numpy --with scipy python tools/record_power.py --sd 0.045 --margin 0.03
"""
import argparse

import numpy as np
from scipy import stats


def takes_record(lead: float, sd: float, margin: float, looks: tuple[int, ...], alpha: float, trials: int, rng) -> float:
    per_look = alpha / len(looks)
    won = np.zeros(trials, bool)
    x = rng.normal(lead - margin, sd, size=(trials, max(looks)))
    for n in looks:
        d = x[:, :n]
        t = d.mean(1) / (d.std(1, ddof=1) / np.sqrt(n))
        won |= stats.t.sf(t, n - 1) < per_look
    return float(won.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sd", type=float, required=True, help="standard deviation of the paired difference between two editors' gains")
    ap.add_argument("--margin", type=float, required=True)
    ap.add_argument("--looks", type=int, nargs="+", default=[5, 10, 20, 40])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--leads", type=float, nargs="+", default=[0.0, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.143])
    ap.add_argument("--trials", type=int, default=40000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    print(f"paired sd {args.sd}, margin {args.margin}, looks {args.looks}")
    for lead in args.leads:
        print(f"  true lead {lead:.3f}: takes the record {takes_record(lead, args.sd, args.margin, tuple(args.looks), args.alpha, args.trials, rng):6.1%}")


if __name__ == "__main__":
    main()
