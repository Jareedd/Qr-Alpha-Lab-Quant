"""The graduation hurdle: what NET annual Sharpe must the next trial clear?

A trial "graduates" only if it clears every pre-registered leg, and the binding
leg in this project's history is the Deflated Sharpe Ratio (DSR >= 0.95) -- the
single leg that failed trial #8 (carry: net SR 0.87, DSR 0.865). The DSR
benchmarks a strategy's Sharpe against the expected maximum of N noise draws,
penalised for skew, kurtosis and sample length (Bailey & Lopez de Prado 2014).

This script inverts the repo's own metrics to answer, in interpretable units:
for a given trial count N, sample length n_obs, and return skew/kurtosis, what
net ANNUAL Sharpe does a strategy need so that DSR == 0.95? No market data is
touched; this is pure arithmetic on the existing metrics module, so it spends
zero trials and is safe to re-run any time.

Run:  PYTHONPATH=src .venv/Scripts/python.exe scripts/graduation_hurdle.py

Key reading of the output (see writeup/graduation_candidates_2026-06-14.md):
  * The hurdle scales ~1/sqrt(n_obs): a 2-year strategy needs net SR ~2.3 to
    graduate; a 15-year one needs ~0.88. Data-history depth is as decisive as
    edge size when ranking candidates.
  * Skew is second-order: carry-like -1.87 skew lifts the bar only ~3-5% over a
    symmetric strategy. Trial #8 failed because 0.87 < 1.09 (its sample length),
    not because skew alone disqualified it.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from quantlab.metrics import TRADING_DAYS, expected_max_sharpe


def deflation_benchmark_pp(n_trials: int, n_obs: int) -> float:
    """Per-period sr* exactly as deflated_sharpe_ratio() builds it (var_sr=1/n)."""
    return expected_max_sharpe(n_trials, 1.0 / n_obs, n_obs)


def required_ann_sharpe(
    n_trials: int, n_obs: int, skew: float, kurt: float, target: float = 0.95
) -> tuple[float, float]:
    """Net annual Sharpe such that DSR == target, plus the deflation benchmark.

    Inverts PSR(benchmark=sr*) = target. PSR z-score is
        z = (sr_pp - sr*) * sqrt(n-1) / denom,
        denom = sqrt(1 - skew*sr_pp + (kurt-1)/4 * sr_pp**2),
    which is implicit in sr_pp because denom depends on it; fixed-point solve.
    ``kurt`` is non-Fisher (normal == 3), matching probabilistic_sharpe_ratio.
    """
    z_t = stats.norm.ppf(target)
    s_star = deflation_benchmark_pp(n_trials, n_obs)
    sr = s_star + z_t / np.sqrt(n_obs - 1)  # seed assuming denom ~ 1
    for _ in range(200):
        denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
        sr_new = s_star + z_t * denom / np.sqrt(n_obs - 1)
        if abs(sr_new - sr) < 1e-13:
            break
        sr = sr_new
    return sr * np.sqrt(TRADING_DAYS), s_star * np.sqrt(TRADING_DAYS)


# Return-distribution scenarios (label, skew, non-Fisher kurtosis):
SCENARIOS = [
    ("symmetric", 0.0, 3.0),
    ("pos-skew", 0.5, 4.0),
    ("mild-neg", -1.0, 7.0),
    ("carry-like", -1.87, 9.0),
]

# Sample lengths in trading days, with the horizon each corresponds to:
SAMPLES = [
    (504, "~2 yr"),
    (1008, "~4 yr"),
    (1512, "~6 yr"),
    (2342, "~6 yr crypto (trial #8)"),
    (3378, "~15 yr equity (trials #2-7)"),
]


def main() -> None:
    for n_trials in (9, 10):
        print(f"\n=== Graduation hurdle at N = {n_trials} (DSR >= 0.95) ===")
        print(f"{'n_obs':>6} {'sample':>28} {'shape':>11} "
              f"{'req net SR(ann)':>15} {'defl bench':>11}")
        print("-" * 76)
        for n_obs, sample_tag in SAMPLES:
            for shape, skew, kurt in SCENARIOS:
                req, bench = required_ann_sharpe(n_trials, n_obs, skew, kurt)
                print(f"{n_obs:>6} {sample_tag:>28} {shape:>11} "
                      f"{req:>15.2f} {bench:>11.2f}")
            print()
    print("Check: trial #8 (net SR 0.87, skew -1.87, n_obs 2342, N=8) needed "
          f"{required_ann_sharpe(8, 2342, -1.87, 9.0)[0]:.2f} -> failed (DSR 0.865).")


if __name__ == "__main__":
    main()
