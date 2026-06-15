"""Multi-signal combination for the execution/risk engine.

A real multi-manager book runs many weakly-correlated alphas; combining them is
where diversification of *edge* (not just risk) comes from. Each signal is
cross-sectionally z-scored first — so a noisy, wide-dispersion signal can't
dominate a clean one by scale alone — then blended. IC-aware weighting lets you
down-weight signals with no recent predictive power, which is the honest default
given that most candidate signals here have ~none. Pure, source-agnostic,
past-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_z(panel: pd.DataFrame) -> pd.DataFrame:
    """Row-wise (per-date) cross-sectional z-score: (x − mean) / std."""
    return panel.sub(panel.mean(axis=1), axis=0).div(panel.std(axis=1) + 1e-12, axis=0)


def combine_signals(
    signals: dict[str, pd.DataFrame], weights: dict[str, float] | None = None
) -> pd.DataFrame:
    """Blend signals into one composite: z-score each, then a weighted sum
    (weights normalized by their absolute total; equal-weight if omitted).
    Frames are aligned on the union of dates/names, missing entries treated as
    neutral (0 after z-scoring)."""
    if not signals:
        raise ValueError("no signals to combine")
    w = weights or {k: 1.0 for k in signals}
    total = sum(abs(v) for v in w.values()) or 1.0
    composite = None
    for k, panel in signals.items():
        term = cross_sectional_z(panel) * (w.get(k, 0.0) / total)
        composite = term if composite is None else composite.add(term, fill_value=0.0)
    return composite


def trailing_ic(
    signal: pd.DataFrame, forward_returns: pd.DataFrame,
    lookback: int = 12, min_periods: int = 6,
) -> pd.Series:
    """Per-date trailing-mean cross-sectional rank IC of ``signal`` vs forward
    returns (past-only, shifted). The honest input to IC-weighting: a signal
    with no recent IC gets ~zero weight rather than equal billing."""
    ics = []
    for d in signal.index:
        a = signal.loc[d].dropna()
        b = forward_returns.loc[d].dropna() if d in forward_returns.index else pd.Series(dtype=float)
        common = a.index.intersection(b.index)
        ics.append(a[common].rank().corr(b[common].rank()) if len(common) >= 6 else np.nan)
    ic = pd.Series(ics, index=signal.index, dtype=float)
    return ic.rolling(lookback, min_periods=min_periods).mean().shift(1)
