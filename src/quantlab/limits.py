"""Risk limits for the execution/risk engine: position, gross, turnover, and
drawdown controls. Applied AFTER sizing, BEFORE execution. Past-only where
time-dependent. Each is a small, separately-tested transform so the engine just
composes them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cap_position(weights, max_weight: float):
    """Clip each name's weight to ±``max_weight`` (sign preserved). Works on a
    Series (one date) or a DataFrame (a panel)."""
    return weights.clip(lower=-max_weight, upper=max_weight)


def cap_gross(weights, max_gross: float):
    """Scale down so gross exposure (sum |w|) ≤ ``max_gross``; never scales UP.
    Per-row for a DataFrame, whole-vector for a Series."""
    if isinstance(weights, pd.DataFrame):
        gross = weights.abs().sum(axis=1)
        scale = (max_gross / gross).clip(upper=1.0).replace([np.inf, -np.inf], 1.0).fillna(1.0)
        return weights.mul(scale, axis=0)
    gross = float(weights.abs().sum())
    return weights * min(1.0, max_gross / gross) if gross > 0 else weights


def cap_turnover(prev_weights: pd.Series, target_weights: pd.Series,
                 max_turnover: float) -> pd.Series:
    """Move only partway from ``prev`` toward ``target`` so the trade
    sum |Δw| ≤ ``max_turnover``. Identity when the desired trade is already
    within budget — the limit bites only when it must."""
    trade = target_weights.sub(prev_weights, fill_value=0.0)
    tt = float(trade.abs().sum())
    if tt <= max_turnover or tt == 0.0:
        return target_weights
    return prev_weights.add(trade * (max_turnover / tt), fill_value=0.0)


def drawdown_scale(returns: pd.Series, max_dd: float = 0.15,
                   floor: float = 0.25) -> pd.Series:
    """Past-only exposure multiplier from realized drawdown: full size until the
    drawdown exceeds ``max_dd``, then de-gross linearly toward ``floor`` as the
    drawdown deepens (reaching ``floor`` at 2×max_dd). Shifted one period — today's
    size reacts to the drawdown realized through yesterday, never the future."""
    eq = (1.0 + returns.fillna(0.0)).cumprod()
    dd = eq / eq.cummax() - 1.0
    excess = (-dd - max_dd).clip(lower=0.0)
    mult = (1.0 - (excess / max_dd) * (1.0 - floor)).clip(lower=floor)
    return mult.shift(1).fillna(1.0)
