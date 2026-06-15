"""H6 discount-reversion backtest harness (trial #11) — the registered strategy.

Consumes a TOTAL-RETURN price panel (distribution-inclusive; yfinance adjusted
close for real data, the synthetic CEF panel for the gate) and a DISCOUNT panel,
both WEEKLY (the cadence Stage-1 found the free NAV data supports), and
implements the H6 registration verbatim:

- Signal: per fund, z = (discount - mean_52w)/std_52w (past-only; the absolute
  discount level is a value trap, the z vs own history is the signal).
- Book: dollar-neutral equal-weight quintiles — LONG the most-NEGATIVE z (widest
  discount extreme, expected to revert up), SHORT the most-POSITIVE z (richest).
  Rebalanced every 4 weeks, held between.
- Label/return: weekly distribution-inclusive total return.
- Costs: 25 bps/side on turnover (linear headline; tail spreads are wide).

Weights formed at week w earn from w+1 (PIT by construction). The registered
paired controls (label shuffle, daily-NAV-only subuniverse) and the seasonality
subreport are computed from the objects this module returns. Mirrors
perp_carry.py so trial #11 is the same machine as trial #8, pointed at funds.

Sign convention: reversion is PROFITABLE => low z predicts HIGH forward return
=> the IC of z vs forward return is NEGATIVE (t_NW <= -2 is the success
direction), exactly as the carry signal's raw-funding IC is negative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab import cef


def discount_z(discount: pd.DataFrame, lookback: int = 52,
               min_periods: int = 26) -> pd.DataFrame:
    """H6 signal: each fund's discount vs its OWN trailing 52-week mean/std
    (past-only). Reuses the Stage-1-pinned primitive."""
    return cef.discount_zscore(discount, lookback=lookback, min_periods=min_periods)


def reversion_weights(
    signal: pd.DataFrame,
    quantile: float = 0.2,
    rebalance: int = 4,
    universe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Dollar-neutral equal-weight quintile book: LONG the lowest-z (widest
    discount), SHORT the highest-z (richest), rebalanced every ``rebalance``
    weeks and held between. Mirrors perp_carry.carry_weights' full-vector reset
    (the fix for stale positions surviving a rebalance and breaking
    dollar-neutrality). ``universe`` (boolean week x fund) restricts eligibility
    — used by the NAV-staleness paired control."""
    sig = signal.where(universe) if universe is not None else signal
    target = pd.DataFrame(np.nan, index=signal.index, columns=signal.columns)
    for i in range(0, len(signal.index), rebalance):
        row = sig.iloc[i].dropna()
        n_side = int(len(row) * quantile)
        target.iloc[i] = 0.0                       # full reset every rebalance
        if n_side < 2:
            continue
        longs = row.nsmallest(n_side).index        # widest discount -> revert up
        shorts = row.nlargest(n_side).index        # richest -> revert down
        target.iloc[i, target.columns.get_indexer(longs)] = 0.5 / n_side
        target.iloc[i, target.columns.get_indexer(shorts)] = -0.5 / n_side
    return target.ffill(limit=rebalance - 1).fillna(0.0)


def reversion_backtest(
    price: pd.DataFrame,
    discount: pd.DataFrame,
    cost_bps_per_side: float = 25.0,
    quantile: float = 0.2,
    rebalance: int = 4,
    lookback: int = 52,
    universe: pd.DataFrame | None = None,
    returns_override: pd.DataFrame | None = None,
) -> dict:
    """Run the registered reversion book on weekly panels. Weights at w earn
    from w+1. ``returns_override`` (a pre-built total-return panel) feeds the
    label-shuffle control without recomputing weights; otherwise total return =
    ``price.pct_change``."""
    total_ret = (returns_override if returns_override is not None
                 else price.pct_change(fill_method=None))
    signal = discount_z(discount, lookback=lookback)
    weights = reversion_weights(signal, quantile=quantile, rebalance=rebalance,
                                universe=universe)
    held = weights.shift(1)

    gross = (held * total_ret).sum(axis=1, min_count=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (cost_bps_per_side / 1e4)
    net = (gross - cost).dropna()
    ann_turnover = float(turnover.sum() / max(len(weights), 1) * 52)
    return {
        "net": net,
        "gross": gross.dropna(),
        "weights": weights,
        "held": held,
        "signal": signal,
        "total_ret": total_ret,
        "annual_turnover": ann_turnover,
    }


def forward_total_return(price: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
    """Forward ``horizon``-week total return per fund (the IC label): the sum of
    weekly total returns over (w, w+h]."""
    tot = price.pct_change(fill_method=None)
    return tot.shift(-1).rolling(horizon).sum().shift(-(horizon - 1))


def shuffle_returns(total_ret: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Registered paired control: permute total returns ACROSS funds within each
    week (destroys the discount->return link, preserves marginals). The same
    weights applied to this must earn ~nothing; if they earn, the harvest was
    not reversion."""
    rng = np.random.default_rng(seed)
    arr = total_ret.to_numpy().copy()
    for i in range(arr.shape[0]):
        row = arr[i]
        finite = np.where(np.isfinite(row))[0]
        if len(finite) > 1:
            arr[i, finite] = row[rng.permutation(finite)]
    return pd.DataFrame(arr, index=total_ret.index, columns=total_ret.columns)
