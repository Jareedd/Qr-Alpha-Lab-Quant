"""Closed-end fund (H6) Stage-1 analytics: total-return construction and the
discount-z signal.

Source-agnostic by design — every function operates on aligned pandas frames
(price / NAV / discount / distribution), so the construction logic is pinned by
known-answer tests BEFORE any real universe is assembled. This is the Stage-1
spec's "total-return machinery pinned by known-answer tests" requirement (a 10%
special distribution must register as INCOME, not as discount widening or a
price crash).

Protocol note (H6 is two-stage): NOTHING here computes a signal-vs-forward-
return relationship. That is Stage 2 — the registered trial — and is gated on a
PROPOSED H6 registration + a trial spend. These are descriptive/feature
primitives only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def total_return(price: pd.DataFrame, distributions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Daily distribution-inclusive total return: ``(P_t + D_t) / P_{t-1} - 1``,
    where ``D_t`` is the cash distribution with EX-date t (0 if none).

    CEF returns are mostly distributions; a price-only return UNDERSTATES them
    and, because wide-discount funds tend to distribute heavily, biases a
    long-discount book DOWNWARD — a conservative lower bound, never the headline.
    Passing ``distributions=None`` returns exactly that price-only lower bound,
    explicitly labelled as such by the argument.
    """
    price = price.astype(float)
    if distributions is None:
        return price.pct_change(fill_method=None)
    dist = distributions.reindex_like(price).fillna(0.0)
    return (price + dist) / price.shift(1) - 1.0


def discount(price: pd.DataFrame, nav: pd.DataFrame) -> pd.DataFrame:
    """Premium/discount to NAV, ``(P - NAV) / NAV``. Negative = trading at a
    discount (cheap vs net asset value). A distribution drops P and NAV
    together, so the discount is (to first order) unaffected by it — which is
    exactly why discount-based signals are robust to CEFs' heavy payouts."""
    return (price - nav) / nav


def discount_zscore(
    disc: pd.DataFrame, lookback: int = 252, min_periods: int = 126
) -> pd.DataFrame:
    """H6 signal: each fund's discount today vs its OWN trailing mean/std
    (past-only, so no look-ahead). A wide-discount extreme relative to the
    fund's own history is a large NEGATIVE z. This is reversion-from-extremes,
    NOT the absolute discount level — the absolute level is a value trap
    (some funds are structurally cheap forever)."""
    mean = disc.rolling(lookback, min_periods=min_periods).mean()
    std = disc.rolling(lookback, min_periods=min_periods).std()
    return (disc - mean) / std


# --- H6 Stage-2 backtest: cross-sectional discount-z mean-reversion ----------
# The signal we TRADE is -discount_zscore, so a HIGH signal = a discount wide vs
# the fund's own history = BUY. Dollar-neutral, equal-weight quantiles, held
# between rebalances. Weights at t earn from t+1 (PIT). Mirrors perp_carry.

def reversion_weights(
    disc: pd.DataFrame, lookback: int = 52, min_periods: int = 26,
    quantile: float = 0.2, rebalance: int = 4,
) -> pd.DataFrame:
    """Dollar-neutral equal-weight book: LONG the widest-discount-vs-own-history
    quantile, SHORT the narrowest/premium quantile, rebalanced every
    ``rebalance`` periods and held between. Full-vector reset at each rebalance
    (the perp_carry stale-position fix)."""
    sig = -discount_zscore(disc, lookback, min_periods)  # high = wide discount = buy
    target = pd.DataFrame(np.nan, index=disc.index, columns=disc.columns)
    for i in range(0, len(disc.index), rebalance):
        row = sig.iloc[i].dropna()
        n_side = int(len(row) * quantile)
        target.iloc[i] = 0.0
        if n_side < 2:
            continue
        longs = row.nlargest(n_side).index
        shorts = row.nsmallest(n_side).index
        target.iloc[i, target.columns.get_indexer(longs)] = 0.5 / n_side
        target.iloc[i, target.columns.get_indexer(shorts)] = -0.5 / n_side
    return target.ffill(limit=rebalance - 1).fillna(0.0)


def reversion_backtest(
    disc: pd.DataFrame, ret: pd.DataFrame, cost_bps_per_side: float = 25.0,
    lookback: int = 52, min_periods: int = 26, quantile: float = 0.2,
    rebalance: int = 4, periods_per_year: int = 52,
) -> dict:
    """Run the discount-reversion book on a discount panel + a return panel
    (weekly). Returns daily/weekly gross & net series, turnover, weights."""
    w = reversion_weights(disc, lookback, min_periods, quantile, rebalance)
    held = w.shift(1)
    gross = (held * ret).sum(axis=1, min_count=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    net = (gross - turnover * cost_bps_per_side / 1e4).dropna()
    return {
        "net": net, "gross": gross.dropna(), "weights": w,
        "annual_turnover": float(turnover.sum() / len(w) * periods_per_year),
    }


def reversion_ic(disc: pd.DataFrame, fwd: pd.DataFrame,
                 lookback: int = 52, min_periods: int = 26, min_names: int = 10) -> pd.Series:
    """Per-period cross-sectional rank IC of the traded signal (-z) vs forward
    return. Positive = the wide-discount book is predictive."""
    sig = -discount_zscore(disc, lookback, min_periods)
    ics = []
    for d in sig.index:
        a, b = sig.loc[d].dropna(), fwd.loc[d].dropna()
        common = a.index.intersection(b.index)
        if len(common) >= min_names:
            ics.append(a[common].rank().corr(b[common].rank()))
    return pd.Series(ics, dtype=float).dropna()
