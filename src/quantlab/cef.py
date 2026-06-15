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
