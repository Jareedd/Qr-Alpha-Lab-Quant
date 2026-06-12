"""Falsification harness for the H2 carry world (synthetic.make_perp_panel).

Three properties get pinned before any real exchange data may be touched
(research-session law 4/5):

1. RECOVER: a one-line carry book (short the funding payers, long the
   receivers) harvests the planted premium in ``planted_carry``.
2. REJECT: the same book finds nothing in ``priced_carry`` — the null world
   where funding is fully compensated by mark-price drift.
3. THE LABEL LESSON, demonstrated not asserted: measuring the same book on
   funding-EXCLUSIVE (price-only) returns flips the verdict — the carry
   premium lives in the funding flows, so a price-only label measures the
   wrong object entirely. This is the bug the H2 registration amendment
   exists to prevent.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from quantlab import metrics
from quantlab.synthetic import make_perp_panel


def _carry_book_returns(
    prices: pd.DataFrame,
    funding: pd.DataFrame,
    lookback: int = 7,
    rebalance: int = 7,
    quantile: float = 0.25,
    funding_inclusive: bool = True,
) -> pd.Series:
    """The cheapest possible carry strategy: rank contracts by trailing
    funding, short the top quantile (they pay), long the bottom (they
    receive), equal-weight, rebalanced every ``rebalance`` days. Weights
    formed at t earn from t+1 (the repo's convention). A long's
    funding-inclusive daily return is price return MINUS funding."""
    rets = prices.pct_change(fill_method=None)
    total = rets - funding if funding_inclusive else rets
    sig = funding.rolling(lookback).mean()

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    n_side = max(1, int(len(prices.columns) * quantile))
    for i in range(lookback, len(prices.index), rebalance):
        row = sig.iloc[i].dropna()
        shorts = row.nlargest(n_side).index
        longs = row.nsmallest(n_side).index
        weights.iloc[i] = 0.0
        weights.iloc[i, weights.columns.get_indexer(shorts)] = -0.5 / n_side
        weights.iloc[i, weights.columns.get_indexer(longs)] = 0.5 / n_side
    held = weights.replace(0.0, np.nan).ffill(limit=rebalance).fillna(0.0)
    return (held.shift(1) * total).sum(axis=1)


def test_carry_world_attrs_and_reproducibility():
    p = make_perp_panel(seed=7)
    assert p.attrs["mode"] == "planted_carry"
    assert p.attrs["carry_gamma"] == pytest.approx(0.3)
    f = p.attrs["funding"]
    assert f.shape == p.shape and f.index.equals(p.index)
    # positive average funding (longs pay, the perp norm), some negative
    assert f.mean().mean() > 0
    assert (f.mean() < 0).any()
    assert p.equals(make_perp_panel(seed=7))  # deterministic
    with pytest.raises(ValueError, match="planted_carry"):
        make_perp_panel(mode="noise")


def test_planted_carry_is_recovered_and_priced_carry_rejected():
    # Pinned as PAIRED DIFFERENTIALS per seed, not absolute levels: at 1500
    # synthetic days the null world's SR has se ~ 0.4, so absolute "null is
    # near zero" thresholds are seed-lottery (measured: priced-world SR of
    # +0.57 on seed 23 from pure luck). The paired difference -- same seed,
    # same draws, the ONLY change is gamma -- isolates the premium and is
    # stable (~1.0-1.2 across seeds). Same lesson as the regime world's
    # vol-artifact: absolute levels lie, paired controls don't.
    diffs, priced_levels = [], []
    for seed in (7, 11, 23):
        planted = make_perp_panel(40, 1500, mode="planted_carry", seed=seed)
        priced = make_perp_panel(40, 1500, mode="priced_carry", seed=seed)
        sr_planted = metrics.sharpe(
            _carry_book_returns(planted, planted.attrs["funding"])
        )
        sr_priced = metrics.sharpe(
            _carry_book_returns(priced, priced.attrs["funding"])
        )
        diffs.append(sr_planted - sr_priced)
        priced_levels.append(sr_priced)
    # RECOVER: the premium is harvestable wherever planted, every seed.
    assert min(diffs) > 0.6
    # REJECT: the null world's MEAN level is luck-sized, not premium-sized
    # (se of the 3-seed mean ~ 0.24; the planted differential is ~1.1).
    assert abs(np.mean(priced_levels)) < 0.45


def test_price_only_label_measures_the_wrong_object():
    # The registration-critical lesson: on funding-EXCLUSIVE returns the
    # same planted-world book looks NEGATIVE (it is short the contracts
    # whose prices drift up with the (1-gamma) funding pass-through), so a
    # price-only backtest would reject a true carry premium. The label must
    # be funding-inclusive or trial #8 measures nothing.
    planted = make_perp_panel(n_assets=40, n_days=1500, mode="planted_carry", seed=7)
    f = planted.attrs["funding"]
    sr_inclusive = metrics.sharpe(_carry_book_returns(planted, f, funding_inclusive=True))
    sr_price_only = metrics.sharpe(_carry_book_returns(planted, f, funding_inclusive=False))
    assert sr_inclusive > 0.5
    assert sr_price_only < 0.0
    assert sr_inclusive - sr_price_only > 1.0
