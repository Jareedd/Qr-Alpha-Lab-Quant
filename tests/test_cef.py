"""H6 CEF Stage-1 analytics (quantlab.cef) — known-answer tests.

Pins the total-return and discount-z construction BEFORE any real CEF universe
is assembled, per the Stage-1 spec. The headline test is the one the spec calls
out by name: a special distribution must register as INCOME, not a price crash
or discount widening.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import cef


def _frame(vals, cols=("F",), start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(vals))
    return pd.DataFrame(vals, index=idx, columns=cols)


def test_special_distribution_is_income_not_a_crash():
    # Price drops 10 -> 9 on the ex-date, but a $1 distribution is paid:
    # total return must be (9 + 1)/10 - 1 = 0, NOT -10%.
    price = _frame([[10.0], [9.0]])
    dist = _frame([[0.0], [1.0]])
    tr = cef.total_return(price, dist)
    assert tr["F"].iloc[1] == pytest.approx(0.0, abs=1e-12)
    # price-only (the conservative lower bound) WOULD show the -10% "loss"
    assert cef.total_return(price, None)["F"].iloc[1] == pytest.approx(-0.10)


def test_distribution_does_not_move_the_discount():
    # On the ex-date P and NAV both fall by the distribution; the discount
    # (P-NAV)/NAV must be ~unchanged (the spec's "not discount widening").
    price = _frame([[9.0], [8.0]])      # P falls 1.0
    nav = _frame([[10.0], [9.0]])       # NAV falls 1.0 (same distribution)
    disc = cef.discount(price, nav)
    assert disc["F"].iloc[0] == pytest.approx((9 - 10) / 10)   # -0.10
    assert disc["F"].iloc[1] == pytest.approx((8 - 9) / 9)     # -0.111...
    # the discount barely moved despite a large distribution-driven price drop
    assert abs(disc["F"].iloc[1] - disc["F"].iloc[0]) < 0.02


def test_discount_zscore_is_past_only():
    # A future extreme must not change today's z (no look-ahead). Build a flat
    # discount then a spike at the very end; the z BEFORE the spike is computed
    # from past data only and cannot reflect it.
    vals = [[-0.05]] * 300 + [[-0.40]]   # long flat, then a wide-discount spike
    disc = _frame(vals)
    z = cef.discount_zscore(disc, lookback=252, min_periods=126)
    # corrupt the future (the spike) and recompute the pre-spike window
    disc2 = disc.copy()
    disc2.iloc[-1] = -0.99
    z2 = cef.discount_zscore(disc2, lookback=252, min_periods=126)
    # the z on the second-to-last day uses only past data -> identical
    assert z["F"].iloc[-2] == pytest.approx(z2["F"].iloc[-2], nan_ok=True)


def test_discount_zscore_flags_extremes_correctly_signed():
    # A discount far BELOW its own trailing mean => large NEGATIVE z (the
    # wide-discount extreme H6 wants to buy).
    rng = np.random.default_rng(0)
    base = -0.05 + 0.01 * rng.standard_normal(300)
    base[-1] = -0.30                      # today: much wider than history
    disc = _frame(base.reshape(-1, 1))
    z = cef.discount_zscore(disc, lookback=252, min_periods=126)
    assert z["F"].iloc[-1] < -3.0         # a strong wide-discount extreme
