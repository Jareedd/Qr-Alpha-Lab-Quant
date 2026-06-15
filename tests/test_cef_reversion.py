"""H6 reversion harness (quantlab.cef_reversion) — known-answer + machinery tests.

The headline test (test_machinery_gate_*) is the falsification gate the trial
runs in-env before touching real data: the harness MUST recover reversion in the
planted synthetic world and find ~nothing in the random-walk null. If this fails,
no real H6 number is trustworthy.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import cef_reversion as cr
from quantlab import metrics
from quantlab.synthetic import make_cef_panel


def _weeks(n, start="2015-01-02"):
    return pd.bdate_range(start, periods=n, freq="W-FRI")


def test_reversion_weights_long_low_z_short_high_z_dollar_neutral():
    # 10 funds, z = 0..9 on the first (rebalance) row; quintile -> 2 per side.
    idx = _weeks(1)
    z = pd.DataFrame([list(range(10))], index=idx,
                     columns=[f"F{i}" for i in range(10)], dtype=float)
    w = cr.reversion_weights(z, quantile=0.2, rebalance=4)
    row = w.iloc[0]
    assert row.sum() == pytest.approx(0.0, abs=1e-12)          # dollar-neutral
    assert row["F0"] == pytest.approx(0.25) and row["F1"] == pytest.approx(0.25)  # lowest z LONG
    assert row["F8"] == pytest.approx(-0.25) and row["F9"] == pytest.approx(-0.25)  # highest z SHORT
    assert row[[f"F{i}" for i in range(2, 8)]].abs().sum() == 0.0  # middle flat


def test_reversion_weights_flatten_dropped_names_on_rebalance():
    # A name in a quintile at rebal 0 that leaves it at rebal 4 must go to 0,
    # not carry its stale weight forward (the dollar-neutrality bug). Needs >=10
    # funds so quintile n_side = int(10*0.2) = 2 clears the n_side<2 guard.
    idx = _weeks(8)
    cols = [f"F{i}" for i in range(10)]
    z = pd.DataFrame(np.nan, index=idx, columns=cols)
    z.iloc[0] = list(range(10))                      # F0 lowest -> LONG
    z.iloc[4] = [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]      # F0 now highest -> SHORT
    w = cr.reversion_weights(z, quantile=0.2, rebalance=4)
    assert w.iloc[0]["F0"] > 0              # long at first rebalance
    assert w.iloc[4]["F0"] < 0             # flipped to short, not stuck long


def test_forward_total_return_is_past_to_future_sum():
    idx = _weeks(6)
    price = pd.DataFrame({"F": [100, 110, 121, 133.1, 146.41, 161.051]}, index=idx)
    fwd = cr.forward_total_return(price, horizon=2)
    # forward h-week return is the SUM of weekly returns (carry convention):
    # 0.10 + 0.10 = 0.20, not the compounded 0.21.
    assert fwd["F"].iloc[0] == pytest.approx(0.20, abs=1e-6)


def test_shuffle_returns_preserves_row_marginals():
    idx = _weeks(3)
    tr = pd.DataFrame(np.arange(12, dtype=float).reshape(3, 4),
                      index=idx, columns=list("ABCD"))
    sh = cr.shuffle_returns(tr, seed=1)
    for i in range(3):
        assert sorted(sh.iloc[i]) == sorted(tr.iloc[i])    # same values, permuted


def test_machinery_gate_planted_reversion_beats_random_walk():
    """The trial's in-env falsification gate, as a unit test (paired per seed)."""
    diffs, planted_srs, null_srs = [], [], []
    for seed in (7, 11, 23):
        planted = make_cef_panel(120, 520, mode="planted_reversion", seed=seed)
        rw = make_cef_panel(120, 520, mode="random_walk", seed=seed)
        sr_p = metrics.sharpe(
            cr.reversion_backtest(planted, planted.attrs["discount"],
                                  cost_bps_per_side=0.0)["net"], periods=52)
        sr_n = metrics.sharpe(
            cr.reversion_backtest(rw, rw.attrs["discount"],
                                  cost_bps_per_side=0.0)["net"], periods=52)
        diffs.append(sr_p - sr_n)
        planted_srs.append(sr_p)
        null_srs.append(sr_n)
    # planted must clearly beat the random-walk null, every seed (paired).
    assert min(diffs) > 0.5, f"planted-null differential too small: {diffs}"
    assert min(planted_srs) > 0.5, f"planted not recovered: {planted_srs}"
    assert max(abs(s) for s in null_srs) < 0.6, f"null not flat: {null_srs}"
