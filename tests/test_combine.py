"""Engine — multi-signal combination known-answer tests."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import combine


def test_cross_sectional_z_is_mean0_std1_per_row():
    df = pd.DataFrame({"A": [1.0, 10], "B": [2, 20], "C": [3, 30]})
    z = combine.cross_sectional_z(df)
    assert z.mean(axis=1).abs().max() < 1e-9
    assert z.std(axis=1).sub(1.0).abs().max() < 0.2          # ddof=1 on 3 pts


def test_combine_equal_weight_average_and_weights():
    s1 = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]})
    s2 = pd.DataFrame({"A": [3.0], "B": [2.0], "C": [1.0]})   # opposite ordering
    eq = combine.combine_signals({"s1": s1, "s2": s2})
    assert eq.abs().max().max() < 1e-9                        # opposite signals cancel
    only1 = combine.combine_signals({"s1": s1, "s2": s2}, weights={"s1": 1.0, "s2": 0.0})
    assert only1.loc[0, "C"] > only1.loc[0, "A"]              # s1 ranking survives
    with pytest.raises(ValueError):
        combine.combine_signals({})


def test_trailing_ic_detects_a_perfect_then_zero_signal():
    idx = pd.bdate_range("2020-01-01", periods=30, freq="W-FRI")
    cols = list("ABCDEFGH")                              # >=6 names: IC needs a cross-section
    rng = np.random.default_rng(0)
    fwd = pd.DataFrame(rng.standard_normal((30, 8)), index=idx, columns=cols)
    sig = fwd.copy()                                          # signal == forward return -> IC 1
    ic = combine.trailing_ic(sig, fwd, lookback=4, min_periods=2)
    assert ic.dropna().iloc[-1] == pytest.approx(1.0, abs=1e-9)
