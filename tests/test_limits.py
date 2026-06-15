"""Engine — risk-limit known-answer tests."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import limits


def test_cap_position_clips_both_signs():
    w = pd.Series({"A": 0.20, "B": -0.15, "C": 0.01})
    capped = limits.cap_position(w, 0.05)
    assert capped["A"] == 0.05 and capped["B"] == -0.05 and capped["C"] == 0.01


def test_cap_gross_scales_down_only():
    w = pd.Series({"A": 1.0, "B": -1.0})        # gross 2.0
    assert limits.cap_gross(w, 1.0).abs().sum() == pytest.approx(1.0)
    assert limits.cap_gross(w, 5.0).equals(w)   # under cap -> unchanged
    df = pd.DataFrame([[0.5, -0.5], [1.0, -1.0]], columns=["A", "B"])
    out = limits.cap_gross(df, 1.0)
    assert out.abs().sum(axis=1).max() == pytest.approx(1.0)


def test_cap_turnover_limits_trade_and_is_identity_under_cap():
    prev = pd.Series({"A": 0.0, "B": 0.0})
    target = pd.Series({"A": 0.5, "B": -0.5})    # desired turnover 1.0
    half = limits.cap_turnover(prev, target, 0.5)
    assert half.sub(prev).abs().sum() == pytest.approx(0.5)
    assert half["A"] == pytest.approx(0.25)      # moved halfway
    assert limits.cap_turnover(prev, target, 2.0).equals(target)   # under cap


def test_drawdown_scale_full_then_degrosses():
    up = pd.Series(np.full(50, 0.01))            # steadily rising -> no drawdown
    assert limits.drawdown_scale(up, max_dd=0.15).min() == pytest.approx(1.0)
    # a 30% crash then flat -> drawdown 0.30 > max_dd 0.15 -> de-grossed, >= floor
    crash = pd.Series([0.0] * 10 + [-0.30] + [0.0] * 20)
    ds = limits.drawdown_scale(crash, max_dd=0.15, floor=0.25)
    assert ds.iloc[-1] < 1.0
    assert ds.min() >= 0.25 - 1e-9
