"""Execution/risk engine — position-sizing known-answer tests.

Pins the honest-Kelly machine, whose headline property is the project's core
lesson encoded in code: a zero/uncertain edge sizes to ~zero. Nothing here
touches market data.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import sizing


def test_kelly_fraction_closed_form_and_guards():
    assert sizing.kelly_fraction(0.001, 0.02) == pytest.approx(0.001 / 0.02**2)  # 2.5
    assert sizing.kelly_fraction(-0.01, 0.02) == 0.0     # negative edge -> stand aside
    assert sizing.kelly_fraction(0.01, 0.0) == 0.0       # degenerate vol -> 0


def test_sharpe_standard_error_shrinks_with_n():
    assert sizing.sharpe_standard_error(0.0, 100) == pytest.approx(np.sqrt(1 / 100))
    assert sizing.sharpe_standard_error(0.5, 10_000) < sizing.sharpe_standard_error(0.5, 100)


def test_zero_or_negative_edge_sizes_to_zero():
    # THE lesson, in code: an unconfident edge gets no leverage.
    assert sizing.kelly_under_uncertainty(0.0, n_obs=500) == 0.0
    assert sizing.kelly_under_uncertainty(-0.3, n_obs=5000) == 0.0
    # a small positive Sharpe on thin data is not confidently > 0 -> 0
    assert sizing.kelly_under_uncertainty(0.05, n_obs=30) == 0.0


def test_size_grows_with_confidence_and_data():
    weak = sizing.kelly_under_uncertainty(0.8, n_obs=200)
    strong = sizing.kelly_under_uncertainty(0.8, n_obs=5000)
    assert 0.0 < weak < strong                       # more data -> tighter LB -> bigger
    # never exceeds full fractional Kelly on the point estimate
    assert strong <= 0.5 * 0.8 + 1e-9


def test_fractional_kelly_knob_scales_linearly():
    half = sizing.kelly_under_uncertainty(1.0, n_obs=4000, fraction=0.5)
    quarter = sizing.kelly_under_uncertainty(1.0, n_obs=4000, fraction=0.25)
    assert quarter == pytest.approx(half * 0.5, rel=1e-9)


def test_vol_target_scale_doubles_and_halves():
    idx = pd.bdate_range("2020-01-01", periods=2520)            # ~10y daily
    rng = np.random.default_rng(0)
    # construct returns with ~20% annual vol
    r = pd.Series(rng.standard_normal(2520) * (0.20 / np.sqrt(252)), index=idx)
    scale = sizing.vol_target_scale(r, target_vol=0.10)
    assert scale == pytest.approx(0.5, rel=0.1)                 # 20% book -> 0.5x for 10%
    assert sizing.vol_target_scale(r, target_vol=0.20) == pytest.approx(2 * scale, rel=1e-9)
    assert sizing.vol_target_scale(pd.Series([0.0, 0.0]), 0.10) == 0.0


def test_size_book_collapses_a_no_edge_book():
    idx = pd.bdate_range("2020-01-01", periods=1000)
    w = pd.Series({"A": 0.5, "B": -0.5})
    book = pd.Series(np.random.default_rng(1).standard_normal(1000) * 0.01, index=idx)
    # zero estimated Sharpe -> gross exposure goes to zero regardless of vol target
    sized = sizing.size_book(w, book, sharpe_hat=0.0, n_obs=1000, target_vol=0.10)
    assert sized.abs().sum() == 0.0
    # a confident edge -> nonzero, dollar-neutral preserved
    sized2 = sizing.size_book(w, book, sharpe_hat=1.2, n_obs=4000, target_vol=0.10)
    assert sized2.abs().sum() > 0
    assert sized2.sum() == pytest.approx(0.0, abs=1e-12)        # stays market-neutral
