"""Baseline strategies: correctness and a falsification-style sanity check.

The planted synthetic panel's predictable component IS cross-sectional 12-1
momentum, so the momentum baseline must earn a clearly positive gross Sharpe
on it -- if it doesn't, either the baseline or the panel is broken.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import backtest, baselines, features, models, validation
from quantlab.synthetic import make_panel


def _setup(mode="planted", seed=7, n_assets=40, n_days=2000):
    prices = make_panel(n_assets=n_assets, n_days=n_days, mode=mode, seed=seed)
    feats = features.build_features(prices)
    panel = features.stack_panel(feats, features.build_labels(prices))
    sp = validation.WalkForwardSplitter(min_train_days=504, test_days=126, embargo_days=21)
    preds = models.walk_forward_predict(panel, sp)
    return prices, feats, preds


def test_momentum_baseline_is_dollar_neutral():
    prices, feats, preds = _setup()
    w = baselines.momentum_baseline_weights(feats, preds.index)
    active = w[w.abs().sum(axis=1) > 0]
    assert len(active) > 0
    assert (active.sum(axis=1).abs() < 1e-9).all()
    assert ((active.abs().sum(axis=1) - 1.0).abs() < 1e-9).all()


def test_momentum_baseline_recovers_planted_momentum():
    # Use the standard falsification panel (60 assets x 3000 days): the
    # planted signal is weak by design, and a smaller panel leaves too few
    # names per decile for a stable threshold (gross SR ~1.2 here vs ~0.27
    # on a 40x2000 panel).
    prices, feats, preds = _setup(mode="planted", n_assets=60, n_days=3000)
    w = baselines.momentum_baseline_weights(feats, preds.index)
    res = backtest.run_backtest(w, prices, cost_bps=0.0)
    gross_sr = res["gross"].mean() / res["gross"].std() * np.sqrt(252)
    assert gross_sr > 0.5  # the planted signal is literally this feature


def test_momentum_baseline_uses_only_oos_dates():
    prices, feats, preds = _setup()
    w = baselines.momentum_baseline_weights(feats, preds.index)
    oos_dates = preds.index.get_level_values("date").unique()
    assert w.index.isin(oos_dates).all()


def test_equal_weight_returns_match_row_mean():
    prices, _, _ = _setup(mode="noise", seed=3)
    ew = baselines.equal_weight_returns(prices)
    expected = prices.pct_change().mean(axis=1)
    pd.testing.assert_series_equal(ew, expected)
    start = prices.index[100]
    assert baselines.equal_weight_returns(prices, start=start).index[0] == start
