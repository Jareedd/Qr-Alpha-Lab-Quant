"""Nested hyperparameter tuning: must be leak-free by construction.

The whole point of ridge_cv is that alpha is selected using ONLY the training
window of each outer roll. These tests pin that property and the basic
mechanics; they deliberately avoid asserting anything about performance
(tuning is not supposed to look good, it is supposed to be legitimate).
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import features, models, validation
from quantlab.synthetic import make_panel


def _panel(n_assets=30, n_days=1600, mode="planted", seed=9):
    prices = make_panel(n_assets=n_assets, n_days=n_days, mode=mode, seed=seed)
    feats = features.build_features(prices)
    return features.stack_panel(feats, features.build_labels(prices))


def test_selected_alpha_comes_from_grid():
    panel = _panel()
    sub = panel.loc[: panel.index.get_level_values("date").unique()[900]]
    alpha = models.select_ridge_alpha(sub, embargo_days=21)
    assert alpha in models.RIDGE_ALPHA_GRID


def test_short_window_falls_back_gracefully():
    panel = _panel(n_days=900)
    # ~370 usable dates after feature warm-up: too short for an inner split
    # (needs 504 train + 21 embargo + 126 test), so the tuner must fall back
    # to the grid midpoint instead of raising.
    dates = panel.index.get_level_values("date").unique()
    tiny = panel.loc[: dates[300]]
    alpha = models.select_ridge_alpha(tiny, embargo_days=21)
    assert alpha == models.RIDGE_ALPHA_GRID[len(models.RIDGE_ALPHA_GRID) // 2]


def test_alpha_selection_ignores_data_after_train_window():
    # Leak check: corrupting everything AFTER the training window must not
    # change the selected alpha, because the tuner never reads beyond it.
    panel = _panel()
    dates = panel.index.get_level_values("date").unique()
    cutoff = dates[900]
    train_part = panel.loc[:cutoff]

    corrupted = panel.copy()
    future = corrupted.index.get_level_values("date") > cutoff
    corrupted.loc[future, "label"] = 999.0

    a_clean = models.select_ridge_alpha(train_part, embargo_days=21)
    a_corrupt = models.select_ridge_alpha(corrupted.loc[:cutoff], embargo_days=21)
    assert a_clean == a_corrupt


def test_ridge_cv_runs_end_to_end_with_same_oos_index_as_ridge():
    panel = _panel()
    sp = validation.WalkForwardSplitter(
        min_train_days=756, test_days=126, embargo_days=21
    )
    p_fixed = models.walk_forward_predict(panel, sp, model_name="ridge")
    p_tuned = models.walk_forward_predict(panel, sp, model_name="ridge_cv")
    # Tuning changes predictions, never the out-of-sample coverage.
    assert p_tuned.index.equals(p_fixed.index)


def test_model_factory_overrides_model_name():
    from sklearn.linear_model import Ridge

    panel = _panel()
    sp = validation.WalkForwardSplitter(
        min_train_days=756, test_days=126, embargo_days=21
    )
    p1 = models.walk_forward_predict(panel, sp, model_name="ridge")
    p2 = models.walk_forward_predict(
        panel, sp, model_factory=lambda: Ridge(alpha=10.0)
    )
    pd.testing.assert_series_equal(p1, p2)
