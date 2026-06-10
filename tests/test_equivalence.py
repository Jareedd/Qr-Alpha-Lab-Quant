"""Vectorized hot paths must match their naive reference implementations.

The IC and weight-construction code was vectorized for speed. Speed is worth
nothing if it changes results, so each optimized function is pinned to a
deliberately simple per-date loop here. If a future 'optimization' drifts the
numbers, these tests fail loudly.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import backtest, features, models, validation
from quantlab.synthetic import make_panel


def _oos_preds_and_panel(n_assets=30, n_days=1400, seed=11):
    prices = make_panel(n_assets=n_assets, n_days=n_days, mode="planted", seed=seed)
    feats = features.build_features(prices)
    panel = features.stack_panel(feats, features.build_labels(prices))
    sp = validation.WalkForwardSplitter(min_train_days=504, test_days=126, embargo_days=21)
    preds = models.walk_forward_predict(panel, sp)
    return preds, panel


def test_information_coefficient_matches_naive_groupby():
    preds, panel = _oos_preds_and_panel()
    fast = models.information_coefficient(preds, panel)

    df = pd.DataFrame({"pred": preds, "label": panel["label"]}).dropna()

    def _rank_ic(g):
        if len(g) < 5:
            return np.nan
        return g["pred"].rank().corr(g["label"].rank())

    naive = df.groupby(level="date").apply(_rank_ic).dropna()
    pd.testing.assert_series_equal(fast, naive, check_names=False, atol=1e-12)


def test_predictions_to_weights_matches_naive_loop():
    preds, _ = _oos_preds_and_panel()
    fast = backtest.predictions_to_weights(preds, quantile=0.1, rebalance_every=21)

    wide = preds.unstack("ticker")
    rebal_dates = wide.index[::21]
    naive = pd.DataFrame(0.0, index=rebal_dates, columns=wide.columns)
    for d in rebal_dates:
        row = wide.loc[d].dropna()
        if len(row) < 10:
            continue
        k = max(1, int(len(row) * 0.1))
        naive.loc[d, row.nlargest(k).index] = 0.5 / k
        naive.loc[d, row.nsmallest(k).index] = -0.5 / k

    pd.testing.assert_frame_equal(fast, naive, atol=1e-15)


def test_weights_handle_sparse_dates_like_naive():
    # Dates with < 10 valid predictions must produce all-zero weights.
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=4)
    tickers = [f"T{i}" for i in range(20)]
    wide = pd.DataFrame(rng.standard_normal((4, 20)), index=dates, columns=tickers)
    wide.iloc[2, 5:] = np.nan  # only 5 valid names on the third date
    preds = wide.stack()
    preds.index.names = ["date", "ticker"]
    w = backtest.predictions_to_weights(preds, rebalance_every=1)
    assert (w.loc[dates[2]] == 0.0).all()
    assert (w.loc[dates[0]].abs().sum() - 1.0) < 1e-9
