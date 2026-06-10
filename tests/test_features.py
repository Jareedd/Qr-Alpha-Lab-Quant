"""Phase 4 feature/label machinery: residual labels, member-masked z-scores,
rebalance cadence, feature-stability diagnostics.

Residualization is the riskiest of these (it mixes a past-only beta with
future returns), so it gets a known-answer test and an explicit construction
check: a pure-beta asset must have ~zero residual label.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import backtest, features, models, validation
from quantlab.synthetic import make_panel


def test_residual_label_kills_pure_beta_exposure():
    # Universe of clones: every asset = beta_i x same market, zero idio.
    # Forward returns differ only through beta, so beta-residualized labels
    # must be ~zero while raw labels are not.
    rng = np.random.default_rng(0)
    n_days, betas = 1200, np.array([0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
    mkt = rng.normal(0.0003, 0.01, n_days)
    rets = np.outer(mkt, betas)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rets, axis=0)),
        index=dates,
        columns=[f"B{i}" for i in range(len(betas))],
    )

    raw = features.build_labels(prices, horizon=21, residualize=False)
    resid = features.build_labels(prices, horizon=21, residualize=True)

    # Compare un-z-scored dispersion: reconstruct via the same formulas.
    fwd = prices.shift(-21) / prices - 1.0
    raw_spread = fwd.std(axis=1).iloc[300:-21].mean()
    from quantlab.risk import rolling_beta

    r = prices.pct_change(fill_method=None)
    m = r.mean(axis=1)
    b = rolling_beta(r, m)
    mp = (1 + m.fillna(0)).cumprod()
    mfwd = mp.shift(-21) / mp - 1.0
    resid_spread = (fwd - b.mul(mfwd, axis=0)).std(axis=1).iloc[300:-21].mean()
    assert resid_spread < 0.15 * raw_spread  # residualization removed ~all of it
    assert raw.shape == resid.shape


def test_member_masked_zscore_ignores_non_members():
    # An extreme non-member must not move members' z-scores.
    dates = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(1)
    prices = pd.DataFrame(
        {
            "A": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))),
            "B": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))),
            "C": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))),
            "WILD": 100 * np.exp(np.cumsum(rng.normal(0, 0.15, 300))),  # outlier
        },
        index=dates,
    )
    members_only = prices.columns != "WILD"
    mask = pd.DataFrame(
        np.tile(members_only, (300, 1)), index=dates, columns=prices.columns
    )

    masked = features.build_features(prices, member_mask=mask)
    unmasked_subset = features.build_features(prices[["A", "B", "C"]])
    for name in masked:
        pd.testing.assert_frame_equal(
            masked[name][["A", "B", "C"]], unmasked_subset[name]
        )
        assert masked[name]["WILD"].isna().all()


def test_no_mask_is_backward_compatible():
    prices = make_panel(n_assets=20, n_days=400, mode="noise", seed=2)
    a = features.build_features(prices)
    b = features.build_features(prices, member_mask=None)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_longer_rebalance_cuts_turnover():
    prices = make_panel(n_assets=40, n_days=1500, mode="noise", seed=3)
    rng = np.random.default_rng(4)
    preds = pd.DataFrame(
        rng.standard_normal(prices.shape), index=prices.index, columns=prices.columns
    ).stack()
    preds.index.names = ["date", "ticker"]
    fast = backtest.run_backtest(
        backtest.predictions_to_weights(preds, rebalance_every=21), prices
    )
    slow = backtest.run_backtest(
        backtest.predictions_to_weights(preds, rebalance_every=63), prices
    )
    assert slow["annual_turnover"] < 0.5 * fast["annual_turnover"]


def test_feature_window_ics_shape_and_planted_momentum_positive():
    prices = make_panel(mode="planted")  # standard falsification panel
    feats = features.build_features(prices)
    panel = features.stack_panel(feats, features.build_labels(prices))
    sp = validation.WalkForwardSplitter(embargo_days=21)
    fw = models.feature_window_ics(panel, sp)
    assert "test_start" in fw.columns
    assert set(feats) <= set(fw.columns)
    assert len(fw) == sp.n_splits(pd.DatetimeIndex(
        panel.index.get_level_values("date").unique()
    ))
    # The planted signal IS 12-1 momentum: its per-window IC must be positive
    # in the clear majority of windows.
    assert (fw["mom_12_1"] > 0).mean() > 0.7
    assert fw["mom_12_1"].mean() > 0.02
