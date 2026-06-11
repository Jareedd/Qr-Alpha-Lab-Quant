"""Risk neutralization: known-answer tests, neutrality checks, leak check.

Neutralization code is dangerous to get wrong silently -- a buggy projection
still produces plausible-looking weights. Every property claimed in risk.py
is asserted here: betas recover ground truth, projections actually zero the
exposures they claim to, gross stays fixed, and nothing reads the future.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import backtest, risk
from quantlab.synthetic import make_panel


def test_rolling_beta_recovers_known_beta():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", periods=1500)
    mkt = pd.Series(rng.normal(0, 0.01, 1500), index=dates)
    noise = rng.normal(0, 0.002, (1500, 2))
    assets = pd.DataFrame(
        {"HALF": 0.5 * mkt + noise[:, 0], "DOUBLE": 2.0 * mkt + noise[:, 1]},
        index=dates,
    )
    b = risk.rolling_beta(assets, mkt).dropna()
    assert abs(b["HALF"].mean() - 0.5) < 0.05
    assert abs(b["DOUBLE"].mean() - 2.0) < 0.1


def test_rolling_beta_uses_only_past_data():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2015-01-01", periods=800)
    mkt = pd.Series(rng.normal(0, 0.01, 800), index=dates)
    assets = pd.DataFrame({"A": 1.2 * mkt + rng.normal(0, 0.003, 800)}, index=dates)

    b_full = risk.rolling_beta(assets, mkt)
    corrupted = assets.copy()
    corrupted.iloc[600:] = 99.0  # poison the future
    b_corrupt = risk.rolling_beta(corrupted, mkt)
    pd.testing.assert_frame_equal(b_full.iloc[:600], b_corrupt.iloc[:600])


def test_sector_demean_zeroes_sector_means():
    prices = make_panel(n_assets=40, n_days=300, mode="noise", seed=2)
    sectors = prices.attrs["sectors"]
    rng = np.random.default_rng(3)
    wide = pd.DataFrame(
        rng.standard_normal(prices.shape), index=prices.index, columns=prices.columns
    )
    preds = wide.stack()
    preds.index.names = ["date", "ticker"]

    neut = risk.neutralize_predictions_by_sector(preds, sectors)
    sec = neut.index.get_level_values("ticker").map(sectors.get)
    by = neut.groupby([neut.index.get_level_values("date"), sec]).mean()
    assert by.abs().max() < 1e-12


def test_beta_projection_zeroes_exposure_keeps_neutrality_and_gross():
    prices = make_panel(n_assets=50, n_days=900, mode="noise", seed=4)
    true_betas = prices.attrs["betas"]
    rng = np.random.default_rng(5)
    wide = pd.DataFrame(
        rng.standard_normal(prices.shape), index=prices.index, columns=prices.columns
    )
    preds = wide.stack()
    preds.index.names = ["date", "ticker"]
    w = backtest.predictions_to_weights(preds)

    rets = prices.pct_change(fill_method=None)
    mkt = rets.mean(axis=1)
    betas = risk.rolling_beta(rets, mkt)
    w_n = risk.beta_neutralize_weights(w, betas)

    active = w_n[w_n.abs().sum(axis=1) > 0]
    for d, row in list(active.iterrows())[5:]:  # skip beta warm-up period
        b = betas.loc[:d].iloc[-1].reindex(row.index).fillna(1.0)
        assert abs(row.sum()) < 1e-9                      # still dollar neutral
        assert abs((row * b).sum()) < 1e-8                # ex-ante beta ~ 0
        orig_gross = w.loc[d].abs().sum()
        assert abs(row.abs().sum() - orig_gross) < 1e-9   # gross preserved

    # The projection should have moved weights (raw deciles are not beta
    # neutral by accident) but not replaced them wholesale: compare signs on
    # the originally-active cells only. The explicit .dropna() matters:
    # pandas 3 stack() keeps the NaN cells that w[w != 0] masks out, and
    # counting them as mismatches dilutes overlap to ~0.2 (active fraction
    # x true overlap) -- the comparison must be active-cells-only under
    # both stack semantics.
    sign_orig = np.sign(w[w != 0]).stack().dropna()
    sign_neut = np.sign(w_n[w != 0]).stack().dropna()
    overlap = (sign_orig == sign_neut.reindex(sign_orig.index)).mean()
    assert overlap > 0.5

    # And the true simulated betas confirm exposure dropped vs raw weights.
    tb = pd.Series(true_betas)
    raw_exp = (w * tb).sum(axis=1).abs().mean()
    neut_exp = (w_n * tb).sum(axis=1).abs().mean()
    assert neut_exp < raw_exp


def test_risk_report_flags_a_deliberately_beta_loaded_portfolio():
    prices = make_panel(n_assets=40, n_days=900, mode="noise", seed=6)
    sectors = prices.attrs["sectors"]
    rets = prices.pct_change(fill_method=None)
    mkt = rets.mean(axis=1)
    betas = risk.rolling_beta(rets, mkt)

    # Long-only portfolio: beta ~ 1 by construction, and the report must say so.
    dates = prices.index[300::21]
    w = pd.DataFrame(1.0 / prices.shape[1], index=dates, columns=prices.columns)
    daily_w = w.reindex(prices.index).ffill().shift(1).fillna(0.0)
    port_rets = (daily_w * rets).sum(axis=1)

    rep = risk.risk_report(port_rets.loc[dates[0]:], mkt, daily_w, betas, sectors)
    assert rep["market_corr"] > 0.9
    assert 0.7 < rep["realized_beta_mean"] < 1.3
    assert rep["n_sectors"] == len(set(sectors.values()))
