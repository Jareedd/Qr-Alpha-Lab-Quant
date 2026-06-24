import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import risk_model as rm


def test_rolling_factor_betas_reduces_to_market_beta_K1():
    idx = pd.bdate_range("2020-01-01", periods=400)
    mkt = pd.Series(np.random.default_rng(0).standard_normal(400) * 0.01, index=idx)
    assets = pd.DataFrame({"A": 2.0 * mkt, "B": 0.5 * mkt})
    ref = rm.rolling_market_beta(assets, mkt, 252, 126)
    fb = rm.rolling_factor_betas(
        assets, pd.DataFrame({"mkt": mkt}), 252, 126, fit_intercept=True
    )
    pd.testing.assert_frame_equal(fb["mkt"], ref, atol=1e-9, check_names=False)
    assert fb["mkt"]["A"].iloc[-1] == pytest.approx(2.0, abs=1e-9)
    assert fb["mkt"]["B"].iloc[-1] == pytest.approx(0.5, abs=1e-9)


def test_rolling_factor_betas_recovers_known_loadings():
    idx = pd.bdate_range("2020-01-01", periods=500)
    rng = np.random.default_rng(11)
    Mkt = rng.standard_normal(500) * 0.01
    HML = rng.standard_normal(500) * 0.008
    asset = 1.5 * Mkt + 0.8 * HML + rng.standard_normal(500) * 0.001
    factors = pd.DataFrame({"Mkt": Mkt, "HML": HML}, index=idx)
    fb = rm.rolling_factor_betas(
        pd.DataFrame({"SYN": asset}, index=idx), factors, 252, 126
    )
    assert fb["Mkt"]["SYN"].iloc[-1] == pytest.approx(1.5, abs=2e-2)
    assert fb["HML"]["SYN"].iloc[-1] == pytest.approx(0.8, abs=2e-2)


def test_rolling_factor_betas_recovers_correlated_noisefree():
    idx = pd.bdate_range("2020-01-01", periods=600)
    rng = np.random.default_rng(3)
    f1 = rng.standard_normal(600) * 0.01
    f2 = 0.7 * f1 + 0.3 * rng.standard_normal(600) * 0.01  # correlated
    asset = 2.0 * f1 - 1.0 * f2  # noise-free
    factors = pd.DataFrame({"f1": f1, "f2": f2}, index=idx)
    fb = rm.rolling_factor_betas(
        pd.DataFrame({"A": asset}, index=idx), factors, 252, 126
    )
    assert fb["f1"]["A"].iloc[-1] == pytest.approx(2.0, abs=1e-8)
    assert fb["f2"]["A"].iloc[-1] == pytest.approx(-1.0, abs=1e-8)


def test_factor_betas_use_only_past_data():
    idx = pd.bdate_range("2015-01-01", periods=800)
    rng = np.random.default_rng(1)
    Mkt = pd.Series(rng.normal(0, 0.01, 800), index=idx)
    HML = pd.Series(rng.normal(0, 0.008, 800), index=idx)
    asset = pd.DataFrame(
        {"A": 1.2 * Mkt + 0.5 * HML + rng.normal(0, 0.003, 800)}, index=idx
    )
    factors = pd.DataFrame({"Mkt": Mkt, "HML": HML})
    full = rm.rolling_factor_betas(asset, factors)
    P = 600
    ac = asset.copy()
    ac.iloc[P:] = 99.0
    fc = factors.copy()
    fc.iloc[P:] = 99.0
    corr = rm.rolling_factor_betas(ac, fc)
    for name in factors.columns:
        pd.testing.assert_frame_equal(full[name].iloc[:P], corr[name].iloc[:P])
    # boundary off-by-one probe: loading at the LAST pre-poison date (P-1) is
    # byte-identical (its trailing window ends at P-1 < P, cannot read P).
    for name in factors.columns:
        np.testing.assert_array_equal(
            full[name].iloc[P - 1].to_numpy(), corr[name].iloc[P - 1].to_numpy()
        )


def test_cross_sectional_neutralize_zeroes_exposure():
    loadings = pd.DataFrame(
        {"value": [2.0, 1.0, 0.5], "dollar": [1.0, 1.0, 1.0]}, index=list("ABC")
    )
    s = pd.Series({"A": 0.6, "B": -0.1, "C": -0.2})
    r = rm.cross_sectional_neutralize(s, loadings)
    assert float(loadings["value"].to_numpy() @ r.to_numpy()) == pytest.approx(
        0.0, abs=1e-12
    )
    assert r.sum() == pytest.approx(0.0, abs=1e-12)


def test_loadings_at_builds_neutralizable_matrix():
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(7)
    Mkt = rng.standard_normal(400) * 0.01
    HML = rng.standard_normal(400) * 0.008
    assets = {
        f"S{i}": 1.0 * Mkt + (0.3 * i) * HML + rng.standard_normal(400) * 0.002
        for i in range(4)
    }
    ar = pd.DataFrame(assets, index=idx)
    fb = rm.rolling_factor_betas(ar, pd.DataFrame({"Mkt": Mkt, "HML": HML}, index=idx))
    L = rm.loadings_at(fb, idx[-1], add_dollar=True)
    assert L.columns.tolist() == ["Mkt", "HML", "dollar"]
    assert L.shape == (4, 3)
    w = pd.Series({c: (1.0 if i % 2 else -1.0) for i, c in enumerate(ar.columns)})
    wn = rm.neutralize_weights(w, L)
    assert rm.net_factor_exposure(wn, L).abs().max() < 1e-9


def test_factor_betas_guards():
    idx = pd.bdate_range("2020-01-01", periods=50)
    ar = pd.DataFrame({"A": np.zeros(50)}, index=idx)
    f = pd.DataFrame({"m": np.zeros(50)}, index=idx)
    with pytest.raises(ValueError):
        rm.rolling_factor_betas(ar, f, lookback=0)
    with pytest.raises(ValueError):
        rm.rolling_factor_betas(ar, f, lookback=10, min_periods=20)
