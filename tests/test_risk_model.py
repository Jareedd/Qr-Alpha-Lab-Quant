"""Execution/risk engine — factor-neutral risk model known-answer tests."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import risk_model as rm


def test_rolling_market_beta_recovers_known_beta():
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(0)
    mkt = pd.Series(rng.standard_normal(400) * 0.01, index=idx)
    assets = pd.DataFrame({"A": 2.0 * mkt, "B": 0.5 * mkt})  # exact betas 2.0, 0.5
    betas = rm.rolling_market_beta(assets, mkt, lookback=252, min_periods=126)
    assert betas["A"].iloc[-1] == pytest.approx(2.0, abs=1e-9)
    assert betas["B"].iloc[-1] == pytest.approx(0.5, abs=1e-9)


def test_neutralize_zeroes_net_factor_exposure():
    loadings = pd.DataFrame({"beta": [2.0, 1.0, 0.5]}, index=["A", "B", "C"])
    w = pd.Series({"A": 0.5, "B": -0.3, "C": -0.2})         # net beta = 0.6
    assert rm.net_factor_exposure(w, loadings)["beta"] == pytest.approx(0.6)
    wn = rm.neutralize_weights(w, loadings)
    assert rm.net_factor_exposure(wn, loadings)["beta"] == pytest.approx(0.0, abs=1e-12)


def test_neutralize_two_factors_and_dollar_neutral():
    # beta + a ones column => zero net beta AND zero net dollar exposure
    loadings = pd.DataFrame({"beta": [1.5, 0.8, 1.2, 0.4],
                             "dollar": [1.0, 1.0, 1.0, 1.0]},
                            index=list("ABCD"))
    w = pd.Series({"A": 0.6, "B": -0.1, "C": -0.2, "D": 0.1})
    wn = rm.neutralize_weights(w, loadings)
    exp = rm.net_factor_exposure(wn, loadings)
    assert exp["beta"] == pytest.approx(0.0, abs=1e-12)
    assert exp["dollar"] == pytest.approx(0.0, abs=1e-12)


def test_predicted_vol_quadratic_form():
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.09]], index=["A", "B"], columns=["A", "B"])
    w = pd.Series({"A": 0.5, "B": -0.5})
    # wᵀΣw = .25*.04 + .25*.09 = .0325 ; per-period vol = sqrt(.0325)
    assert rm.predicted_vol(w, cov, periods=1) == pytest.approx(np.sqrt(0.0325))
    assert rm.predicted_vol(w, cov, periods=252) == pytest.approx(np.sqrt(0.0325 * 252))


def test_sample_covariance_shrinkage_endpoints():
    rng = np.random.default_rng(1)
    r = pd.DataFrame(rng.standard_normal((500, 3)) * 0.01, columns=list("ABC"))
    raw = rm.sample_covariance(r, shrinkage=0.0)
    full = rm.sample_covariance(r, shrinkage=1.0)
    assert raw.equals(r.cov())                                  # δ=0 is sample cov
    off = full.to_numpy()[~np.eye(3, dtype=bool)]
    assert np.allclose(off, 0.0)                                # δ=1 zeros off-diagonals
    assert np.allclose(np.diag(full.to_numpy()), np.diag(r.cov().to_numpy()))
    with pytest.raises(ValueError):
        rm.sample_covariance(r, shrinkage=1.5)
