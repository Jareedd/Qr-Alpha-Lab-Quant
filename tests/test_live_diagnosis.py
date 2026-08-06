"""Known-answer tests for the §5.0 live beta-leak diagnosis machinery.

Every function gets a case where the right answer is known by construction:
planted beta must be recovered, a zero execution gap must read as zero, a
one-day injected divergence must localize to that day, and a book that went
through risk.beta_neutralize_weights must show ~0 ex-ante beta under the
mirrored computation (the pin that the mirror is faithful).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import risk
from quantlab.live_diagnosis import (
    beta_decomposition,
    broker_returns,
    exante_beta_path,
    execution_gap,
    rolling_beta_series,
    summaries_frame,
)


def _dates(n, start="2026-01-02"):
    return pd.bdate_range(start, periods=n)


# ---------------------------------------------------------------------------
# summaries_frame / broker_returns
# ---------------------------------------------------------------------------

def test_summaries_frame_counts_and_side_split():
    records = [
        {"asof": "2026-07-02", "equity": 1_000_000.0, "orders_sent": 100,
         "n_failed": 0, "orders_failed": []},
        {"asof": "2026-07-01", "equity": 990_000.0, "orders_sent": 46,
         "n_failed": 49,
         "orders_failed": [{"side": "sell", "error": "x"}] * 19
         + [{"side": "buy", "error": "y"}]},
    ]
    df = summaries_frame(records)
    assert list(df.index) == [pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-02")]
    row = df.loc["2026-07-01"]
    # n_failed is the true count; the side split only covers the retained list
    assert row["n_failed"] == 49
    assert row["failed_sells"] == 19 and row["failed_buys"] == 1


def test_broker_returns_exact_arithmetic():
    eq = pd.Series([100.0, 110.0, 99.0], index=_dates(3))
    r = broker_returns(eq)
    assert np.isclose(r.iloc[0], 0.10)
    assert np.isclose(r.iloc[1], -0.10)
    assert len(r) == 2


# ---------------------------------------------------------------------------
# beta_decomposition / rolling_beta_series
# ---------------------------------------------------------------------------

def test_beta_decomposition_recovers_planted_beta():
    rng = np.random.default_rng(7)
    idx = _dates(500)
    mkt = pd.Series(rng.normal(0, 0.01, 500), index=idx)
    noise = pd.Series(rng.normal(0, 0.002, 500), index=idx)
    strat = 0.30 * mkt + noise
    d = beta_decomposition(strat, mkt)
    assert abs(d["beta"] - 0.30) < 0.03
    # planted market variance share: b^2 var_m / (b^2 var_m + var_e)
    expected_share = (0.30**2 * 0.01**2) / (0.30**2 * 0.01**2 + 0.002**2)
    assert abs(d["mkt_var_share"] - expected_share) < 0.05
    assert d["n_obs"] == 500


def test_beta_decomposition_independent_stream_is_flat():
    rng = np.random.default_rng(11)
    idx = _dates(400)
    mkt = pd.Series(rng.normal(0, 0.01, 400), index=idx)
    strat = pd.Series(rng.normal(0, 0.004, 400), index=idx)
    d = beta_decomposition(strat, mkt)
    assert abs(d["beta"]) < 3 * d["beta_se_iid"] + 0.05
    assert d["mkt_var_share"] < 0.05


def test_beta_decomposition_refuses_tiny_sample():
    idx = _dates(2)
    s = pd.Series([0.01, -0.01], index=idx)
    with pytest.raises(ValueError):
        beta_decomposition(s, s)


def test_rolling_beta_series_matches_manual_window():
    rng = np.random.default_rng(3)
    idx = _dates(60)
    mkt = pd.Series(rng.normal(0, 0.01, 60), index=idx)
    strat = 0.5 * mkt + pd.Series(rng.normal(0, 0.003, 60), index=idx)
    rb = rolling_beta_series(strat, mkt, window=15)
    last = pd.concat([strat, mkt], axis=1, keys=["s", "m"]).iloc[-15:]
    manual = np.cov(last["s"], last["m"], ddof=1)[0, 1] / last["m"].var(ddof=1)
    assert np.isclose(rb.iloc[-1], manual)
    assert len(rb) == 60 - 15 + 1


# ---------------------------------------------------------------------------
# execution_gap
# ---------------------------------------------------------------------------

def test_execution_gap_zero_when_identical_and_localizes_divergence():
    idx = _dates(10)
    r = pd.Series(np.linspace(-0.002, 0.002, 10), index=idx)
    same = execution_gap(r, r)
    assert np.allclose(same["gap"], 0.0)

    broker = r.copy()
    broker.iloc[4] += 0.02  # one bad execution day
    df = execution_gap(broker, r)
    assert np.isclose(df["gap"].iloc[4], 0.02)
    off_day = df["gap"].drop(df.index[4])
    assert np.allclose(off_day, 0.0)
    # cumulative gap persists after the divergence day
    assert df["cum_gap"].iloc[-1] > 0.015


# ---------------------------------------------------------------------------
# exante_beta_path — the mirror pin against risk.beta_neutralize_weights
# ---------------------------------------------------------------------------

def test_exante_beta_of_projected_book_is_zero():
    rng = np.random.default_rng(21)
    n_days, n_names = 300, 30
    idx = _dates(n_days)
    cols = [f"T{i:02d}" for i in range(n_names)]
    mkt = pd.Series(rng.normal(0, 0.01, n_days), index=idx)
    loadings = rng.uniform(0.5, 1.5, n_names)
    rets = pd.DataFrame(
        np.outer(mkt, loadings) + rng.normal(0, 0.01, (n_days, n_names)),
        index=idx, columns=cols,
    )
    betas = risk.rolling_beta(rets, mkt, window=120, min_periods=60)

    raw = pd.Series(rng.normal(0, 0.01, n_names), index=cols)
    raw -= raw.mean()
    wdf = pd.DataFrame([raw], index=[idx[-1]])
    projected = risk.beta_neutralize_weights(wdf, betas)

    path = exante_beta_path({idx[-1]: projected.iloc[0]}, betas)
    assert len(path) == 1
    assert abs(path.iloc[0]) < 1e-10  # projection removed w·beta exactly

    # sanity: the UNprojected book was NOT beta-flat under the same betas
    naive = exante_beta_path({idx[-1]: raw}, betas)
    assert abs(naive.iloc[0]) > 1e-4
