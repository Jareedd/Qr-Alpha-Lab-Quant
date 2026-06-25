"""H1 PRE-REGISTERED construction — offline known-answer tests.

Pins the machinery the 2026-06-16 / 2026-06-24 amendments freeze:
  * CBOP/A arithmetic (profitability NET of accruals) and the cbop_signal z-score;
  * value-weighted quintile L/S — dollar-neutral, value-weighted (bigger-cap name
    in a side carries more weight), and reducing to equal-weight when caps match;
  * the HML-loading alignment is PAST-ONLY (poison-the-future pin) on synthetic
    factors.

No network: synthetic / stub data only.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import fundamentals as fnd
from quantlab import fundamentals_data as fdat
from quantlab import risk_model as rm


# --- FIELD_TAGS: shares added for value-weighting --------------------------- #

def test_shares_field_tag_registered():
    assert "shares" in fdat.FIELD_TAGS
    assert fdat.FIELD_TAGS["shares"] == [
        "CommonStockSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "CommonStockSharesIssued",
    ]


# --- CBOP/A arithmetic (profitability net of accruals) ---------------------- #

def test_cbop_over_assets_arithmetic():
    d = pd.to_datetime(["2023-02-15"])
    gp = pd.Series([40.0], index=d)
    ni = pd.Series([16.0], index=d)
    cfo = pd.Series([10.0], index=d)
    a = pd.Series([100.0], index=d)
    # CBOP = GP - (NI - CFO) = 40 - (16 - 10) = 34; /A = 0.34
    out = fnd.cbop_over_assets(gp, ni, cfo, a)
    assert out.iloc[0] == pytest.approx(0.34)


def test_cbop_subtracts_accruals_not_adds():
    """A more accrual-heavy firm (NI >> CFO) must have LOWER CBOP/A — the
    accruals leg is SUBTRACTED (subsumed), not added."""
    d = pd.to_datetime(["2023-02-15"])
    gp = pd.Series([40.0], index=d)
    cfo = pd.Series([10.0], index=d)
    a = pd.Series([100.0], index=d)
    clean = fnd.cbop_over_assets(gp, pd.Series([10.0], index=d), cfo, a).iloc[0]  # NI=CFO -> 0 accruals
    dirty = fnd.cbop_over_assets(gp, pd.Series([25.0], index=d), cfo, a).iloc[0]  # NI>>CFO -> +accruals
    assert dirty < clean
    assert clean == pytest.approx(0.40)          # 40 - 0
    assert dirty == pytest.approx(0.25)          # 40 - 15


def test_cbop_over_assets_pit_ffill_and_division():
    """Filing-date PIT: union index ffilled; a later assets-only filing reuses the
    last known flow numerators (no lookahead, division uses the freshest assets)."""
    gp = pd.Series([40.0], index=pd.to_datetime(["2023-02-15"]))
    ni = pd.Series([16.0], index=pd.to_datetime(["2023-02-15"]))
    cfo = pd.Series([10.0], index=pd.to_datetime(["2023-02-15"]))
    a = pd.Series([100.0, 200.0], index=pd.to_datetime(["2023-02-15", "2023-08-15"]))
    out = fnd.cbop_over_assets(gp, ni, cfo, a)
    assert out.loc["2023-02-15"] == pytest.approx(0.34)
    assert out.loc["2023-08-15"] == pytest.approx(34.0 / 200.0)


# --- cbop_signal: cross-sectional z-score ----------------------------------- #

def test_cbop_signal_is_cross_sectional_zscore():
    d = pd.to_datetime(["2023-02-15"])
    cbop_a = pd.DataFrame({"A": [0.30], "B": [0.20], "C": [0.10]}, index=d)
    sig = fnd.cbop_signal(cbop_a)
    row = sig.loc[d[0]]
    # ordering preserved, demeaned (sum ~ 0), unit-ish scaled
    assert row["A"] > row["B"] > row["C"]
    assert row.sum() == pytest.approx(0.0, abs=1e-9)
    # matches a manual z using pandas' cross-sectional std (ddof=1, the
    # _zscore_rows convention) with the +1e-12 guard.
    vals = pd.Series([0.30, 0.20, 0.10])
    manual = (vals - vals.mean()) / (vals.std() + 1e-12)   # ddof=1
    np.testing.assert_allclose(row.to_numpy(), manual.to_numpy(), atol=1e-6)


# --- value-weighted quintile L/S -------------------------------------------- #

def _single_date_panel(values, cols, mc=None):
    d = pd.to_datetime(["2023-02-15"])
    sig = pd.DataFrame([values], index=d, columns=cols, dtype=float)
    cap = None if mc is None else pd.DataFrame([mc], index=d, columns=cols, dtype=float)
    return d, sig, cap


def test_vw_quintile_is_dollar_neutral():
    cols = [f"F{i}" for i in range(10)]
    d, sig, cap = _single_date_panel(
        list(range(10)), cols, mc=[1.0 + i for i in range(10)]
    )
    w = fnd.quality_weights_vw(sig, cap, quantile=0.2).loc[d[0]]
    assert w.sum() == pytest.approx(0.0, abs=1e-12)
    assert w[w > 0].sum() == pytest.approx(0.5, abs=1e-12)    # long side +0.5
    assert w[w < 0].sum() == pytest.approx(-0.5, abs=1e-12)   # short side -0.5
    # quintile membership (top/bottom 2 of 10): highest-signal longs, lowest shorts
    assert w["F9"] > 0 and w["F8"] > 0
    assert w["F0"] < 0 and w["F1"] < 0
    assert (w[[f"F{i}" for i in range(2, 8)]] == 0).all()     # middle flat


def test_vw_quintile_bigger_cap_gets_more_weight():
    cols = [f"F{i}" for i in range(10)]
    # F9, F8 are the long quintile; give F9 a 3x cap so it carries 3x the weight.
    caps = [1.0] * 8 + [1.0, 3.0]          # F8=1, F9=3
    d, sig, cap = _single_date_panel(list(range(10)), cols, mc=caps)
    w = fnd.quality_weights_vw(sig, cap, quantile=0.2).loc[d[0]]
    # within the long side: F9 weight / F8 weight == cap ratio 3:1
    assert w["F9"] == pytest.approx(0.5 * 3.0 / 4.0)
    assert w["F8"] == pytest.approx(0.5 * 1.0 / 4.0)
    assert w["F9"] > w["F8"]


def test_vw_reduces_to_equal_weight_when_caps_equal():
    cols = [f"F{i}" for i in range(10)]
    d, sig, cap = _single_date_panel(list(range(10)), cols, mc=[5.0] * 10)
    w_vw = fnd.quality_weights_vw(sig, cap, quantile=0.2).loc[d[0]]
    w_ew = fnd.quality_weights(sig, quantile=0.2).loc[d[0]]
    pd.testing.assert_series_equal(w_vw, w_ew, check_names=False)


def test_vw_drops_nonpositive_cap_names_and_renormalizes():
    cols = [f"F{i}" for i in range(10)]
    # F8 (a long-quintile name) has a NaN cap -> dropped; long side renormalizes
    # on F9 alone, still summing to +0.5.
    caps = [1.0] * 8 + [np.nan, 2.0]       # F8 NaN, F9=2
    d, sig, cap = _single_date_panel(list(range(10)), cols, mc=caps)
    w = fnd.quality_weights_vw(sig, cap, quantile=0.2).loc[d[0]]
    assert w["F8"] == 0.0
    assert w["F9"] == pytest.approx(0.5)
    assert w[w > 0].sum() == pytest.approx(0.5, abs=1e-12)


def _hand_built_backtest(weights, prices, signal, cost_bps_per_side):
    """Reference backtest from explicit weights, mirroring quality_backtest's P&L
    convention (weights at t earn t->t+1; turnover-cost on |Δw|)."""
    fwd = prices.pct_change(fill_method=None).shift(-1).reindex_like(signal)
    gross = (weights * fwd).sum(axis=1, min_count=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    net = (gross - turnover * cost_bps_per_side / 1e4).dropna()
    return net


def test_quality_backtest_vw_equals_hand_built_vw_weights():
    """M4: passing market_cap must produce EXACTLY the backtest of
    quality_weights_vw — not merely 'different from EW'. Pins the routing."""
    d = pd.bdate_range("2020-01-31", periods=4, freq="BME")
    cols = [f"F{i}" for i in range(10)]
    rng = np.random.default_rng(0)
    sig = pd.DataFrame(rng.standard_normal((4, 10)), index=d, columns=cols)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.standard_normal((4, 10)) * 0.05, axis=0)),
        index=d, columns=cols,
    )
    cap = pd.DataFrame(np.tile(np.arange(1.0, 11.0), (4, 1)), index=d, columns=cols)
    vw = fnd.quality_backtest(sig, prices, cost_bps_per_side=5.0, market_cap=cap)
    w_vw = fnd.quality_weights_vw(sig, cap, quantile=0.2)
    hand = _hand_built_backtest(w_vw, prices, sig, cost_bps_per_side=5.0)
    pd.testing.assert_series_equal(vw["net"], hand, check_names=False)


def test_quality_backtest_equal_caps_reproduce_ew():
    """M4: with equal caps the VW route must reproduce the EW book EXACTLY (the
    VW->EW limiting case), and must differ from EW when caps are unequal."""
    d = pd.bdate_range("2020-01-31", periods=4, freq="BME")
    cols = [f"F{i}" for i in range(10)]
    rng = np.random.default_rng(0)
    sig = pd.DataFrame(rng.standard_normal((4, 10)), index=d, columns=cols)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.standard_normal((4, 10)) * 0.05, axis=0)),
        index=d, columns=cols,
    )
    equal_cap = pd.DataFrame(5.0, index=d, columns=cols)
    ew = fnd.quality_backtest(sig, prices, cost_bps_per_side=5.0)
    vw_equal = fnd.quality_backtest(sig, prices, cost_bps_per_side=5.0,
                                    market_cap=equal_cap)
    pd.testing.assert_series_equal(ew["net"], vw_equal["net"], check_names=False)
    # and unequal caps DO diverge (VW genuinely engaged, not a no-op)
    uneq_cap = pd.DataFrame(np.tile(np.arange(1.0, 11.0), (4, 1)), index=d, columns=cols)
    vw_uneq = fnd.quality_backtest(sig, prices, cost_bps_per_side=5.0,
                                   market_cap=uneq_cap)
    assert not ew["net"].equals(vw_uneq["net"])


# --- HML-loading alignment is PAST-ONLY (poison-the-future pin) -------------- #

def test_hml_loading_is_past_only_on_synthetic_factors():
    """The NEUTRAL arm's value loading is a trailing rolling_factor_betas HML
    beta. Poisoning the FUTURE (returns + factor after date P) must NOT change any
    loading at or before P-1 — law #1, demonstrated not asserted. Mirrors the
    risk-model poison test but on the HML path the H1 runner uses."""
    idx = pd.bdate_range("2015-01-31", periods=120, freq="BME")
    rng = np.random.default_rng(7)
    hml = pd.Series(rng.normal(0.004, 0.03, 120), index=idx)
    rets = pd.DataFrame(
        {f"S{i}": (0.2 * i) * hml + rng.normal(0, 0.02, 120) for i in range(5)},
        index=idx,
    )
    factors = hml.to_frame("HML")
    full = rm.rolling_factor_betas(rets, factors, lookback=36, min_periods=18,
                                   fit_intercept=True)["HML"]
    P = 80
    rc = rets.copy(); rc.iloc[P:] = 99.0
    fc = factors.copy(); fc.iloc[P:] = 99.0
    poisoned = rm.rolling_factor_betas(rc, fc, lookback=36, min_periods=18,
                                       fit_intercept=True)["HML"]
    pd.testing.assert_frame_equal(full.iloc[:P], poisoned.iloc[:P])
    # boundary: the LAST pre-poison loading (P-1) is byte-identical (its window
    # ends at P-1 < P, cannot read the poisoned future).
    np.testing.assert_array_equal(
        full.iloc[P - 1].to_numpy(), poisoned.iloc[P - 1].to_numpy()
    )


def test_value_neutralized_cbop_has_no_accruals_leg():
    """m1: value_neutralized_cbop must residualize the CBOP panel with NO accruals
    blend — identical to value_neutralized_signal(cbop, loading, accruals_a=None),
    and DIFFERENT from a (forbidden) version that blends an accruals panel."""
    d = pd.to_datetime(["2023-02-15"])
    cbop_a = pd.DataFrame({"A": [0.30], "B": [0.20], "C": [0.10], "D": [0.05]}, index=d)
    loading = pd.DataFrame({"A": [2.0], "B": [1.0], "C": [-0.5], "D": [-1.0]}, index=d)
    explicit = fnd.value_neutralized_cbop(cbop_a, loading)
    no_acc = fnd.value_neutralized_signal(cbop_a, loading, accruals_a=None)
    pd.testing.assert_frame_equal(explicit, no_acc)
    # a blended-accruals version is a DIFFERENT object -> the wrapper is not
    # silently carrying an accruals leg.
    acc = pd.DataFrame({"A": [0.20], "B": [0.0], "C": [-0.10], "D": [0.05]}, index=d)
    blended = fnd.value_neutralized_signal(cbop_a, loading, accruals_a=acc)
    assert not explicit.equals(blended)


def test_value_neutralized_signal_zeroes_value_exposure():
    """End-to-end on the H1 NEUTRAL path: residualizing CBOP z against a known
    value loading (+ ones) leaves zero net value exposure and a demeaned signal
    that date (composes cbop_signal with value_neutralized_signal)."""
    d = pd.to_datetime(["2023-02-15"])
    cbop_a = pd.DataFrame({"A": [0.30], "B": [0.20], "C": [0.10], "D": [0.05]}, index=d)
    loading = pd.DataFrame({"A": [2.0], "B": [1.0], "C": [-0.5], "D": [-1.0]}, index=d)
    neutral = fnd.value_neutralized_signal(cbop_a, loading)
    row = neutral.loc[d[0]]
    # net value exposure zeroed, dollar-neutral (demeaned)
    assert float(loading.loc[d[0]].to_numpy() @ row.to_numpy()) == pytest.approx(
        0.0, abs=1e-9)
    assert row.sum() == pytest.approx(0.0, abs=1e-9)


# --- runner glue: hml_loading_panel is PAST-ONLY (n1 leak pin) -------------- #

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def test_hml_loading_panel_alignment_is_past_only():
    """n1: the RUNNER's hml_loading_panel glue (monthly-beta -> as-of-onto-asof)
    must be past-only. Poisoning the monthly prices AFTER a cutoff month must NOT
    change any aligned loading on/before that cutoff (law #1, demonstrated on the
    actual glue the graded run uses). Skips if the FF file is absent (CI / fresh
    clone) — the live HML path is the registered default and IS present locally."""
    import run_h1_trial as rht

    if not os.path.exists(rht.HML_FILE):
        pytest.skip("FF 5-factor monthly file absent; HML glue test is live-path only")

    # genuine monthly price grid (real month-ends so FF HML overlaps), enough
    # months for the trailing-36 / min-18 betas to estimate.
    months = pd.date_range("2012-01-31", periods=90, freq="ME")
    rng = np.random.default_rng(3)
    monthly_px = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.standard_normal((90, 6)) * 0.04, axis=0)),
        index=months, columns=[f"S{i}" for i in range(6)],
    )
    # quarterly rebalance grid spanning the same window
    asof = pd.bdate_range(months[0], months[-1], freq=rht.REBALANCE_FREQ)

    full, mode = rht.hml_loading_panel(monthly_px, asof)
    assert mode == "hml_monthly"
    assert full is not None and full.notna().to_numpy().sum() > 0  # not all-NaN

    cutoff = months[60]
    poisoned_px = monthly_px.copy()
    poisoned_px.loc[poisoned_px.index > cutoff] = 9999.0
    poisoned, _ = rht.hml_loading_panel(poisoned_px, asof)

    pre = asof[asof <= cutoff]
    pd.testing.assert_frame_equal(
        full.reindex(pre), poisoned.reindex(pre)
    )
