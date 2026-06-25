"""H1 graded run (run_h1_trial._run_trial) — offline known-answer / regression.

A no-network stub FundamentalsSource drives the WHOLE registered two-arm graded
run, pinning the construction the 2026-06-16 / 2026-06-24 amendments freeze and
guarding the NO-GO blockers that were fixed:

  * universe EXCLUDES GICS Financials AND Real Estate (the registered cut);
  * BOTH arms use the VALUE-WEIGHTED quintile book (quality_weights_vw), never EW;
  * NEUTRAL loading panel is NOT all-NaN — B3 regression (the bug where the
    quarterly grid was resampled to monthly -> all-NaN HML betas -> NEUTRAL==RAW);
  * the printed verdict is EXACTLY the 4-gate conjunction (no hidden gate);
  * the EW + 12-1 momentum baselines are finite on pandas >= 2.2 — B2 regression
    (DataFrame.mean has no min_count; the old call crashed).

The stub provides BOTH prices() (quarterly grid) and prices_monthly() (genuine
month-end grid) and a value-ORTHOGONAL planted quality alpha. NO socket opens:
fetch_sp500_tables / sector_map are monkeypatched. FF HML is the real local file
(the registered live path); the test skips if it is absent (CI / fresh clone).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from quantlab import fundamentals as fnd
from quantlab import metrics
from quantlab import universe as uni
import run_h1_trial as rht


# --- a small PIT world: 20 names, 2 of which are in EXCLUDED sectors -------- #
# Enough non-excluded names (18) that int(18 * 0.2) = 3 >= 2 forms real quintiles
# (the registration's ~396 names -> ~40-79/quintile; the test only needs >= 2).

_KEEP = [f"K{i:02d}" for i in range(18)]
_EXCL = ["FIN1", "RE1"]
_TICKERS = _KEEP + _EXCL
_SECTORS = {t: "Information Technology" for t in _KEEP}
_SECTORS["FIN1"] = "Financials"        # excluded
_SECTORS["RE1"] = "Real Estate"        # excluded
_START = "2010-01-01"
_END = "2019-12-31"

# Annual filing dates (one 10-K per year) and the real month-end grid.
_FILINGS = pd.to_datetime([f"{y}-03-15" for y in range(2010, 2020)])
_MONTHS = pd.date_range(_START, _END, freq="ME")


def _fake_tables():
    current = pd.DataFrame(
        {"ticker": _TICKERS, "sector": [_SECTORS[t] for t in _TICKERS]}
    )
    changes = pd.DataFrame(columns=["date", "added", "removed", "reason"])
    return current, changes


class _StubSource:
    """Survivorship-safe stub with synthetic CBOP components + a value-ORTHOGONAL
    planted quality alpha, and both quarterly prices() and monthly prices_monthly().
    Each NON-excluded name has a persistent quality level driving BOTH its CBOP/A
    (via the field_series flow numerators) and its forward return (so the quality
    book earns) — but the returns carry NO value-factor tilt, so the NEUTRAL arm
    should survive (value-orthogonal world)."""

    survivorship_safe = True
    start = _START
    end = _END

    def __init__(self):
        rng = np.random.default_rng(7)
        names = [t for t in _TICKERS if _SECTORS[t] not in ("Financials", "Real Estate")]
        # persistent quality per non-excluded name; excluded names get fundamentals
        # too (they must be dropped by the SECTOR filter, not by missing data).
        self._q = {t: float(q) for t, q in zip(_TICKERS, rng.standard_normal(len(_TICKERS)))}
        # monthly returns: a small per-name drift proportional to quality + noise.
        # value-ORTHOGONAL: no shared value factor injected.
        drift = np.array([0.004 * self._q[t] for t in _TICKERS])
        noise = rng.standard_normal((len(_MONTHS), len(_TICKERS))) * 0.03
        rets = drift[None, :] + noise
        self._monthly = pd.DataFrame(
            100 * np.exp(np.cumsum(rets, axis=0)), index=_MONTHS, columns=_TICKERS
        )

    # -- FundamentalsSource interface ------------------------------------- #
    def field_series(self, ticker, field, *, annual_only=False):
        q = self._q.get(ticker)
        if q is None:
            return pd.Series(dtype=float)
        # higher quality -> higher GP and CFO relative to NI (cleaner earnings).
        assets = 1000.0
        gp = 200.0 + 80.0 * q
        ni = 80.0 + 10.0 * q
        cfo = 70.0 + 30.0 * q          # CFO rises with quality -> CBOP rises
        shares = 100.0
        val = {"assets": assets, "gross_profit": gp, "net_income": ni,
               "cfo": cfo, "shares": shares, "revenue": 400.0, "cogs": 200.0 - 80 * q}
        if field not in val:
            return pd.Series(dtype=float)
        # constant-through-time annual series indexed by filing date.
        return pd.Series([val[field]] * len(_FILINGS), index=_FILINGS, name="value")

    def universe(self):
        return list(_TICKERS)

    def prices(self, universe, asof):
        cols = [c for c in self._monthly.columns if c in set(universe)]
        return self._monthly[cols].reindex(asof, method="ffill")

    def prices_monthly(self, universe):
        cols = [c for c in self._monthly.columns if c in set(universe)]
        return self._monthly[cols]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(uni, "fetch_sp500_tables", lambda *a, **k: _fake_tables())
    if not os.path.exists(rht.HML_FILE):
        pytest.skip("FF 5-factor monthly file absent; graded NEUTRAL run is live-path only")
    return _StubSource()


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_registered_constants_pinned():
    """Pin the frozen knobs so a silent edit to the construction trips a test."""
    assert rht.EXCLUDED_SECTORS == ("Financials", "Real Estate")
    assert rht.QUANTILE == 0.2
    assert rht.REBALANCE_FREQ == "BQE"
    assert rht.COST_BPS_PER_SIDE == 10.0
    # quarterly book annualizer (B4): sqrt(4), NOT the monthly-gate 12.
    assert rht.PERIODS_PER_YEAR == 4


def test_graded_run_excludes_financials_and_real_estate(patched):
    out = rht._run_trial(patched, n_trials=12)
    assert "FIN1" not in out["universe"]
    assert "RE1" not in out["universe"]
    # exactly the non-excluded names survive.
    assert set(out["universe"]) == set(_KEEP)


def test_both_arms_use_value_weighted_quintiles(patched, monkeypatch):
    """Both arms (and the lagged legs + momentum) route through quality_weights_vw,
    never the equal-weight quality_weights."""
    vw_calls = {"n": 0}
    ew_calls = {"n": 0}
    real_vw = fnd.quality_weights_vw
    real_ew = fnd.quality_weights

    def _spy_vw(*a, **k):
        vw_calls["n"] += 1
        return real_vw(*a, **k)

    def _spy_ew(*a, **k):
        ew_calls["n"] += 1
        return real_ew(*a, **k)

    monkeypatch.setattr(fnd, "quality_weights_vw", _spy_vw)
    monkeypatch.setattr(fnd, "quality_weights", _spy_ew)
    rht._run_trial(patched, n_trials=12)
    assert vw_calls["n"] > 0           # VW book engaged
    assert ew_calls["n"] == 0          # EW book NEVER used for the graded arms


def test_neutral_loading_panel_not_all_nan(patched):
    """B3 regression: the assembled HML loading panel must carry real estimated
    betas (NOT all-NaN), so the NEUTRAL arm actually neutralizes."""
    out = rht._run_trial(patched, n_trials=12)
    loading = out["loading"]
    assert out["loading_mode"] == "hml_monthly"
    assert loading is not None
    assert loading.notna().to_numpy().sum() > 0
    # and the NEUTRAL arm is genuinely distinct from RAW (not a degenerate demean
    # that would make NEUTRAL == RAW).
    assert not out["raw"]["net"].equals(out["neutral"]["net"])


def test_verdict_is_the_four_gate_conjunction(patched):
    """The reported graduate flag is EXACTLY t_NW>=2 AND beats-both-baselines AND
    DSR>=0.95 AND PBO<=0.5 — no hidden gate, no missing gate."""
    out = rht._run_trial(patched, n_trials=12)
    g = out["gates"]
    expected = g["t_nw"] and g["beats_baselines"] and g["dsr"] and g["pbo"]
    assert out["graduate"] == bool(expected)


def test_baselines_finite_on_pandas_2(patched):
    """B2 regression: the EW + 12-1 momentum baselines must compute (no
    DataFrame.mean(min_count=...) crash) and yield finite Sharpes."""
    out = rht._run_trial(patched, n_trials=12)
    assert np.isfinite(out["sr_ew"])
    assert np.isfinite(out["sr_mom"])


def test_pbo_family_is_four_config_when_lagged_leg_present(patched):
    """B5: with the lagged-assets leg present the PBO family is the 4-config matrix
    (raw/neutral x current/lagged), not a degenerate 2-config one."""
    out = rht._run_trial(patched, n_trials=12)
    assert out["pbo_blocked"] is False
    assert out["pbo"] is not None
    assert out["pbo"]["n_configs"] == 4


def test_momentum_baseline_is_12m_minus_1m_not_12_quarter():
    """M1 regression — the load-bearing baseline-correctness pin.

    The registered baseline is 12-MONTH-minus-1-month momentum (trailing 12m
    return skipping the most recent month) computed on the GENUINE monthly grid.
    The NO-GO bug was 12 shifts on the QUARTERLY rebalance grid = a 12-QUARTER
    (~3-year) lookback — a different object that, on this project's quarterly
    book, also loses the first ~3 years of P&L. ``test_baselines_finite_*`` would
    NOT catch that regression (a 12-quarter book is still finite), so this pins
    the lookback semantics directly.

    Construction: a controlled monthly world where the 12m-1m ranking is the
    EXACT REVERSE of the 12-quarter ranking at an early rebalance, so the two
    formulae trade OPPOSITE books. We assert (a) the book is non-empty from the
    first quarter ~12 months in (a 12-quarter lookback cannot trade there at all
    — it needs 13 quarters) and (b) the sign of an early-period P&L matches the
    12m-1m book, not the 12-quarter one."""
    start, end = "2010-01-01", "2014-12-31"
    months = pd.date_range(start, end, freq="ME")
    asof = pd.bdate_range(start, end, freq=rht.REBALANCE_FREQ)

    # 12 names (so int(12 * QUANTILE)=2 forms real quintiles; with <10 names the
    # book is flat every date and the test would be vacuous). Build cumulative
    # monthly prices with a drift sign that FLIPS after month 13, so a 12-MONTH
    # lookback (which can measure from ~month 13) and a 12-QUARTER lookback (which
    # cannot measure until quarter 13, ~3.25yr in) disagree on the early book.
    n = 12
    names = [f"N{i:02d}" for i in range(n)]
    rng = np.random.default_rng(3)
    px = pd.DataFrame(index=months, columns=names, dtype=float)
    early = {nm: 0.06 - 0.008 * i for i, nm in enumerate(names)}   # spread, sign +
    late = {nm: 0.00 + 0.008 * i for i, nm in enumerate(names)}     # reversed order
    for i, nm in enumerate(names):
        steps = np.where(np.arange(len(months)) < 13, early[nm], late[nm])
        steps = steps + rng.standard_normal(len(months)) * 1e-4    # tiny tie-break
        px[nm] = 100.0 * np.exp(np.cumsum(steps))

    monthly_px = px
    prices = px.reindex(asof, method="ffill")
    # equal shares -> VW weighting is immaterial to whether the book TRADES, which
    # is what this test pins (the monthly book trades in 2011-12; the 12-quarter
    # one cannot, having no lookback window yet).
    shares = pd.DataFrame(1.0, index=asof, columns=names)
    market_cap = prices * shares

    mom_net = rht.momentum_baseline(prices, asof, market_cap, monthly_px)
    assert np.isfinite(metrics.sharpe(mom_net, periods=rht.PERIODS_PER_YEAR))

    # The book must trade from ~1 year in, NOT ~3 years in. The 12-quarter bug
    # cannot produce a single non-zero weight before quarter 13 (~2013-03), so its
    # P&L series is flat across all of 2011-2012. The monthly 12-1 book trades the
    # whole window. Verified empirically: monthly -> 7/7 non-zero, 12-quarter ->
    # 0/7. So this assertion PASSES for the correct construction and FAILS under
    # the M1 regression (non-vacuous).
    early_window = mom_net.loc["2011-06-01":"2012-12-31"]
    assert len(early_window) > 0
    assert (early_window.abs() > 0).any(), (
        "no momentum P&L in 2011-2012 — the baseline is using a >12-month "
        "(likely 12-quarter) lookback, the M1 regression")


def test_compustat_slot_refuses_without_spending_a_trial():
    """A survivorship-SAFE source that does not implement the richer graded
    interface (universe/prices/prices_monthly/start/end) — the CompustatSource
    SLOT — must trigger the SOURCE-NOT-WIRED refusal (sys.exit), spending no
    trial, rather than a raw AttributeError. Covers the runner's slot-guard path."""
    from quantlab.fundamentals_data import CompustatSource

    with pytest.raises(SystemExit) as exc:
        rht._run_trial(CompustatSource(), n_trials=12)
    assert "SOURCE NOT WIRED" in str(exc.value)


def test_data_gate_refuses_survivorship_blocked_free_sec(monkeypatch):
    """DATA GATE: ``--source free_sec`` (current-ticker-only, survivorship-blocked)
    must be REFUSED before any panel assembly — re-committing trial #1's
    survivorship sin is the one thing the gate exists to stop. Covers main()'s
    source-refusal path.

    The two upstream synthetic gates (machinery + neutralization) are stubbed to a
    deterministic PASS: they are exercised by their own suites
    (test_fundamentals.test_machinery_gate_*, test_quality_value.*), and the
    neutralization gate's stochastic SR draw must not make THIS source-refusal
    assertion order-dependent. With both gates passing, reaching the DATA-GATE
    refusal is exactly the path under test — and it must still refuse free_sec."""
    monkeypatch.setattr(fnd, "machinery_gate",
                        lambda *a, **k: {"passed": True, "diffs": [9.9],
                                         "planted_sr": [9.9], "null_sr": [0.0]})
    monkeypatch.setattr(
        fnd, "neutralization_gate",
        lambda *a, **k: {"passed": True, "collapse_ok": True, "survive_ok": True,
                         "sr_matched": True, "placebo_ok": True,
                         "isvalue_neutral_sr": [0.0], "orthogonal_neutral_sr": [2.0],
                         "raw_sr_gaps": [0.0]})
    monkeypatch.setattr(sys, "argv",
                        ["run_h1_trial.py", "--source", "free_sec", "--n-trials", "12"])
    with pytest.raises(SystemExit) as exc:
        rht.main()
    assert "DATA GATE" in str(exc.value)


def test_main_runs_neutralization_gate_before_data_gate(monkeypatch):
    """B3 / amendment clause C: main() must invoke fundamentals.neutralization_gate
    AND it must run BEFORE the DATA GATE (so a survivorship refusal never short-
    circuits the integrity check). Stub the registration + machinery gates, spy the
    neutralization gate, and let the free_sec DATA GATE be the stopping point — the
    spy must have fired by then."""
    monkeypatch.setattr(rht, "require_runnable_registration", lambda *a, **k: None)
    monkeypatch.setattr(
        fnd, "machinery_gate",
        lambda *a, **k: {"passed": True, "diffs": [1.0], "planted_sr": [1.0],
                         "null_sr": [0.0]},
    )
    called = {"n": 0}
    real_ngate = fnd.neutralization_gate

    def _spy_ngate(*a, **k):
        called["n"] += 1
        return real_ngate(*a, **k)

    monkeypatch.setattr(fnd, "neutralization_gate", _spy_ngate)
    monkeypatch.setattr(sys, "argv", ["run_h1_trial.py", "--source", "free_sec"])
    with pytest.raises(SystemExit):
        rht.main()
    assert called["n"] == 1            # neutralization gate ran before the DATA GATE


def test_main_aborts_when_neutralization_gate_fails(monkeypatch):
    """A FAILING neutralization gate must ABORT main() (no trial spent), even with
    the registration + machinery gates passing and a survivorship-safe source named.
    Pins the B3 ABORT semantics the registration requires."""
    monkeypatch.setattr(rht, "require_runnable_registration", lambda *a, **k: None)
    monkeypatch.setattr(
        fnd, "machinery_gate",
        lambda *a, **k: {"passed": True, "diffs": [1.0], "planted_sr": [1.0],
                         "null_sr": [0.0]},
    )
    monkeypatch.setattr(
        fnd, "neutralization_gate",
        lambda *a, **k: {"passed": False, "collapse_ok": False, "survive_ok": True,
                         "sr_matched": True, "placebo_ok": True,
                         "isvalue_neutral_sr": [1.0], "orthogonal_neutral_sr": [2.0],
                         "raw_sr_gaps": [0.1]},
    )
    monkeypatch.setattr(sys, "argv", ["run_h1_trial.py", "--source", "free_xwalk"])
    with pytest.raises(SystemExit) as exc:
        rht.main()
    assert "NEUTRALIZATION GATE FAILED" in str(exc.value)
