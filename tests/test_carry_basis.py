"""Known-answer SIGN-CONVENTION pins for quantlab.carry_basis.

CLAUDE.md: the #1 risk on this repo is a sign error that manufactures a fake
edge. These tests pin the cash-and-carry sign conventions on hand-built series
with hand-computed answers, so any future flip is caught immediately:

  (a) positive funding + flat basis  -> short collects POSITIVE net (minus cost)
  (b) basis WIDENS against the short -> mark-to-market LOSS
  (c) funding NEGATIVE               -> carry NEGATIVE
  (d) gated variant only holds ABOVE the hurdle

Plus: episode cost charged once per episode, rebalance cost on drift,
convergence-neutralization sign, and the annualized summary.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import carry_basis as cb


def _series(vals, start="2022-01-03", name=None):
    idx = pd.bdate_range(start, periods=len(vals))
    return pd.Series(vals, index=idx, name=name)


# ---------------------------------------------------------------------------
# (a) THE HEADLINE PIN: positive funding, flat basis, flat prices ->
#     the short collects POSITIVE gross == funding, and net == funding - cost.
# ---------------------------------------------------------------------------
def test_positive_funding_flat_basis_gives_positive_net():
    # spot == perp, both flat => r_spot = r_perp = 0 => gross == funding.
    spot = _series([100.0, 100.0, 100.0, 100.0])
    perp = _series([100.0, 100.0, 100.0, 100.0])
    funding = _series([0.0, 0.001, 0.001, 0.001])  # +10 bps/day funding

    g = cb.gross_carry_returns(spot, perp, funding)
    # gross daily == funding exactly when prices are flat (r_spot=r_perp=0)
    assert np.allclose(g["gross"].to_numpy(), [0.001, 0.001, 0.001])
    assert (g["gross"] > 0).all()  # POSITIVE: short collects positive funding

    costs = cb.CostParams(roundtrip_bps=5.0, rebalance_bps=1.25)
    res = cb.cash_and_carry_returns(spot, perp, funding, costs)
    # day 1 (first held day) pays the round-trip; later days only rebalance(=0
    # here, since drift=0). So later days' net == gross == +funding.
    assert res["net"].iloc[-1] == pytest.approx(0.001)
    # episode cost charged exactly once, on the first held day
    assert res["episode_cost"].iloc[0] == pytest.approx(5.0e-4)
    assert (res["episode_cost"].iloc[1:] == 0).all()
    # net total still POSITIVE (funding dwarfs the one-off 5bps over 3 days)
    assert cb.summarize_returns(res["net"])["total"] > 0


# ---------------------------------------------------------------------------
# (b) basis WIDENS against the short -> mark-to-market LOSS on the basis leg.
#     Perp rises relative to spot (premium widens): the short loses.
# ---------------------------------------------------------------------------
def test_basis_widening_against_short_is_a_loss():
    # spot flat at 100; perp climbs 100->101->102 (premium widening). Zero
    # funding so the ONLY P&L is the basis leg. A short-perp/long-spot book
    # LOSES when the perp it is short rises faster than the spot it is long.
    spot = _series([100.0, 100.0, 100.0])
    perp = _series([100.0, 101.0, 102.0])
    funding = _series([0.0, 0.0, 0.0])

    g = cb.gross_carry_returns(spot, perp, funding)
    # r_spot=0, r_perp>0 => gross = -r_perp < 0
    assert (g["gross"] < 0).all()
    assert g["dbasis"].gt(0).all()  # basis (premium) is widening
    # and the loss magnitude equals the perp return (the short's MTM)
    assert g["gross"].iloc[0] == pytest.approx(-(101.0 / 100.0 - 1.0))

    costs = cb.CostParams(roundtrip_bps=0.0, rebalance_bps=0.0)  # isolate basis
    res = cb.cash_and_carry_returns(spot, perp, funding, costs)
    assert cb.summarize_returns(res["net"])["total"] < 0  # net LOSS


def test_basis_converging_helps_short():
    # The mirror of (b): perp premium DECAYS toward spot -> the short gains.
    spot = _series([100.0, 100.0, 100.0])
    perp = _series([102.0, 101.0, 100.0])  # premium collapsing
    funding = _series([0.0, 0.0, 0.0])
    g = cb.gross_carry_returns(spot, perp, funding)
    assert (g["gross"] > 0).all()          # convergence credit to the short
    assert g["dbasis"].lt(0).all()         # premium shrinking


# ---------------------------------------------------------------------------
# (c) funding NEGATIVE -> carry NEGATIVE (with flat prices the short PAYS).
# ---------------------------------------------------------------------------
def test_negative_funding_gives_negative_carry():
    spot = _series([100.0, 100.0, 100.0])
    perp = _series([100.0, 100.0, 100.0])
    funding = _series([0.0, -0.002, -0.002])  # negative funding: short PAYS
    g = cb.gross_carry_returns(spot, perp, funding)
    assert (g["gross"] < 0).all()
    assert g["gross"].iloc[0] == pytest.approx(-0.002)
    res = cb.cash_and_carry_returns(spot, perp, funding,
                                    cb.CostParams(0.0, 0.0))
    assert cb.summarize_returns(res["net"])["total"] < 0


# ---------------------------------------------------------------------------
# (d) the funding-gated variant only HOLDS above the 3x hurdle.
# ---------------------------------------------------------------------------
def test_gated_variant_only_holds_above_hurdle():
    costs = cb.CostParams(roundtrip_bps=5.0, rebalance_bps=0.0)
    hurdle = 3.0 * costs.roundtrip          # 3 * 5bps = 15 bps over 3 days
    # Construct funding so the trailing-3d SUM is below the hurdle early and
    # clearly above it late.
    low = 0.0001    # 3-day sum 3bps  < 15bps  -> FLAT
    high = 0.010    # 3-day sum 300bps > 15bps -> IN
    funding = _series([low] * 6 + [high] * 6)
    spot = _series([100.0] * 12)
    perp = _series([100.0] * 12)

    held = cb.funding_gated_held(spot, perp, funding, costs,
                                 hurdle_mult=3.0, window=3)
    # The rolling-3d sum only clears 15bps once the high days fill the window.
    assert not held.iloc[:6].any()          # never in during the low regime
    assert held.iloc[-1]                     # in during the high regime

    res = cb.funding_gated_episodes(spot, perp, funding, costs,
                                    hurdle_mult=3.0, window=3)
    # P&L is zero on flat days, positive (funding) on held days.
    flat_net = res.loc[~res["held"], "net"]
    assert (flat_net == 0).all()
    assert res.loc[res["held"], "gross"].gt(0).all()
    # exactly one contiguous held episode here -> exactly one round-trip charge
    assert cb.episode_count(res["held"]) == 1
    assert (res["episode_cost"] > 0).sum() == 1


def test_gated_episode_cost_charged_once_per_episode():
    # Two separated high-funding blocks => two episodes => two round-trips.
    costs = cb.CostParams(roundtrip_bps=10.0, rebalance_bps=0.0)
    hi, lo = 0.02, 0.0
    funding = _series([hi] * 5 + [lo] * 5 + [hi] * 5)
    spot = _series([100.0] * 15)
    perp = _series([100.0] * 15)
    res = cb.funding_gated_episodes(spot, perp, funding, costs,
                                    hurdle_mult=3.0, window=3)
    assert cb.episode_count(res["held"]) == 2
    assert (res["episode_cost"] > 0).sum() == 2


# ---------------------------------------------------------------------------
# rebalance cost scales with leg drift |r_spot - r_perp|.
# ---------------------------------------------------------------------------
def test_rebalance_cost_tracks_leg_drift():
    spot = _series([100.0, 100.0, 100.0])
    perp = _series([100.0, 101.0, 101.0])  # day1 perp +1% => drift ~1%
    funding = _series([0.0, 0.0, 0.0])
    costs = cb.CostParams(roundtrip_bps=0.0, rebalance_bps=10.0)  # 10 bps
    res = cb.cash_and_carry_returns(spot, perp, funding, costs)
    drift_day1 = abs(0.0 - (101.0 / 100.0 - 1.0))
    assert res["rebalance_cost"].iloc[0] == pytest.approx(10.0e-4 * drift_day1)


# ---------------------------------------------------------------------------
# convergence_neutralization sign: a fully-priced world (funding exactly
# offset by an adverse basis widening) reports ~100% neutralized.
# ---------------------------------------------------------------------------
def test_convergence_neutralization_priced_world():
    # Build a PRICED world: each day the short collects funding f, but the
    # premium widens by exactly f (dbasis = +f), so basis P&L = -f cancels it.
    # Then -(... ) bookkeeping: cum_db == +f over horizon=0 => frac == 1.0.
    f = 0.001
    n = 30
    funding = _series([f] * n)
    # basis grows by f each day: B_t = f * t
    basis_vals = [f * t for t in range(n)]
    basis_series = _series(basis_vals)
    out = cb.convergence_neutralization(funding, basis_series, horizon=0)
    # dB_t = f each day; fraction neutralized = dB/f = 1.0
    assert out["median_frac_neutralized"] == pytest.approx(1.0, abs=1e-9)
    assert out["n"] > 0


def test_convergence_neutralization_unpriced_world():
    # A world where funding is positive but basis is FLAT (no convergence): the
    # funding is NOT neutralized -> fraction ~ 0 -> NOT disqualified.
    f = 0.001
    n = 30
    funding = _series([f] * n)
    basis_series = _series([0.05] * n)  # constant premium, dbasis == 0
    out = cb.convergence_neutralization(funding, basis_series, horizon=0)
    assert out["median_frac_neutralized"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# annualized summary: Sharpe uses 365 periods; total is cumulative net.
# ---------------------------------------------------------------------------
def test_summarize_returns_annualization():
    net = _series([0.001] * 365)
    s = cb.summarize_returns(net, periods=365)
    assert s["n_days"] == 365
    assert s["total"] == pytest.approx(0.365)
    assert s["ann_return"] == pytest.approx(365 * 0.001)
    # constant series => zero vol => sharpe is nan (no risk), not inf
    assert np.isnan(s["sharpe"])


def test_summarize_sharpe_positive_for_positive_noisy_series():
    rng = np.random.default_rng(0)
    net = _series(0.001 + 0.0005 * rng.standard_normal(400))
    s = cb.summarize_returns(net, periods=365)
    assert s["sharpe"] > 0


# ---------------------------------------------------------------------------
# synthetic helper sanity: flat-basis world => gross == funding.
# ---------------------------------------------------------------------------
def test_make_synthetic_carry_flat_basis_gross_equals_funding():
    w = cb.make_synthetic_carry(n_days=20, daily_funding=0.0007,
                                basis_drift=0.0, spot_drift=0.0)
    g = cb.gross_carry_returns(w["spot"], w["perp"], w["funding"])
    assert np.allclose(g["gross"].to_numpy(), 0.0007)


# ---------------------------------------------------------------------------
# deploy_signal: the dormant-but-armed DEPLOY gate (decision-support).
# ---------------------------------------------------------------------------
def test_deploy_signal_high_funding_deploys():
    # ~18%/yr gross funding clears BOTH the 3x cost gate AND risk-free+tail-buffer.
    funding = _series([0.0005] * 60, name="funding")
    costs = {"X": cb.BUCKET_COSTS["major"]}
    r = cb.deploy_signal({"X": funding}, costs, risk_free=0.05,
                         tail_buffer=0.10, window=30)["X"]
    assert r["deploy"] is True
    assert r["cost_gate"] is True
    assert r["net_ann"] > 0.15            # beats rf + buffer
    assert r["excess_over_rf"] > 0.10


def test_deploy_signal_decayed_to_cash_stays_flat():
    # ~5%/yr gross funding ~= the risk-free rate: the cost gate passes, but the
    # tail-buffer gate does NOT -> FLAT. This is the dormant 2025-26 behavior and
    # the whole point of the rule: a yield that merely matches T-bills is not worth
    # the tail, so the binding gate is the risk-free+buffer, not the cost gate.
    funding = _series([0.00014] * 60, name="funding")
    costs = {"X": cb.BUCKET_COSTS["major"]}
    r = cb.deploy_signal({"X": funding}, costs, risk_free=0.05,
                         tail_buffer=0.10, window=30)["X"]
    assert r["deploy"] is False
    assert r["cost_gate"] is True         # cost gate alone would pass...
    assert r["excess_over_rf"] < 0.10     # ...but the tail buffer binds -> FLAT


def test_deploy_signal_below_cost_gate_stays_flat():
    funding = _series([0.000005] * 60, name="funding")   # ~0.18%/yr
    costs = {"X": cb.BUCKET_COSTS["major"]}
    r = cb.deploy_signal({"X": funding}, costs, window=30)["X"]
    assert r["deploy"] is False
    assert r["cost_gate"] is False


def test_deploy_signal_negative_funding_is_flat():
    funding = _series([-0.0002] * 60, name="funding")
    costs = {"X": cb.BUCKET_COSTS["major"]}
    r = cb.deploy_signal({"X": funding}, costs, window=30)["X"]
    assert r["deploy"] is False
    assert r["net_ann"] < 0


def test_deploy_signal_insufficient_history_is_flat():
    funding = _series([0.0005] * 10, name="funding")     # < window=30
    costs = {"X": cb.BUCKET_COSTS["major"]}
    r = cb.deploy_signal({"X": funding}, costs, window=30)["X"]
    assert r["deploy"] is False
    assert r["reason"] == "insufficient_history"
