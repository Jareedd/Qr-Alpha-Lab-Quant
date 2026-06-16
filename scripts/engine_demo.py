"""Execution/risk ENGINE — end-to-end capstone demonstration.

    python scripts/engine_demo.py            # run the demo, write results/engine_demo.json
    python scripts/engine_demo.py --strict   # also assert every honest property (exit 1 on fail)

WHY THIS SCRIPT EXISTS. The engine (`quantlab.engine.PortfolioEngine`) is the
honest "scale it" machinery: it turns a GRADUATED edge into a sized, neutralized,
risk-limited, integer-share order list. But no strategy in this project has
graduated (N=11, zero graduations), so the engine has never been driven by a
real edge. This demo drives it on the SYNTHETIC quality world (ground-truth
planted edge vs a paired null) so the machinery can be exhibited and its central
property proven, without pretending any real alpha exists.

SYNTHETIC ONLY (law #7). `make_quality_panel` is harness data, labeled as such;
this script touches no market data, spends no trial (N unchanged), and never
submits an order. It is reproducible from the seeds below (law #8).

THE PROPERTY IT PROVES. Fed a real edge, the engine commits capital
(avg gross exposure > 0.2) and earns it net of cost; fed an identically-built
NULL, it sizes to ~zero (avg gross exposure < 0.05) — because leverage is driven
by the LOWER confidence bound of the trailing Sharpe (Lo-2002 SE), not the point
estimate. This is the deliberate counterweight to the trial-#11 over-confidence:
the engine refuses to lever an edge it is not statistically sure of.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantlab import combine, risk_model
from quantlab.engine import PortfolioEngine
from quantlab.synthetic import make_quality_panel

PERIODS = 12          # monthly cadence (the quality panel is month-end priced)
LOOKBACK = 36         # 36-month trailing window for the confidence estimate
N_FIRMS = 180
N_PERIODS = 120
SEED = 7
EQUITY = 1_000_000.0


def _ann_sharpe(net: pd.Series, periods: int = PERIODS) -> float:
    s = net.dropna()
    return float(s.mean() / s.std() * np.sqrt(periods)) if s.std() > 0 else 0.0


def _static_market_betas(prices: pd.DataFrame) -> pd.Series:
    """Illustrative per-firm beta to the equal-weight market (full sample).

    Static and full-sample ON PURPOSE: this demo shows the *mechanical* property
    that `neutralize_weights` zeroes `Lᵀw` regardless of how L was estimated.
    The engine's production path uses `risk_model.rolling_market_beta`
    (past-only); that is exercised in test_risk_model.py.
    """
    rets = prices.pct_change(fill_method=None).dropna(how="all")
    mkt = rets.mean(axis=1)
    var = float(mkt.var())
    return rets.apply(lambda col: col.cov(mkt) / var if var > 0 else 0.0)


def run_demo() -> dict:
    eng = PortfolioEngine(periods=PERIODS, lookback=LOOKBACK, target_vol=0.10,
                          max_weight=0.10, max_gross=2.0)

    planted = make_quality_panel(N_FIRMS, N_PERIODS, mode="planted_quality", seed=SEED)
    null = make_quality_panel(N_FIRMS, N_PERIODS, mode="null_quality", seed=SEED)
    sig_p = {"quality": planted.attrs["gp_a"]}
    sig_n = {"quality": null.attrs["gp_a"]}

    # --- 1. the honest property: commit to a real edge, ~zero on the null ----- #
    bt_p = eng.backtest(sig_p, planted, cost_bps=10.0)
    bt_n = eng.backtest(sig_n, null, cost_bps=10.0)

    # --- 2. confidence ramp: slow to lever (early third vs late third) -------- #
    gross_p = bt_p["weights"].abs().sum(axis=1)
    gross_n = bt_n["weights"].abs().sum(axis=1)
    third = max(len(gross_p) // 3, 1)
    ramp = {"early_third_avg_gross": float(gross_p.iloc[:third].mean()),
            "late_third_avg_gross": float(gross_p.iloc[-third:].mean())}

    # --- 3. limits respected -------------------------------------------------- #
    w_p = bt_p["weights"]
    limits_ok = {"max_abs_weight": float(w_p.abs().max().max()),
                 "max_gross": float(w_p.abs().sum(axis=1).max())}

    # --- 4. factor-neutralization: net market-beta exposure -> ~0 ------------- #
    betas = _static_market_betas(planted)
    loadings = pd.DataFrame({"one": 1.0, "mkt_beta": betas})
    w_raw = eng.build(sig_p, planted, loadings=None)
    w_neu = eng.build(sig_p, planted, loadings=loadings)
    last = w_neu.index[-1]
    neutralization = {
        "net_beta_exposure_raw": float(abs(risk_model.net_factor_exposure(
            w_raw.loc[last], loadings)["mkt_beta"])),
        "net_beta_exposure_neutralized": float(abs(risk_model.net_factor_exposure(
            w_neu.loc[last], loadings)["mkt_beta"])),
    }

    # --- 5. real orders for the live client (planted book, $1M) --------------- #
    plan = eng.latest_orders(sig_p, planted, equity=EQUITY)
    orders = plan["orders"].astype(int)
    top = orders.reindex(orders.abs().sort_values(ascending=False).index).head(5)
    execution = {
        "equity": EQUITY,
        "n_orders": plan["n_orders"],
        "gross_notional": plan["gross_notional"],
        "net_notional": plan["net_notional"],
        "shares_are_integer": bool(plan["target_shares"].dtype.kind == "i"),
        "gross_within_cap": bool(plan["gross_notional"] <= 2.0 * EQUITY + 1.0),
        "sample_orders": {k: int(v) for k, v in top.items()},
    }

    # --- 6. multi-signal combiner: a no-IC signal gets ~no weight ------------- #
    rng = np.random.default_rng(SEED + 1)
    noise = pd.DataFrame(rng.standard_normal(planted.attrs["gp_a"].shape),
                         index=planted.index, columns=planted.columns)
    fwd = planted.pct_change(fill_method=None).shift(-1)
    ic_quality = combine.trailing_ic(planted.attrs["gp_a"], fwd, lookback=12).iloc[-1]
    ic_noise = combine.trailing_ic(noise, fwd, lookback=12).iloc[-1]
    combiner = {"trailing_ic_quality": float(ic_quality),
                "trailing_ic_noise": float(ic_noise)}

    return {
        "_meta": {"synthetic": True, "trial_count_impact": 0, "seed": SEED,
                  "n_firms": N_FIRMS, "n_periods": N_PERIODS,
                  "periods_per_year": PERIODS, "lookback": LOOKBACK},
        "honest_property": {
            "planted_avg_gross_exposure": bt_p["avg_gross_exposure"],
            "null_avg_gross_exposure": bt_n["avg_gross_exposure"],
            "planted_net_ann_sharpe": _ann_sharpe(bt_p["net"]),
            "null_net_ann_sharpe": _ann_sharpe(bt_n["net"]),
            "planted_net_mean_period": float(bt_p["net"].mean()),
            "planted_ann_turnover": bt_p["ann_turnover"],
        },
        "confidence_ramp": ramp,
        "limits": limits_ok,
        "neutralization": neutralization,
        "execution": execution,
        "combiner": combiner,
        "_ramp_series": {"planted": gross_p.tolist(), "null": gross_n.tolist()},
    }


def assert_properties(d: dict) -> list[str]:
    """Return a list of FAILED property checks (empty list == all pass)."""
    fails = []
    h, ramp, lim = d["honest_property"], d["confidence_ramp"], d["limits"]
    if not h["planted_avg_gross_exposure"] > 0.2:
        fails.append("engine did NOT commit to the real edge (planted gross <= 0.2)")
    if not h["null_avg_gross_exposure"] < 0.05:
        fails.append("engine levered the NULL (null gross >= 0.05)")
    if not h["planted_net_mean_period"] > 0:
        fails.append("planted book not profitable net of cost")
    if not ramp["late_third_avg_gross"] > ramp["early_third_avg_gross"]:
        fails.append("no confidence ramp (engine not slow-to-lever)")
    if not lim["max_abs_weight"] <= 0.10 + 1e-9:
        fails.append("per-name weight cap breached")
    if not lim["max_gross"] <= 2.0 + 1e-9:
        fails.append("gross cap breached")
    if not d["neutralization"]["net_beta_exposure_neutralized"] < 1e-6:
        fails.append("factor neutralization did not zero net beta exposure")
    if not (d["execution"]["shares_are_integer"] and d["execution"]["gross_within_cap"]):
        fails.append("execution plan not integer / within gross cap")
    if not d["combiner"]["trailing_ic_quality"] > abs(d["combiner"]["trailing_ic_noise"]):
        fails.append("combiner: quality signal did not out-IC the noise signal")
    return fails


def _plot(d: dict, out: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d["_ramp_series"]["planted"], label="planted edge (commits)", lw=2)
    ax.plot(d["_ramp_series"]["null"], label="null (sized to ~0)", lw=2)
    ax.set_title("Engine gross exposure over time — commits to a real edge, refuses the null")
    ax.set_xlabel("month"); ax.set_ylabel("gross exposure (sum |w|)")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out, "engine_demo_ramp.png"), dpi=120)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any honest property fails (CI-style gate)")
    args = ap.parse_args()

    d = run_demo()
    fails = assert_properties(d)

    h = d["honest_property"]
    print("=" * 72)
    print("EXECUTION/RISK ENGINE - end-to-end demo (SYNTHETIC, N unchanged)")
    print("=" * 72)
    print(f"  Honest property:")
    print(f"    planted edge  -> avg gross {h['planted_avg_gross_exposure']:.3f}, "
          f"net Sharpe {h['planted_net_ann_sharpe']:.2f} (commits)")
    print(f"    null (no edge)-> avg gross {h['null_avg_gross_exposure']:.3f}, "
          f"net Sharpe {h['null_net_ann_sharpe']:.2f} (sized to ~0)")
    print(f"  Confidence ramp (slow to lever): early third {d['confidence_ramp']['early_third_avg_gross']:.3f} "
          f"-> late third {d['confidence_ramp']['late_third_avg_gross']:.3f}")
    print(f"  Limits: max |w| {d['limits']['max_abs_weight']:.3f} (cap 0.10), "
          f"max gross {d['limits']['max_gross']:.3f} (cap 2.0)")
    print(f"  Neutralization: net beta {d['neutralization']['net_beta_exposure_raw']:.4f} "
          f"-> {d['neutralization']['net_beta_exposure_neutralized']:.2e} after projection")
    print(f"  Orders @ ${EQUITY:,.0f}: {d['execution']['n_orders']} integer-share orders, "
          f"gross ${d['execution']['gross_notional']:,.0f}")
    print(f"  Combiner: quality IC {d['combiner']['trailing_ic_quality']:.3f} vs "
          f"noise IC {d['combiner']['trailing_ic_noise']:.3f}")
    print("-" * 72)
    print("  PROPERTY CHECKS: " + ("ALL PASS" if not fails else f"{len(fails)} FAILED"))
    for f in fails:
        print(f"    FAIL: {f}")

    os.makedirs(args.out, exist_ok=True)
    d.pop("_ramp_series_kept_for_plot", None)
    ramp_series = d.pop("_ramp_series")
    with open(os.path.join(args.out, "engine_demo.json"), "w") as fh:
        json.dump(d, fh, indent=2)
    d["_ramp_series"] = ramp_series
    _plot(d, args.out)
    print(f"  Wrote {os.path.join(args.out, 'engine_demo.json')} + engine_demo_ramp.png")

    return 1 if (args.strict and fails) else 0


if __name__ == "__main__":
    sys.exit(main())
