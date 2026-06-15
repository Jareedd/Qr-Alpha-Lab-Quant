"""H6 / trial #11 — the single registered CEF discount-reversion run.

    python scripts/run_cef_reversion.py --hypothesis H6 --n-trials 11

Order of operations is the registration's (writeup/preregistered_hypotheses.md),
not negotiable:
1. Registration gate: H6 must be PROPOSED (law #3, mechanized).
2. Machinery gate, in THIS environment, immediately before the real run: the
   synthetic planted_reversion world must be recovered and random_walk rejected
   (paired per-seed differential). If the harness can't tell reversion from its
   absence today, no real number is trustworthy -- abort.
3. The real run on the assembled weekly panels (CEFConnect discount + yfinance
   total return), net of 25 bps/side, with the registered paired controls
   (label shuffle, daily-NAV-only subuniverse), the Dec-Jan seasonality
   subreport, and the largest-mcap-decile robustness check.
4. Report against the registered success AND kill criteria. This script does NOT
   edit research_log.md or bump N -- spending the trial is a deliberate human
   act after reading the result.
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
from scipy import stats

from quantlab import cef_data, cef_reversion, metrics
from quantlab.registry import require_runnable_registration
from quantlab.synthetic import make_cef_panel

WEEKS_PER_YEAR = 52
HORIZON = 4            # 4-week (~21 trading day) forward label / hold
N_TRIALS_DEFAULT = 11


def _machinery_gate() -> None:
    """Synthetic falsification gate (law #4): planted reversion must beat the
    random-walk null, paired per seed, before any real number is trusted."""
    print("[gate] synthetic CEF world: planted_reversion must beat random_walk...")
    diffs = []
    for seed in (7, 11, 23):
        planted = make_cef_panel(120, 520, mode="planted_reversion", seed=seed)
        rw = make_cef_panel(120, 520, mode="random_walk", seed=seed)
        sr_p = metrics.sharpe(cef_reversion.reversion_backtest(
            planted, planted.attrs["discount"], cost_bps_per_side=0.0)["net"],
            periods=WEEKS_PER_YEAR)
        sr_n = metrics.sharpe(cef_reversion.reversion_backtest(
            rw, rw.attrs["discount"], cost_bps_per_side=0.0)["net"],
            periods=WEEKS_PER_YEAR)
        diffs.append(sr_p - sr_n)
        print(f"  seed {seed}: planted SR {sr_p:+.2f} | random-walk SR {sr_n:+.2f}")
    if min(diffs) <= 0.5:
        sys.exit(f"MACHINERY GATE FAILED: planted-null differential "
                 f"{min(diffs):.2f} <= 0.5 -- harness cannot tell reversion from "
                 "its absence; the real run is not trustworthy.")
    print(f"[gate] PASS (min paired differential {min(diffs):.2f})")


def _ic_tstat(signal: pd.DataFrame, fwd: pd.DataFrame, horizon: int):
    """Cross-sectional rank IC of discount-z vs forward total return, per week,
    Newey-West t (lags=horizon). NEGATIVE is the profitable reversion direction
    (low z = wide discount -> high forward return)."""
    ics = []
    for d in signal.index:
        a, b = signal.loc[d].dropna(), fwd.loc[d].dropna()
        common = a.index.intersection(b.index)
        if len(common) >= 6:
            ics.append(a[common].rank().corr(b[common].rank()))
    ic = pd.Series(ics).dropna()
    return float(ic.mean()), float(metrics.newey_west_tstat(ic, lags=horizon))


def _weekly_summary(net, gross, ann_turnover, n_trials) -> dict:
    """Like metrics.summary but annualized WEEKLY (the registered cadence);
    DSR/PSR are per-period and frequency-agnostic, reused as-is."""
    return {
        "ann_return_net": float(net.mean() * WEEKS_PER_YEAR),
        "ann_vol": float(net.std() * np.sqrt(WEEKS_PER_YEAR)),
        "sharpe_gross": metrics.sharpe(gross, periods=WEEKS_PER_YEAR),
        "sharpe_net": metrics.sharpe(net, periods=WEEKS_PER_YEAR),
        "max_drawdown": metrics.max_drawdown(net),
        "annual_turnover": ann_turnover,
        "skew": float(stats.skew(net.dropna())),
        "psr": metrics.probabilistic_sharpe_ratio(net),
        "dsr": metrics.deflated_sharpe_ratio(net, n_trials=n_trials),
        "n_trials_assumed": n_trials,
        "n_weeks": int(net.dropna().shape[0]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypothesis", required=True)
    ap.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    try:
        require_runnable_registration(args.hypothesis)
    except RuntimeError as exc:
        sys.exit(f"REGISTRATION GATE: {exc}")
    print(f"[registration] {args.hypothesis} verified PROPOSED -- this run "
          f"SPENDS a trial at N={args.n_trials}. Log it whatever it says.")

    _machinery_gate()

    # Universe: the Stage-1 tradable tail (CEFConnect lists CEFs; exclude any
    # BDC-category contaminant). Stale-NAV funds KEPT (isolated by the control).
    snap = cef_data.daily_snapshot()
    tail = snap[(snap["MarketCapUSDm"] < 400) & (snap["dollar_adv"] >= 250_000)]
    cat = tail.get("CategoryName", pd.Series("", index=tail.index)).fillna("")
    tail = tail[~cat.str.contains("BDC", case=False)]
    print(f"[data] tradable tail: {len(tail)} funds; assembling weekly panels...")

    panels = cef_data.build_weekly_panels(tail.index.tolist())
    price, discount = panels["price"], panels["discount"]
    print(f"[data] panels: {price.shape[1]} funds x {price.shape[0]} weeks")

    res = cef_reversion.reversion_backtest(price, discount)
    net, gross = res["net"], res["gross"]
    fwd = cef_reversion.forward_total_return(price, HORIZON)
    ic_mean, ic_t = _ic_tstat(res["signal"], fwd, HORIZON)

    stats_d = _weekly_summary(net, gross, res["annual_turnover"], args.n_trials)
    stats_d["mean_rank_ic"] = ic_mean
    stats_d["ic_tstat_newey_west"] = ic_t

    # Baseline: equal-weight long-only CEF (buy-hold, ~no turnover).
    ew = price.pct_change(fill_method=None).mean(axis=1).dropna()
    stats_d["baseline_ew_cef_sharpe"] = metrics.sharpe(ew, periods=WEEKS_PER_YEAR)

    # Control 1: label shuffle (same weights, returns permuted across funds).
    shuf = cef_reversion.shuffle_returns(res["total_ret"], seed=0)
    ctrl = (res["held"] * shuf).sum(axis=1, min_count=1).dropna()
    stats_d["control_shuffled_label_sharpe"] = metrics.sharpe(ctrl, periods=WEEKS_PER_YEAR)

    # Control 2: NAV-staleness — re-run on the daily-NAV-only subuniverse.
    lag = snap["nav_lag_days"].reindex(price.columns)
    daily_nav = lag[lag <= 1].index
    mask = pd.DataFrame(False, index=price.index, columns=price.columns)
    mask[daily_nav] = True
    res_dn = cef_reversion.reversion_backtest(price, discount, universe=mask)
    stats_d["control_daily_nav_only_sharpe"] = metrics.sharpe(res_dn["net"], periods=WEEKS_PER_YEAR)
    stats_d["n_daily_nav_funds"] = int(len(daily_nav))

    # Seasonality subreport: SR excluding Dec & Jan (tax-loss window).
    no_dj = net[~net.index.month.isin([12, 1])]
    stats_d["sharpe_net_ex_dec_jan"] = metrics.sharpe(no_dj, periods=WEEKS_PER_YEAR)

    # Robustness: drop the largest-mcap decile of the universe.
    mcap = snap["MarketCapUSDm"].reindex(price.columns)
    cutoff = mcap.quantile(0.9)
    keep = mcap[mcap <= cutoff].index
    res_sm = cef_reversion.reversion_backtest(price[keep], discount[keep])
    stats_d["sharpe_net_ex_largest_mcap_decile"] = metrics.sharpe(res_sm["net"], periods=WEEKS_PER_YEAR)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "metrics_h6_reversion.json"), "w") as f:
        json.dump(stats_d, f, indent=2)
    fig, ax = plt.subplots(figsize=(10, 5))
    (1 + net).cumprod().plot(ax=ax, label=f"net reversion (SR {stats_d['sharpe_net']:.2f})")
    (1 + ctrl).cumprod().plot(ax=ax, alpha=0.6,
                              label=f"shuffled-label control (SR {stats_d['control_shuffled_label_sharpe']:.2f})")
    ax.legend(); ax.set_title(f"H6 CEF discount reversion — DSR {stats_d['dsr']:.2f} @ N={args.n_trials}")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "equity_h6_reversion.png"), dpi=120)

    print("\n=== H6 CEF discount reversion — trial #11 (registered, 25bps/side, weekly) ===")
    for k, v in stats_d.items():
        print(f"  {k:>34}: {v:.4f}" if isinstance(v, float) else f"  {k:>34}: {v}")
    crit = (stats_d["ic_tstat_newey_west"] <= -2 and stats_d["sharpe_net"] > 0
            and stats_d["sharpe_net"] > stats_d["baseline_ew_cef_sharpe"]
            and stats_d["dsr"] >= 0.95
            and stats_d["sharpe_net_ex_largest_mcap_decile"] > 0
            and stats_d["control_daily_nav_only_sharpe"] > 0
            and abs(stats_d["control_shuffled_label_sharpe"]) < 0.3)
    print(f"\n  REGISTERED SUCCESS CRITERIA: {'MET' if crit else 'NOT MET'} "
          "(t_NW<=-2 AND net SR>0 AND >EW baseline AND DSR>=0.95 AND survives "
          "ex-largest-mcap-decile AND daily-NAV control holds AND shuffle ~flat)")


if __name__ == "__main__":
    main()
