"""§5.0 beta-leak diagnosis over the committed live record (read-only).

Marks the INTENDED book (committed weights_*.csv x public prices, using the
monitor's own t+1 convention), takes the BROKER equity stream from the
committed summary_*.json, and decomposes both against SPY. The comparison
separates a construction leak (intended book carries beta) from an
execution leak (intended is flat, broker is not) from idio concentration
(both flat; the vol is breadth, not factor).

Monitoring-class analysis of the committed audit trail: NO trial spent, NO
registration required, the frozen live config untouched. Public yfinance
marks, same as the nightly monitor; the broker record stays authoritative.

Run:  python scripts/live_beta_diagnosis.py
Out:  results/live_beta_diagnosis.json + a printed memo block.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import monitor, risk  # noqa: E402
from quantlab.data import load_prices  # noqa: E402
from quantlab.live_diagnosis import (  # noqa: E402
    beta_decomposition,
    broker_returns,
    exante_beta_path,
    execution_gap,
    load_summaries,
    rolling_beta_series,
)

MASS_FAILURE = 10  # a cycle with >= this many rejected orders is a "mass failure"


def _cycle_compound(daily: pd.Series, cycle_dates: pd.DatetimeIndex) -> pd.Series:
    """Compound a daily return stream into per-cycle returns on cycle dates."""
    cum = (1.0 + daily).cumprod()
    on_cycles = cum.reindex(cycle_dates, method="ffill")
    return on_cycles.pct_change().dropna()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live-dir", default="results/live")
    ap.add_argument("--start", default="2025-06-01",
                    help="price history start (needs ~1y before the live window for betas)")
    ap.add_argument("--cache-dir", default="data_cache/live_diag")
    ap.add_argument("--window", type=int, default=15,
                    help="rolling-beta window in cycles (declared with every number)")
    ap.add_argument("--out", default="results/live_beta_diagnosis.json")
    args = ap.parse_args()

    weights_by_date, _ = monitor.load_live_records(args.live_dir)
    summaries = load_summaries(args.live_dir)
    tickers = sorted(set().union(*[set(w.index) for w in weights_by_date.values()]))
    print(f"[diag] {len(weights_by_date)} cycles, {len(tickers)} distinct held names, "
          f"{summaries['n_failed'].sum():.0f} total rejected orders")

    prices = load_prices(tickers + ["SPY"], start=args.start,
                         cache_dir=args.cache_dir, min_coverage=0.0)
    spy = prices.pop("SPY")
    mkt_daily = spy.pct_change(fill_method=None).dropna()

    # -- the two return streams -------------------------------------------
    intended_daily = monitor.realized_book_returns(weights_by_date, prices)
    broker = broker_returns(summaries["equity"])
    cycle_dates = broker.index
    intended_cycle = _cycle_compound(intended_daily, summaries.index)
    mkt_cycle = _cycle_compound(mkt_daily, summaries.index)

    # -- decompositions ----------------------------------------------------
    d_broker = beta_decomposition(broker, mkt_cycle)
    d_intended = beta_decomposition(intended_cycle.reindex(cycle_dates).dropna(), mkt_cycle)
    rb_broker = rolling_beta_series(broker, mkt_cycle, window=args.window)
    rb_intended = rolling_beta_series(intended_daily, mkt_daily, window=args.window)

    # -- execution gap, annotated with the failure record ------------------
    gap = execution_gap(broker, intended_cycle)
    gap = gap.join(summaries[["orders_sent", "n_failed", "failed_sells", "failed_buys"]])
    # a failed rebalance on cycle d corrupts the book HELD from d+1 onward:
    gap["n_failed_prev"] = gap["n_failed"].shift(1).fillna(0)
    mass_prev = gap["n_failed_prev"] >= MASS_FAILURE
    gap_after_mass = gap.loc[mass_prev, "gap"].abs()
    gap_normal = gap.loc[~mass_prev, "gap"].abs()

    # -- ex-ante beta of the logged books (construction's own claim) -------
    rets = prices.pct_change(fill_method=None)
    mkt_own = rets.mean(axis=1)  # approximates construction's member-mean market
    betas_own = risk.rolling_beta(rets, mkt_own)
    betas_spy = risk.rolling_beta(rets, mkt_daily)
    exante_own = exante_beta_path(weights_by_date, betas_own)
    exante_spy = exante_beta_path(weights_by_date, betas_spy)

    # -- structure: gross, cadence, and the worst residual window -----------
    dates = sorted(weights_by_date)
    gross = pd.Series({d: weights_by_date[d].abs().sum() for d in dates})
    l1 = []
    for a, b in zip(dates[:-1], dates[1:]):
        ua = weights_by_date[a].reindex(
            weights_by_date[a].index.union(weights_by_date[b].index)).fillna(0.0)
        ub = weights_by_date[b].reindex(ua.index).fillna(0.0)
        l1.append((ub - ua).abs().sum())
    l1 = pd.Series(l1, index=dates[1:])
    backtest_ref, backtest_metrics = "results/metrics_sp500_ridge_both_residlabel.json", None
    if os.path.exists(backtest_ref):
        with open(backtest_ref) as f:
            bm = json.load(f)
        backtest_metrics = {"ann_vol": bm.get("ann_vol"),
                            "annual_turnover_one_way": bm.get("annual_turnover"),
                            "sharpe_net": bm.get("sharpe_net")}

    # worst contiguous 4-cycle stretch of the intended book, with the same-window
    # market move and what the ex-ante beta would have implied:
    cum4 = (1 + intended_cycle).rolling(4).apply(np.prod, raw=True) - 1
    worst_end = cum4.idxmin()
    w_slice = intended_cycle.loc[:worst_end].iloc[-4:]
    mkt_slice = mkt_cycle.reindex(w_slice.index)
    beta_at = exante_spy.reindex(w_slice.index, method="ffill").mean()
    worst_window = {
        "cycles": [str(d.date()) for d in w_slice.index],
        "intended_cum": float((1 + w_slice).prod() - 1),
        "spy_cum": float((1 + mkt_slice).prod() - 1),
        "exante_beta_mean": float(beta_at),
        "beta_implied_cum": float(beta_at * ((1 + mkt_slice).prod() - 1)),
    }

    # -- sector nets from the logged books (cached table; declared) --------
    sector_report = None
    cur_path = os.path.join("data_cache", "sp500_current.parquet")
    if os.path.exists(cur_path):
        from quantlab.universe import sector_map
        sectors = sector_map(pd.read_parquet(cur_path), tickers)
        book = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
        for d, w in weights_by_date.items():
            if d in book.index:
                book.loc[d, :] = 0.0
                book.loc[d, w.index.intersection(book.columns)] = w
        book = book.ffill()
        sector_report = risk.risk_report(
            intended_daily, mkt_daily, book.shift(1).dropna(how="all"),
            betas_own, sectors, window=args.window)

    result = {
        "asof": str(summaries.index[-1].date()),
        "n_cycles": int(len(weights_by_date)),
        "window": args.window,
        "broker": d_broker,
        "intended": d_intended,
        "rolling_beta_p95_abs": {
            "broker": float(rb_broker.abs().quantile(0.95)) if len(rb_broker) else None,
            "intended": float(rb_intended.abs().quantile(0.95)) if len(rb_intended) else None,
        },
        "execution_gap": {
            "cum_gap_final": float(gap["cum_gap"].iloc[-1]),
            "gap_ann_vol": float(gap["gap"].std(ddof=1) * np.sqrt(252)),
            "mean_abs_gap_after_mass_failure": float(gap_after_mass.mean()) if len(gap_after_mass) else None,
            "mean_abs_gap_normal": float(gap_normal.mean()),
            "n_mass_failure_cycles": int((summaries["n_failed"] >= MASS_FAILURE).sum()),
            "worst_5_gap_cycles": [
                {"date": str(d.date()), "gap": float(r["gap"]),
                 "n_failed_prev_cycle": int(r["n_failed_prev"])}
                for d, r in gap.reindex(gap["gap"].abs().sort_values(ascending=False)
                                        .head(5).index).iterrows()],
        },
        "structure": {
            "gross_mean": float(gross.mean()), "gross_max": float(gross.max()),
            "l1_weight_change_per_cycle_mean": float(l1.mean()),
            "annual_turnover_one_way": float(l1.mean() / 2 * 252),
            "deployed_backtest_ref": backtest_ref,
            "deployed_backtest": backtest_metrics,
            "worst_4cycle_window": worst_window,
        },
        "exante_beta": {
            "vs_own_market_mean_abs": float(exante_own.abs().mean()),
            "vs_own_market_p95_abs": float(exante_own.abs().quantile(0.95)),
            "vs_spy_mean_abs": float(exante_spy.abs().mean()),
            "vs_spy_p95_abs": float(exante_spy.abs().quantile(0.95)),
            "vs_spy_signed_mean": float(exante_spy.mean()),
            "vs_spy_frac_positive": float((exante_spy > 0).mean()),
            "vs_spy_series": {str(d.date()): round(float(v), 4)
                              for d, v in exante_spy.items()},
            "note": "own-market betas approximate construction's member-mean "
                    "with the mean over held names; vs_own ~0 confirms the "
                    "projection did its job on its own terms",
        },
        "sector_report_intended": sector_report,
        "per_cycle": [
            {"date": str(d.date()), **{k: (float(r[k]) if pd.notna(r[k]) else None)
             for k in ("broker", "intended", "gap", "cum_gap", "n_failed")}}
            for d, r in gap.iterrows()],
        "limitations": [
            f"{len(broker)} broker-return observations — every beta carries a wide CI; "
            "iid OLS standard errors are reported and are optimistic under autocorrelation",
            "intended book marked at yfinance adjusted closes (the monitor's convention); "
            "broker fills differ (costs, partial fills, timing)",
            "sector map is as-of-today (cached current table), not PIT",
            "ex-ante betas recomputed from held-names data, not the construction's exact "
            "member-mean inputs — vs_own_market is the like-for-like check",
        ],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    b, i = result["broker"], result["intended"]
    print(f"""
[diag] ================= §5.0 BETA-LEAK DIAGNOSIS ({result['asof']}) =================
[diag] BROKER equity  : ann vol {b['ann_vol_strat']:6.1%}  beta {b['beta']:+.3f} (se {b['beta_se_iid']:.3f})  mkt-var share {b['mkt_var_share']:5.1%}  corr {b['corr']:+.2f}
[diag] INTENDED book  : ann vol {i['ann_vol_strat']:6.1%}  beta {i['beta']:+.3f} (se {i['beta_se_iid']:.3f})  mkt-var share {i['mkt_var_share']:5.1%}  corr {i['corr']:+.2f}
[diag] rolling |beta| p95 (w={args.window}): broker {result['rolling_beta_p95_abs']['broker']}, intended {result['rolling_beta_p95_abs']['intended']}
[diag] ex-ante w·beta (own mkt): mean|.| {result['exante_beta']['vs_own_market_mean_abs']:.4f}, p95|.| {result['exante_beta']['vs_own_market_p95_abs']:.4f}
[diag] ex-ante w·beta_SPY: signed mean {result['exante_beta']['vs_spy_signed_mean']:+.3f}, frac>0 {result['exante_beta']['vs_spy_frac_positive']:.0%}
[diag] structure      : gross {result['structure']['gross_mean']:.3f}, one-way turnover {result['structure']['annual_turnover_one_way']:.1f}x/yr vs deployed backtest {result['structure']['deployed_backtest']}
[diag] worst 4 cycles : {result['structure']['worst_4cycle_window']}
[diag] execution gap  : cum {result['execution_gap']['cum_gap_final']:+.2%}, ann vol {result['execution_gap']['gap_ann_vol']:6.1%}, mass-failure cycles {result['execution_gap']['n_mass_failure_cycles']}
[diag]   mean |gap| after mass failure: {result['execution_gap']['mean_abs_gap_after_mass_failure']}
[diag]   mean |gap| normal cycles     : {result['execution_gap']['mean_abs_gap_normal']:.4f}
[diag] worst gap cycles: {result['execution_gap']['worst_5_gap_cycles']}
[diag] full JSON -> {args.out}
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
