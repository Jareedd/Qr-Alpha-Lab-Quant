"""H13 graded run (trial #14) — the PRE-REGISTERED post-earnings-announcement
drift (PEAD) study, the project's FIRST institutional-data hypothesis.

Executes the H13 registration in writeup/preregistered_hypotheses.md FAITHFULLY:
SUE-quantile event study (enter T+2, hold 60d, top/bottom-quintile long/short,
10 bps/side) plus a monthly cross-sectional variant, the registered paired
controls (drift-vs-reaction T+2/5/10, surprise-shuffle placebo, size-tercile
split), the two baselines (EW, 12-1 momentum), and the DSR @ N=14 / NW-t / MDE.
It composes existing, separately-tested pieces — the pead.py compute layer + a
price source — and adds no new strategy logic and no new knobs.

The whole point of this harness: it is built and OFFLINE-TESTED on a synthetic
PEAD world BEFORE any Bloomberg data exists, so the only thing standing between
here and a graded trial #14 is the operator's CSV. It is DATA-GATED — a graded
run REQUIRES a real Bloomberg surprise CSV at data_cache/bloomberg/
pead_surprises.csv (the consensus estimate free data cannot give us).

Order of operations (each gate aborts via sys.exit, spending NO trial):
  registration gate -> machinery gate (synthetic planted vs null PEAD, paired)
  -> DATA GATE (Bloomberg CSV required, price source required) -> parse CSV +
  build SUE + load prices -> event study + cross-sectional -> controls ->
  baselines + DSR/NW-t/MDE -> pre-registered verdict block -> structured dict.

Does NOT auto-bump N, does NOT auto-log (the row is added by hand after sign-off,
same as trial #12), and does NOT fetch real data on the import path.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from quantlab import metrics, pead
from quantlab.registry import require_runnable_registration

# --------------------------------------------------------------------------- #
# FROZEN module constants (the H13 registered config; pinned by tests).
# --------------------------------------------------------------------------- #
ENTER_LAG = 2                # enter T+2 (skip the announcement-day jump; PIT)
HOLD = 60                    # hold ~60 trading days (the drift window)
QUANTILE = 0.2              # top/bottom QUINTILE long/short by SUE
COST_BPS = 10.0             # 10 bps/side
PERIODS_PER_YEAR = 252     # daily event-time book
CS_WINDOW_DAYS = 90        # cross-sectional: trailing window for most-recent SUE
N_TRIALS_DEFAULT = 14      # the DSR uses N=14 (this is trial #14)
DRIFT_RETENTION_FLOOR = 0.5  # T+5 must retain >=50% of T+2 (drift-vs-reaction)

BLOOMBERG_CSV = os.path.join("data_cache", "bloomberg", "pead_surprises.csv")
PULL_DOC = "writeup/bloomberg_pead_pull.md"


# --------------------------------------------------------------------------- #
# Baselines (the registration requires beating BOTH: equal-weight, 12-1 mom).
# --------------------------------------------------------------------------- #

def equal_weight_baseline(prices: pd.DataFrame) -> float:
    """Long-only equal-weight monthly book on the priceable universe (EW
    baseline). Monthly cadence -> annualized at sqrt(12). pandas 2.x:
    ``.mean(axis=1).dropna()`` (no min_count on mean)."""
    monthly = prices.resample("ME").last().pct_change(fill_method=None).shift(-1)
    ew = monthly.mean(axis=1).dropna()
    return metrics.sharpe(ew, periods=12)


def momentum_baseline(prices: pd.DataFrame) -> float:
    """12-1 momentum, dollar-neutral EW quintile L/S on the monthly grid (CLAUDE.md
    baseline #5): trailing 12m total return skipping the most recent month,
    ranked cross-sectionally, rebalanced monthly, 10 bps/side."""
    monthly = prices.resample("ME").last()
    mom = monthly.shift(1) / monthly.shift(12) - 1.0     # 12-1 on months
    fwd = monthly.pct_change(fill_method=None).shift(-1)
    weights = pd.DataFrame(0.0, index=monthly.index, columns=monthly.columns)
    for t in monthly.index:
        sig = mom.loc[t].dropna()
        if sig.shape[0] < 5:
            continue
        lo, hi = sig.quantile(QUANTILE), sig.quantile(1.0 - QUANTILE)
        longs, shorts = sig[sig >= hi].index, sig[sig <= lo].index
        if len(longs) == 0 or len(shorts) == 0:
            continue
        weights.loc[t, longs] = 0.5 / len(longs)
        weights.loc[t, shorts] = -0.5 / len(shorts)
    gross = (weights * fwd.reindex_like(weights)).sum(axis=1, min_count=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    net = (gross - turnover * COST_BPS / 1e4).dropna()
    return metrics.sharpe(net, periods=12)


# --------------------------------------------------------------------------- #
# The graded run (reachable only once the DATA GATE passes).
# --------------------------------------------------------------------------- #

def _run_trial(events: pd.DataFrame, prices: pd.DataFrame, n_trials: int,
               volumes: pd.DataFrame | None = None) -> dict:
    """The graded PEAD run on a SUE-scored event table + a price panel.

    ``events`` must already carry a ``sue`` column (pead.compute_sue output);
    ``prices`` is a (day x ticker) total-return panel. This is the pure scoring
    core shared by the live harness and the offline end-to-end test — it touches
    no network and no files."""
    # ---- primary event study (enter T+2, hold 60, quintiles, 10 bps) -------- #
    es = pead.pead_event_study(
        events, prices, enter_lag=ENTER_LAG, hold=HOLD, quantile=QUANTILE,
        cost_bps=COST_BPS, periods=PERIODS_PER_YEAR)
    port = es["daily_portfolio"]
    sr = es["net_sharpe"]
    t_nw = es["t_nw"]
    dsr = (metrics.deflated_sharpe_ratio(port, n_trials=n_trials)
           if len(port) > 30 else float("nan"))

    # ---- monthly cross-sectional variant ------------------------------------ #
    cs = pead.pead_cross_sectional(
        events, prices, window_days=CS_WINDOW_DAYS, quantile=QUANTILE,
        cost_bps=COST_BPS, periods=12)

    # ---- controls ----------------------------------------------------------- #
    dvr = pead.drift_vs_reaction(
        events, prices, lags=(2, 5, 10), hold=HOLD, quantile=QUANTILE,
        cost_bps=COST_BPS, periods=PERIODS_PER_YEAR)
    shuffle_sr = pead.surprise_shuffle_sr(
        events, prices, enter_lag=ENTER_LAG, hold=HOLD, quantile=QUANTILE,
        cost_bps=COST_BPS, periods=PERIODS_PER_YEAR)
    terc = pead.by_size_tercile(
        events, prices, volumes=volumes, enter_lag=ENTER_LAG, hold=HOLD,
        quantile=QUANTILE, cost_bps=COST_BPS, periods=PERIODS_PER_YEAR)

    # ---- baselines (monthly cross-sectional books) -------------------------- #
    sr_ew = equal_weight_baseline(prices)
    sr_mom = momentum_baseline(prices)

    # ---- MDE (printed beside the result; the power check) -------------------- #
    n_obs = int(len(port))
    var_sr = 1.0 / max(n_obs, 1)
    mde_pp = metrics.expected_max_sharpe(n_trials, var_sr, n_obs)
    mde_ann = mde_pp * np.sqrt(PERIODS_PER_YEAR)

    # ---- pre-registered verdict (the success-criteria conjunction) ---------- #
    t_nw_ok = (t_nw is not None and not np.isnan(t_nw) and t_nw >= 2.0)
    sr_pos = sr > 0
    beats_baselines = sr > sr_ew and sr > sr_mom
    dsr_ok = (not np.isnan(dsr)) and dsr >= 0.95
    # drift-vs-reaction: T+5 retains >=50% of T+2 (and T+2 itself positive).
    retention = dvr["retention_t5_over_t2"]
    drift_ok = sr > 0 and retention >= DRIFT_RETENTION_FLOOR
    # not confined to the illiquid (smallest) tercile: the effect must survive
    # OUTSIDE tercile 0 (at least one of the larger terciles 1/2 net SR > 0).
    terc_sr = terc.get("tercile_sharpe", {})
    not_illiquid_only = any(
        terc_sr.get(k, 0.0) > 0 for k in (1, 2)) if terc_sr else False

    graduate = (t_nw_ok and sr_pos and beats_baselines and dsr_ok
                and drift_ok and not_illiquid_only)

    result = {
        "event_study": es, "cross_sectional": cs,
        "net_sharpe": sr, "t_nw": t_nw, "dsr": dsr, "n_obs": n_obs,
        "drift_vs_reaction": dvr, "shuffle_sr": shuffle_sr, "size_tercile": terc,
        "sr_ew": sr_ew, "sr_mom": sr_mom, "mde_ann": mde_ann,
        "gates": {
            "t_nw": t_nw_ok, "sr_pos": sr_pos, "beats_baselines": beats_baselines,
            "dsr": dsr_ok, "drift_vs_reaction": drift_ok,
            "not_illiquid_only": not_illiquid_only,
        },
        "graduate": bool(graduate),
    }
    return result


def _print_verdict(res: dict, n_trials: int) -> None:
    es = res["event_study"]
    dvr = res["drift_vs_reaction"]
    terc = res["size_tercile"]
    print("\n=== H13 PEAD result (SUE-quantile event study; enter T+2, hold 60) ===")
    print(f"  event study : net SR {res['net_sharpe']:+.3f}  t_NW {res['t_nw']:+.2f}"
          f"  DSR {res['dsr']:.3f}  n_events {es['n_events']} "
          f"(long {es['n_long']} / short {es['n_short']})  n_obs {res['n_obs']}")
    cs = res["cross_sectional"]
    print(f"  cross-sect. : net SR {cs['net_sharpe']:+.3f}  t_NW {cs['t_nw']:+.2f}"
          f"  months {cs['n_months']}  turnover {cs['annual_turnover']:.2f}/yr")
    print(f"  baselines   : EW SR {res['sr_ew']:+.3f} | 12-1 mom SR {res['sr_mom']:+.3f}")
    print(f"  drift/react : "
          + "  ".join(f"T+{k} SR {v:+.3f}" for k, v in dvr["sharpe_by_lag"].items())
          + f"  (T+5/T+2 retention {dvr['retention_t5_over_t2']:.2f}, "
          f"floor {DRIFT_RETENTION_FLOOR})")
    print(f"  shuffle     : surprise-shuffle placebo SR {res['shuffle_sr']:+.3f} (|SR|<0.3)")
    ts = terc.get("tercile_sharpe", {})
    tn = terc.get("tercile_n", {})
    print("  size tercile: "
          + "  ".join(f"T{k}(n{tn.get(k,0)}) SR {ts.get(k,float('nan')):+.3f}"
                      for k in sorted(ts)) + "  (T0=smallest/least-liquid)")
    print(f"  MDE @ N={n_trials}, n_obs={res['n_obs']}: net annual SR hurdle "
          f"~{res['mde_ann']:.3f} (DSR>=0.95 deflation)")

    g = res["gates"]
    print("\n=== PRE-REGISTERED VERDICT (graduate iff ALL hold) ===")
    print(f"  1. t_NW >= +2 ............. {res['t_nw']:+.2f}  -> "
          f"{'PASS' if g['t_nw'] else 'FAIL'}")
    print(f"  2. net SR>0 & > baselines . {res['net_sharpe']:+.3f} vs EW "
          f"{res['sr_ew']:+.3f}, mom {res['sr_mom']:+.3f}  -> "
          f"{'PASS' if (g['sr_pos'] and g['beats_baselines']) else 'FAIL'}")
    print(f"  3. DSR >= 0.95 ........... {res['dsr']:.3f}  -> "
          f"{'PASS' if g['dsr'] else 'FAIL'}")
    print(f"  4. drift-vs-reaction ..... T+5/T+2 retention "
          f"{dvr['retention_t5_over_t2']:.2f} >= {DRIFT_RETENTION_FLOOR}  -> "
          f"{'PASS' if g['drift_vs_reaction'] else 'FAIL'}")
    print(f"  5. not illiquid-only ..... larger terciles SR "
          f"{[round(ts.get(k,float('nan')),3) for k in (1,2)]}  -> "
          f"{'PASS' if g['not_illiquid_only'] else 'FAIL'}")
    print(f"  >>> H13 {'GRADUATES' if res['graduate'] else 'does NOT graduate'} "
          "(log the row by hand; N becomes 14).")


def _load_prices_for_events(events: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Load delisting-inclusive prices for exactly the tickers in ``events`` from
    the requested PRICE source. Network happens HERE (call-time), never at import.
    Shared by both the Bloomberg-CSV and FMP earnings paths."""
    tickers = sorted(events["ticker"].unique())
    start = str((events['ann_date'].min() - pd.Timedelta(days=400)).date())
    end = str((events['ann_date'].max() + pd.Timedelta(days=120)).date())
    if source_name == "tiingo":
        from quantlab.tiingo_data import TiingoSource
        src = TiingoSource()
        prices = src.prices(tickers, start, end)
    elif source_name == "sec_xwalk":
        from quantlab.sec_xwalk_source import SurvivorshipSafeSECSource
        src = SurvivorshipSafeSECSource(start=start, end=end)
        # daily delisting-inclusive panel over the window for these tickers
        daily = pd.bdate_range(start, end)
        prices = src.prices(tickers, daily)
    else:
        sys.exit(f"unknown --price-source {source_name!r}")
    if prices.empty:
        sys.exit(
            f"\nDATA GATE: price source '{source_name}' returned NO prices for the "
            "event tickers — cannot run the event study. Check the source / cache; "
            "no trial spent (N unchanged).")
    print(f"[prices] {prices.shape[1]} priceable tickers x {prices.shape[0]} days "
          f"({source_name}, delisting-inclusive).")
    return prices


def load_csv_events(csv_path: str) -> pd.DataFrame:
    """Parse the Bloomberg surprise CSV -> SUE-scored events. Pure (no network)."""
    parsed = pead.parse_pead_csv(csv_path)
    events = pead.compute_sue(parsed)
    tickers = sorted(events["ticker"].unique())
    print(f"[data] parsed {len(parsed)} surprise rows -> {len(events)} SUE events "
          f"across {len(tickers)} tickers; SUE methods "
          f"{dict(events['sue_method'].value_counts())}")
    return events


def _pit_sp500_universe() -> list[str]:
    """Every ticker that was an S&P 500 member at any point in the PIT window
    (the survivorship-safe universe from quantlab.universe). Network here (the
    cached Wikipedia tables); never at import."""
    from quantlab import universe as U
    current, changes = U.fetch_sp500_tables()
    intervals = U.build_membership_intervals(current, changes)
    return U.all_members_in_window(intervals)


def build_fmp_events(universe: list[str], max_names: int | None,
                     do_cross_check: bool) -> tuple[pd.DataFrame, dict]:
    """Build SUE-scored events from FMP for ``universe`` (capped at ``max_names``),
    run the leakage gate, and return ``(events_with_sue, validation_report)``.

    The events frame is the gate's INPUT (carries last_updated for the recency
    caveat); SUE is computed only AFTER the gate passes (the caller decides). The
    optional Alpha Vantage cross-check is attempted only if ``do_cross_check`` and
    ALPHAVANTAGE_API_KEY is present (skipped gracefully otherwise)."""
    from quantlab import fmp_earnings as fmp
    print(f"[data] FMP earnings pull over {len(universe)} symbols"
          + (f" (capped at {max_names} this run; cached names are free)" if max_names else "")
          + " ...")
    raw_events = fmp.build_events(universe, max_names=max_names)
    cross = None
    if do_cross_check:
        # Sample the names actually pulled, so the overlap is non-empty.
        pulled = sorted(raw_events["ticker"].unique()) if not raw_events.empty else universe
        cross = fmp.fetch_alphavantage_cross_check(pulled, max_names=10)
        if cross is None:
            print("[data] Alpha Vantage cross-check skipped "
                  "(no ALPHAVANTAGE_API_KEY or no data).")
    report = fmp.validate_pead_events(raw_events, cross_check=cross)
    return raw_events, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypothesis", default="H13")
    ap.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    ap.add_argument("--source", choices=["fmp", "csv"], default="fmp",
                    help="earnings source: 'fmp' (free, gated) or 'csv' (Bloomberg).")
    ap.add_argument("--price-source", choices=["tiingo", "sec_xwalk"], default="tiingo",
                    help="price source for the event-study returns leg.")
    ap.add_argument("--max-names", type=int, default=None,
                    help="cap on distinct FMP symbols pulled this run (free-tier "
                         "quota; cached names are free on re-runs).")
    ap.add_argument("--cross-check", action="store_true",
                    help="attempt an Alpha Vantage cross-source PIT check "
                         "(needs ALPHAVANTAGE_API_KEY; skipped otherwise).")
    ap.add_argument("--csv", default=BLOOMBERG_CSV)
    args = ap.parse_args()

    # 1) Registration gate (law #3): H13 must be PROPOSED.
    try:
        require_runnable_registration(args.hypothesis)
    except RuntimeError as exc:
        sys.exit(f"REGISTRATION GATE: {exc}")
    print(f"[registration] {args.hypothesis} verified PROPOSED -- a graded run "
          f"spends a trial at N={args.n_trials}.")

    # 2) Machinery gate (law #4): planted_pead recovered, null_pead rejected,
    #    paired per seed. Runs on SYNTHETIC data only (no network, no trial).
    print("[gate] synthetic PEAD world: planted drift must beat null (paired)...")
    gate = pead.machinery_gate()
    for s, p, n, d in zip(gate["seeds"], gate["planted_sr"], gate["null_sr"],
                          gate["diffs"]):
        print(f"  seed {s}: planted SR {p:+.2f} | null SR {n:+.2f} | diff {d:+.2f}")
    if not gate["passed"]:
        sys.exit(
            f"MACHINERY GATE FAILED: harness cannot tell planted PEAD from its "
            f"absence (min planted {min(gate['planted_sr']):+.2f}, max null "
            f"{max(gate['null_sr']):+.2f}); abort, no trial spent.")
    print(f"[gate] PASS (min paired differential {min(gate['diffs']):.2f})")

    # 3) DATA GATE. Two earnings sources, each gated:
    #    csv  -> the existing Bloomberg surprise CSV (must exist).
    #    fmp  -> free FMP estimates, GATED by the leakage validator. The gate
    #            fails CLOSED: if the free estimates look backfilled / non-PIT /
    #            degenerate, we sys.exit BEFORE spending a trial (N unchanged).
    if args.source == "csv":
        if not os.path.exists(args.csv):
            sys.exit(
                "\nDATA GATE: no Bloomberg PEAD data at "
                f"{args.csv}.\n  The harness is built and offline-validated (the "
                "machinery gate above PASSED on the synthetic PEAD world), but a "
                "GRADED trial #14 needs the at-the-announcement consensus estimate "
                "that free data lacks.\n  -> Pull it per the bounded, license-"
                f"respecting checklist in {PULL_DOC} (one small CSV: ticker, "
                "ann_date, actual_eps, est_eps [+ std_est/surprise_pct if they "
                "fit the limit]), drop it at the path above, and re-run.\n  No "
                "trial spent; N unchanged.")
        print(f"[data] Bloomberg surprise CSV found at {args.csv}.")
        events = load_csv_events(args.csv)
    else:  # fmp
        universe = _pit_sp500_universe()
        raw_events, report = build_fmp_events(
            universe, args.max_names, args.cross_check)
        # Print the validation report either way (PASS or FAIL).
        from quantlab.fmp_earnings import format_validation_report
        print(format_validation_report(report))
        if not report["passed"]:
            sys.exit(
                "\nDATA GATE (FMP leakage validator) FAILED — the free earnings "
                "feed shows at least one gross red flag of a non-PIT / broken "
                "source:\n  - " + "\n  - ".join(report["reasons"])
                + "\n  Failing CLOSED: a contaminated graded trial is worse than "
                "no trial. No trial spent; N unchanged. Fix/replace the feed (or "
                "fall back to --source csv) and re-run.")
        # Gate passed -> NOW compute SUE (drop the validation-only last_updated col).
        events = pead.compute_sue(
            raw_events.drop(columns=["last_updated"], errors="ignore"))
        print(f"[data] FMP gate PASSED -> {len(events)} SUE events across "
              f"{events['ticker'].nunique()} tickers; SUE methods "
              f"{dict(events['sue_method'].value_counts())}")
        print("[data] REMINDER: gate PASS == no gross red flags, NOT proven "
              "point-in-time (see CAVEAT above).")

    # 4) load delisting-inclusive prices for the event tickers (network here).
    prices = _load_prices_for_events(events, args.price_source)

    # 5-7) event study + cross-sectional + controls + baselines + verdict.
    res = _run_trial(events, prices, args.n_trials)
    _print_verdict(res, args.n_trials)


if __name__ == "__main__":
    main()
