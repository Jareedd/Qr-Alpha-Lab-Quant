"""H10 graded run (trial #13) — the PRE-REGISTERED opportunistic insider
cluster-buying long-vs-EW book.

Executes the H10 STAGE-2 FROZEN CONFIG in writeup/preregistered_hypotheses.md
(the 2026-06-25 freeze block), FAITHFULLY — this is the machinery that runs WHAT
WAS REGISTERED, not a placeholder and not a tuned variant. It composes existing,
separately-tested pieces (insider signal layer + Form 4 data layer + the
survivorship-safe SEC source); it adds no new strategy logic, no new knobs.

What the frozen config requires (all enforced below):
  * Universe = PIT S&P 500 (SurvivorshipSafeSECSource.universe()), NO sector
    exclusion (sector-neutrality is in the SIGNAL).
  * Signal (per date t x name) = trailing W=90d count of DISTINCT OPPORTUNISTIC
    open-market BUYERS (Form 4 P/A, filed_date<=t & >t-90d), NET of distinct
    opportunistic open-market SELLERS (S/D, same window/PIT), sector-demeaned
    (GICS, current map) then cross-sectionally z-scored. Cluster gate k=2.
  * Label = forward 1-month total return (non-overlapping at monthly cadence ->
    NW lags=1). Costs 10 bps/side on realized turnover. PERIODS_PER_YEAR=12.
  * Book (PRIMARY, long-vs-EW) = LONG top decile (quantile=0.10) of the signal
    among k>=2 cluster names, EW; SHORT the full priceable universe, EW.
    Dollar-neutral by construction; beta ~ 0.
  * The opportunistic arm GRADUATES; the routine arm + lag-1 arms + controls are
    reported alongside (a buying-pressure artifact must not count as information).

Order of operations (each gate aborts via sys.exit, spending NO trial):
  registration gate -> machinery gate -> DATA GATE (survivorship-safe source
  required; SOURCE-NOT-WIRED refusal for an unwired slot) -> assemble real panels
  -> POWER GATE (abort-without-N if thin) -> opportunistic arm + routine arm +
  lag-1 arms -> controls (label-shuffle placebo, -30% price-survivorship bound)
  -> PBO/MDE -> 7-gate verdict. Does NOT auto-bump N, does NOT auto-log, does NOT
  fetch real data on the import path.

POWER NOTE (the H10-specific risk): the documented edge concentrates in small-caps;
testing on the large-cap survivorship-safe S&P 500 universe is the conservative-on-
price-survivorship choice but is the POWER risk. The POWER GATE (n_obs>=60,
median basket>=5) may LEGITIMATELY abort with no trial spent — logged as a
"free-data-limitation finding," exactly the trial-#10 fee-first precedent.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from quantlab import insider, metrics, universe as uni
from quantlab.insider_data import InsiderSource
from quantlab.registry import require_runnable_registration

# --------------------------------------------------------------------------- #
# FROZEN module constants (the H10 STAGE-2 config; pinned by tests).
# --------------------------------------------------------------------------- #
WINDOW_DAYS = 90              # trailing signal window (calendar days)
CLUSTER_K = 2                 # k>=2 distinct opportunistic buyers = a cluster
QUANTILE = 0.10              # top DECILE long basket (NOT quintiles)
REBALANCE_FREQ = "ME"        # MONTHLY, month-end as-of dates
PERIODS_PER_YEAR = 12        # monthly book -> sqrt(12)
COST_BPS_PER_SIDE = 10.0     # 10 bps/side on realized turnover
LABEL_HORIZON = "1M"         # forward 1-month total return (non-overlapping)
SURVIVORSHIP_DOWN = -0.30    # trial-#2 delisting terminal return (down-scenario)
N_TRIALS_DEFAULT = 13        # the DSR uses N=13 (this is trial #13)
MIN_N_OBS = 60               # POWER floor (a): monthly periods w/ non-empty basket
MIN_BASKET = 5               # POWER floor (b): median per-date long-basket size


# --------------------------------------------------------------------------- #
# Panel assembly — long-form Form 4 buy/sell panels by CIK (survivorship-safe).
# --------------------------------------------------------------------------- #

def assemble_insider_panels(
    source, insider_source, members: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Resolve each PIT member -> CIK via the SURVIVORSHIP-SAFE path, then fetch
    Form 4 ``purchases``/``sells`` by CIK and concat into long-form panels.

    Form 4 persists by CIK after a ticker dies/renames — that is H10's whole
    survivorship-safety claim on the SIGNAL side. We therefore resolve the
    (possibly DEAD) ticker to its CIK via ``source._cik_for`` (current SEC map
    first, then the dead-name crosswalk) and fetch by CIK, never by the current
    ticker map alone (which drops dead names — the survivorship sin). The ``ticker``
    column on each row is overwritten with the PIT member symbol so the signal's
    cross-section is keyed on the universe symbol, not the (current) issuer symbol.

    Returns ``(purchases_panel, sells_panel, cik_by_ticker)``."""
    buy_frames, sell_frames, cik_by_ticker = [], [], {}
    for tkr in members:
        cik = source._cik_for(tkr)
        cik_by_ticker[tkr] = cik
        if cik is None:
            continue
        buys = insider_source.purchases(cik)
        sells = insider_source.sells(cik)
        if not buys.empty:
            buys = buys.copy()
            buys["ticker"] = tkr
            buy_frames.append(buys)
        if not sells.empty:
            sells = sells.copy()
            sells["ticker"] = tkr
            sell_frames.append(sells)
    cols = ["owner_name", "role", "shares", "value", "transaction_date",
            "ticker", "accession"]
    empty = pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="filed_date"))
    purchases = (pd.concat(buy_frames).sort_index() if buy_frames else empty.copy())
    sells = (pd.concat(sell_frames).sort_index() if sell_frames else empty.copy())
    purchases.index.name = "filed_date"
    sells.index.name = "filed_date"
    return purchases, sells, cik_by_ticker


# --------------------------------------------------------------------------- #
# Baselines (the registration requires beating BOTH: EW long-only, 12-1 momentum).
# --------------------------------------------------------------------------- #

def equal_weight_baseline(prices: pd.DataFrame, asof: pd.DatetimeIndex) -> pd.Series:
    """Long-only equal-weight book on the available universe (the EW baseline).

    pandas 2.x note: DataFrame.mean has NO ``min_count`` argument (that lives on
    .sum); ``.mean(axis=1).dropna()`` is the correct equivalent — a row with at
    least one finite return yields its mean, an all-NaN row is dropped."""
    fwd = prices.pct_change(fill_method=None).shift(-1).reindex(asof)
    return fwd.mean(axis=1).dropna()


def momentum_baseline(
    prices: pd.DataFrame, asof: pd.DatetimeIndex, monthly_px: pd.DataFrame,
) -> pd.Series:
    """12-1 momentum rank, dollar-neutral EW decile L/S on the monthly grid /
    cost convention (CLAUDE.md baseline #5).

    12-MONTH-minus-1-month momentum (trailing 12m total return skipping the most
    recent month) on the genuine MONTHLY price series, as-of aligned onto the
    month-end rebalance grid and ranked cross-sectionally there. The book trades
    at the registered monthly cadence with the same cost. Decile (QUANTILE=0.10)
    EW long-short — the registered book's quantile."""
    mom_m = (monthly_px.shift(1) / monthly_px.shift(12) - 1.0)  # 12-1 on months
    mom = mom_m.reindex(mom_m.index.union(asof)).sort_index().ffill().reindex(asof)
    sig = insider._zscore_rows(mom)
    res = insider.cluster_backtest(
        sig, prices, quantile=QUANTILE, cost_bps_per_side=COST_BPS_PER_SIDE)
    return res["net"]


# --------------------------------------------------------------------------- #
# One arm: net signal -> long-vs-EW book -> net SR / NW t / DSR.
# --------------------------------------------------------------------------- #

def run_arm(
    signal: pd.DataFrame, n_buyers_mask: pd.DataFrame, prices: pd.DataFrame,
    n_trials: int, label: str,
) -> dict:
    """Build the long-vs-EW book from a (signal, mask) pair, price it net of cost,
    and return net SR / NW t / DSR / turnover / basket sizes."""
    weights = insider.long_vs_ew_weights(
        signal, n_buyers_mask, prices.reindex(signal.index), quantile=QUANTILE)
    fwd = prices.pct_change(fill_method=None).shift(-1).reindex_like(weights)
    gross = (weights * fwd).sum(axis=1, min_count=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    net = (gross - turnover * COST_BPS_PER_SIDE / 1e4).dropna()
    sr = metrics.sharpe(net, periods=PERIODS_PER_YEAR)        # monthly -> sqrt(12)
    t_nw = metrics.newey_west_tstat(net, lags=1)              # 1m non-overlapping
    dsr = metrics.deflated_sharpe_ratio(net, n_trials=n_trials)
    # Long-basket diagnostics: a name is "long" when its weight is strictly above
    # the (negative) EW short level — i.e. it received the +0.5/n_long tilt.
    long_count = (weights > 0).sum(axis=1)
    nonempty = long_count[long_count > 0]
    return {
        "label": label, "net": net, "gross": gross.dropna(), "weights": weights,
        "sharpe": sr, "t_nw": t_nw, "dsr": dsr,
        "annual_turnover": float(turnover.sum() / max(len(weights), 1)
                                 * PERIODS_PER_YEAR),
        "n_obs": int(net.shape[0]),
        "n_nonempty_baskets": int(nonempty.shape[0]),
        "median_basket": float(nonempty.median()) if not nonempty.empty else 0.0,
    }


def _shuffle_label_placebo(
    signal: pd.DataFrame, n_buyers_mask: pd.DataFrame, prices: pd.DataFrame,
    seed: int = 13,
) -> float:
    """Label-shuffle placebo: forward returns shuffled CROSS-SECTIONALLY within
    each date, then the SAME book priced on them. |SR| must be ~0 (else leakage).
    PIT-safe: only the label is permuted, never the signal, so this measures
    pure-luck dispersion of the book, not any real edge."""
    weights = insider.long_vs_ew_weights(
        signal, n_buyers_mask, prices.reindex(signal.index), quantile=QUANTILE)
    fwd = prices.pct_change(fill_method=None).shift(-1).reindex_like(weights)
    rng = np.random.default_rng(seed)
    shuffled = fwd.copy()
    for d in shuffled.index:
        row = shuffled.loc[d]
        finite = row.dropna()
        if len(finite) > 1:
            vals = finite.to_numpy().copy()
            rng.shuffle(vals)
            shuffled.loc[d, finite.index] = vals
    gross = (weights * shuffled).sum(axis=1, min_count=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    net = (gross - turnover * COST_BPS_PER_SIDE / 1e4).dropna()
    return metrics.sharpe(net, periods=PERIODS_PER_YEAR)


def _survivorship_bounded_sr(arm: dict, prices: pd.DataFrame) -> float:
    """Price-side survivorship -30% bound: long-basket names whose price goes
    UNPRICEABLE (NaN) mid-hold are assigned the trial-#2 delisting terminal return
    (SURVIVORSHIP_DOWN) for that period, and the book SR recomputed.

    For a LONG insider-BUY signal the missing-price gap is OPTIMISTIC (names that
    went to zero drop out of the priceable universe), so this is the load-bearing
    honest caveat: a name held long whose forward return is NaN (it stopped being
    priced) earns the -30% down-scenario instead of silently contributing nothing.
    Reuses the arm's already-built weights; only the long-leg NaN forward returns
    are overridden (the EW short leg is the priceable benchmark, unchanged)."""
    weights = arm["weights"]
    fwd = prices.pct_change(fill_method=None).shift(-1).reindex_like(weights)
    bounded = fwd.copy()
    long_mask = weights > 0
    nan_long = long_mask & bounded.isna()
    bounded = bounded.mask(nan_long, SURVIVORSHIP_DOWN)
    gross = (weights * bounded).sum(axis=1, min_count=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    net = (gross - turnover * COST_BPS_PER_SIDE / 1e4).dropna()
    return metrics.sharpe(net, periods=PERIODS_PER_YEAR)


def _run_trial(source, insider_source, n_trials: int) -> dict:
    """The graded long-vs-EW run — reachable only with a survivorship-safe source."""
    # The graded run consumes a richer interface than the bare slot: universe()/
    # prices()/prices_monthly()/start/end. Surface a missing leg as a clear "slot
    # not wired" refusal (not a raw AttributeError), spending NO trial.
    for attr in ("universe", "prices", "prices_monthly", "start", "end"):
        if not hasattr(source, attr):
            sys.exit(
                f"\nSOURCE NOT WIRED: {type(source).__name__} passes the DATA GATE "
                f"(survivorship_safe) but does not implement '{attr}' — it is a slot. "
                "Connect a source exposing universe()/prices()/prices_monthly()/"
                "start/end (+ _cik_for for Form 4 resolution). The long-vs-EW "
                "harness above is proven; no trial spent (N unchanged).")

    # ---- universe: PIT S&P 500 members (NO sector exclusion) ---------------- #
    members = source.universe()
    current, _ = uni.fetch_sp500_tables()
    sectors = uni.sector_map(current, members)
    print(f"[universe] {len(members)} PIT S&P 500 members (NO sector exclusion; "
          "sector-neutrality is in the signal).")

    # ---- MONTHLY month-end as-of grid over the source's start..end ---------- #
    asof = pd.bdate_range(source.start, source.end, freq=REBALANCE_FREQ)

    # ---- Form 4 panels: buys + sells by CIK (survivorship-safe resolution) -- #
    purchases, sells, cik_by_ticker = assemble_insider_panels(
        source, insider_source, members)
    n_resolved = sum(1 for v in cik_by_ticker.values() if v is not None)
    print(f"[panels] {n_resolved}/{len(members)} members resolved to a CIK; "
          f"{purchases.shape[0]} buy rows, {sells.shape[0]} sell rows.")

    # ---- prices on the rebalance grid (delisting-inclusive) ----------------- #
    prices = source.prices(members, asof)
    monthly_px = source.prices_monthly(members)

    # ---- OPPORTUNISTIC arm (the graduation arm) ----------------------------- #
    opp_sig, opp_mask = insider.net_cluster_buy_signal(
        purchases, sells, asof, tickers=list(prices.columns),
        window_days=WINDOW_DAYS, sector_map=sectors, classify="opportunistic")
    opp = run_arm(opp_sig, opp_mask, prices, n_trials, "OPPORTUNISTIC")

    # ---- POWER GATE (on REAL data, abort WITHOUT spending N) ----------------- #
    n_obs_basket = opp["n_nonempty_baskets"]
    median_basket = opp["median_basket"]
    var_sr = 1.0 / max(opp["n_obs"], 1)
    mde_pp = metrics.expected_max_sharpe(n_trials, var_sr, opp["n_obs"])
    mde_ann = mde_pp * np.sqrt(PERIODS_PER_YEAR)             # monthly -> sqrt(12)
    print(f"[power] n_obs with non-empty long basket = {n_obs_basket} "
          f"(floor {MIN_N_OBS}); median per-date long-basket size = "
          f"{median_basket:.1f} (floor {MIN_BASKET}).")
    print(f"[power] realized MDE @ N={n_trials}, n_obs={opp['n_obs']}: net annual "
          f"SR hurdle ~{mde_ann:.3f} (DSR>=0.95 deflation, monthly sqrt(12)).")
    if n_obs_basket < MIN_N_OBS or median_basket < MIN_BASKET:
        sys.exit(
            "\nPOWER GATE: underpowered on free data "
            f"(n_obs_basket {n_obs_basket} < {MIN_N_OBS} or median basket "
            f"{median_basket:.1f} < {MIN_BASKET}) — no trial spent, log as a "
            "free-data-limitation finding (the trial-#10 fee-first precedent: we "
            "cannot test this on free data).")

    # ---- ROUTINE control arm (CMP central result) --------------------------- #
    rou_sig, rou_mask = insider.net_cluster_buy_signal(
        purchases, sells, asof, tickers=list(prices.columns),
        window_days=WINDOW_DAYS, sector_map=sectors, classify="routine")
    routine = run_arm(rou_sig, rou_mask, prices, n_trials, "ROUTINE")

    # ---- entry-lag arms (signal lagged +1 period) --------------------------- #
    opp_lag = run_arm(opp_sig.shift(1), opp_mask.shift(1).fillna(0.0), prices,
                      n_trials, "OPPORTUNISTIC-lag1")
    rou_lag = run_arm(rou_sig.shift(1), rou_mask.shift(1).fillna(0.0), prices,
                      n_trials, "ROUTINE-lag1")

    # ---- baselines ---------------------------------------------------------- #
    ew = equal_weight_baseline(prices, asof)
    mom = momentum_baseline(prices, asof, monthly_px)
    sr_ew = metrics.sharpe(ew, periods=PERIODS_PER_YEAR)
    sr_mom = metrics.sharpe(mom, periods=PERIODS_PER_YEAR)

    # ---- controls: label-shuffle placebo + -30% survivorship bound ---------- #
    shuffle_sr = _shuffle_label_placebo(opp_sig, opp_mask, prices)
    bounded_sr = _survivorship_bounded_sr(opp, prices)

    # ---- PBO across the 4-config family {opp,routine} x {lag0,lag1} ---------- #
    mat = pd.DataFrame({
        "opp_lag0": opp["net"], "routine_lag0": routine["net"],
        "opp_lag1": opp_lag["net"], "routine_lag1": rou_lag["net"],
    })
    valid = mat.dropna(how="any")
    if not valid.empty:
        mat = mat.loc[valid.index.min(): valid.index.max()]
    n_common = int(mat.dropna(how="any").shape[0])
    pbo_out, pbo_blocked = None, False
    legs_nonempty = all(
        a["net"].shape[0] > 0 for a in (opp, routine, opp_lag, rou_lag))
    if not legs_nonempty:
        pbo_blocked = True
        print("[pbo] *** BLOCKED *** a leg of the 4-config family is empty; "
              "never grade on a degenerate matrix (the B5 rule from H1).")
    elif n_common >= 4:
        from quantlab import pbo
        nsp = min(6, n_common - (n_common % 2))
        nsp = max(2, nsp)
        try:
            pbo_out = pbo.cscv_pbo(mat, n_splits=nsp)
        except ValueError as exc:
            print(f"[pbo] not computable ({exc}).")
            pbo_blocked = True
    else:
        print(f"[pbo] not computable (only {n_common} common observations).")
        pbo_blocked = True

    # ---- report ------------------------------------------------------------- #
    print("\n=== H10 long-vs-EW result (opportunistic insider cluster buys) ===")
    for arm in (opp, routine, opp_lag, rou_lag):
        print(f"  {arm['label']:>20}: net SR {arm['sharpe']:+.3f}  "
              f"t_NW {arm['t_nw']:+.2f}  DSR {arm['dsr']:.3f}  "
              f"turnover {arm['annual_turnover']:.2f}/yr  n_obs {arm['n_obs']}  "
              f"median basket {arm['median_basket']:.1f}")
    print(f"  baselines: EW SR {sr_ew:+.3f} | 12-1 mom SR {sr_mom:+.3f}")
    print(f"  controls: label-shuffle placebo SR {shuffle_sr:+.3f} (|SR|<0.3) | "
          f"survivorship -30% bounded SR {bounded_sr:+.3f} (base {opp['sharpe']:+.3f})")
    if pbo_out is not None:
        print(f"  PBO {pbo_out['pbo']:.3f}  (n_configs {pbo_out['n_configs']}, "
              f"n_obs {pbo_out['n_obs']}, splits {pbo_out['n_splits']})")
    else:
        print("  PBO: not graded (4-config family incomplete -> verdict BLOCKED).")

    # ---- 7-gate pre-registered verdict (graduate iff ALL hold on the OPP arm) #
    t_nw_ok = (opp["t_nw"] is not None and not np.isnan(opp["t_nw"])
               and opp["t_nw"] >= 2.0)
    sr_pos = opp["sharpe"] > 0
    beats_baselines = opp["sharpe"] > sr_ew and opp["sharpe"] > sr_mom
    dsr_ok = opp["dsr"] >= 0.95
    pbo_ok = (not pbo_blocked) and (pbo_out is not None) and (pbo_out["pbo"] <= 0.5)
    # entry-lag gate: lag-1 SR >= 0.5*lag-0 SR AND lag-1 SR > 0. Self-contained:
    # also require lag-0 SR > 0, so the "retains >=50%" ratio is never satisfied
    # vacuously by a NEGATIVE lag-0 baseline (gate 2 already backstops graduation,
    # but the printed PASS/FAIL must be honest on its own).
    entry_lag_ok = (opp["sharpe"] > 0
                    and opp_lag["sharpe"] > 0
                    and opp_lag["sharpe"] >= 0.5 * opp["sharpe"])
    # routine differential: opp SR > routine SR AND routine |t_NW| < 2.
    routine_t = routine["t_nw"]
    routine_t_ins = (routine_t is None or np.isnan(routine_t)
                     or abs(routine_t) < 2.0)
    routine_diff_ok = (opp["sharpe"] > routine["sharpe"]) and routine_t_ins
    # price-survivorship bound must NOT flip the verdict: the bounded book must
    # still be SR>0 AND beat both baselines (the kill rule is a FLIP base->fail).
    bound_beats = bounded_sr > 0 and bounded_sr > sr_ew and bounded_sr > sr_mom
    survivorship_ok = bound_beats  # if base passes these, bound must too (no flip)

    graduate = (
        t_nw_ok and sr_pos and beats_baselines and dsr_ok and pbo_ok
        and entry_lag_ok and routine_diff_ok and survivorship_ok
    )

    print("\n=== PRE-REGISTERED VERDICT (OPPORTUNISTIC arm; 7-gate conjunction) ===")
    print(f"  1. t_NW >= +2 .......... {opp['t_nw']:+.2f}  -> "
          f"{'PASS' if t_nw_ok else 'FAIL'}")
    print(f"  2. net SR>0 & > baselin. {opp['sharpe']:+.3f} vs EW {sr_ew:+.3f}, "
          f"mom {sr_mom:+.3f}  -> {'PASS' if (sr_pos and beats_baselines) else 'FAIL'}")
    print(f"  3. DSR >= 0.95 ......... {opp['dsr']:.3f}  -> "
          f"{'PASS' if dsr_ok else 'FAIL'}")
    pbo_str = f"{pbo_out['pbo']:.3f}" if pbo_out is not None else "BLOCKED"
    print(f"  4. PBO <= 0.5 (4-cfg) .. {pbo_str}  -> {'PASS' if pbo_ok else 'FAIL'}")
    print(f"  5. entry-lag gate ...... lag1 SR {opp_lag['sharpe']:+.3f} "
          f"(>= 0.5*{opp['sharpe']:+.3f} & >0)  -> "
          f"{'PASS' if entry_lag_ok else 'FAIL'}")
    rt = f"{routine_t:+.2f}" if (routine_t is not None and not np.isnan(routine_t)) else "n/a"
    print(f"  6. routine differential. opp SR {opp['sharpe']:+.3f} > routine SR "
          f"{routine['sharpe']:+.3f} & routine |t_NW| {rt} < 2  -> "
          f"{'PASS' if routine_diff_ok else 'FAIL'}")
    print(f"  7. -30% bound no flip .. bounded SR {bounded_sr:+.3f} still beats "
          f"baselines  -> {'PASS' if survivorship_ok else 'FAIL'}")
    print(f"  >>> H10 {'GRADUATES' if graduate else 'does NOT graduate'} "
          "(log the row by hand; N becomes 13).")

    # Structured result — the verdict is EXACTLY the 7-gate conjunction (no hidden
    # gate) for programmatic adjudication / regression tests.
    return {
        "opp": opp, "routine": routine, "opp_lag": opp_lag, "routine_lag": rou_lag,
        "sr_ew": sr_ew, "sr_mom": sr_mom,
        "shuffle_sr": shuffle_sr, "bounded_sr": bounded_sr,
        "pbo": pbo_out, "pbo_blocked": pbo_blocked,
        "mde_ann": mde_ann, "n_obs": opp["n_obs"],
        "median_basket": median_basket, "n_obs_basket": n_obs_basket,
        "gates": {
            "t_nw": t_nw_ok, "sr_pos": sr_pos, "beats_baselines": beats_baselines,
            "dsr": dsr_ok, "pbo": pbo_ok, "entry_lag": entry_lag_ok,
            "routine_diff": routine_diff_ok, "survivorship": survivorship_ok,
        },
        "graduate": bool(graduate),
        "universe": members,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypothesis", default="H10")
    ap.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    ap.add_argument(
        "--source",
        choices=["free_xwalk", "free_sec", "compustat"],
        default="free_xwalk",
    )
    args = ap.parse_args()

    # 1) Registration gate (law #3): H10 must be PROPOSED.
    try:
        require_runnable_registration(args.hypothesis)
    except RuntimeError as exc:
        sys.exit(f"REGISTRATION GATE: {exc}")
    print(f"[registration] {args.hypothesis} verified PROPOSED.")

    # 2) Machinery gate (law #4): the EXISTING insider.machinery_gate — planted-
    #    opportunistic world recovered, null rejected, paired per seed.
    print("[gate] synthetic insider world: planted-opportunistic must beat null "
          "(paired)...")
    gate = insider.machinery_gate()
    for s, p, n in zip((7, 11, 23), gate["planted_sr"], gate["null_sr"]):
        print(f"  seed {s}: planted SR {p:+.2f} | null SR {n:+.2f}")
    if not gate["passed"]:
        sys.exit(f"MACHINERY GATE FAILED: differential {min(gate['diffs']):.2f} "
                 "<= 0.5 — harness cannot tell the planted cluster-buy alpha from "
                 "its absence; abort.")
    print(f"[gate] PASS (min paired differential {min(gate['diffs']):.2f})")

    # 3) DATA GATE: a graded trial requires a survivorship-safe source.
    if args.source in ("free_sec", "free_xwalk"):
        from quantlab.fundamentals_data import FreeSECSource
        from quantlab.sec_xwalk_source import SurvivorshipSafeSECSource
        sources = {"free_sec": FreeSECSource, "free_xwalk": SurvivorshipSafeSECSource}
    else:
        from quantlab.fundamentals_data import CompustatSource
        sources = {"compustat": CompustatSource}
    source = sources[args.source]()
    if not source.survivorship_safe:
        sys.exit(
            "\nDATA GATE: the free SEC source is SURVIVORSHIP-BLOCKED -- its "
            "ticker->CIK map is current-only (~73% coverage; dead/renamed names "
            "dropped, audit 2026-06-14). A graded H10 trial on it would re-commit "
            "trial #1's survivorship sin, so this run is REFUSED and spends no "
            "trial (N unchanged).\n  -> Use --source free_xwalk (free, "
            "survivorship-safe via the name->CIK crosswalk). Form 4s persist by "
            "CIK after a ticker dies, so the SIGNAL side is survivorship-safe — "
            "the whole point of H10.")

    insider_source = InsiderSource()
    _run_trial(source, insider_source, args.n_trials)


if __name__ == "__main__":
    main()
