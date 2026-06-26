"""H12 graded run (trial #13) — the PRE-REGISTERED BROADER-BASKET opportunistic
insider cluster-buying long-vs-EW book (long ALL eligible cluster names vs EW).

Executes the H12 FROZEN CONFIG in writeup/preregistered_hypotheses.md (the block
"### H12: Broader-basket opportunistic insider cluster-buying ...", frozen
2026-06-25), FAITHFULLY. H12 is the POWERED redesign of H10: H10's TOP-DECILE book
(quantile=0.10, ~2-name basket) POWER-ABORTED pre-spend (it touched NO forward
returns); H12 longs the FULL eligible (k>=2) cluster set (quantile=1.0). This is a
FRESH pre-registration motivated by buy-DENSITY (a power requirement), NOT a
relaxation of a failed result — H10 had no result to relax.

The ONLY two differences from H10 (everything else is byte-identical):
  (a) DATA SOURCE for the SIGNAL is ``BulkInsiderSource`` (the SEC bulk Form 3/4/5
      data sets), NOT the raw-XML crawl ``InsiderSource`` (which is recent-only /
      incomplete for prolific filers — see its COMPLETENESS WARNING — so the crawl
      must NOT be used here);
  (b) the book longs the FULL eligible cluster set (QUANTILE = 1.0), not the top
      decile.
Universe (PIT S&P 500), 90d window, k>=2 cluster gate, monthly 1-month-forward
label, 10 bps/side, the 7-gate verdict, the routine/lag/shuffle/survivorship
controls, the 4-config PBO, and the power gate are IDENTICAL to H10 — and the
quantile-agnostic helpers are REUSED by import from ``run_h10_trial``.

What the frozen config requires (all enforced below):
  * Universe = PIT S&P 500 (SurvivorshipSafeSECSource.universe()), NO sector
    exclusion (sector-neutrality is in the SIGNAL).
  * Signal (per date t x name) = trailing W=90d count of DISTINCT OPPORTUNISTIC
    open-market BUYERS (Form 4 P/A, filed_date<=t & >t-90d), NET of distinct
    opportunistic open-market SELLERS (S/D, same window/PIT), sector-demeaned
    (GICS, current map) then cross-sectionally z-scored. Cluster gate k=2. The
    SIGNAL is keyed on the PIT universe member symbol (via issuer_cik), so the
    cross-section aligns with the price panel.
  * Label = forward 1-month total return (non-overlapping at monthly cadence ->
    NW lags=1). Costs 10 bps/side on realized turnover. PERIODS_PER_YEAR=12.
  * Book (PRIMARY, long-vs-EW) = LONG equal-weight ALL eligible (k>=2) cluster
    names (quantile=1.0); SHORT the full priceable universe, EW. Dollar-neutral
    by construction; beta ~ 0.
  * The opportunistic arm GRADUATES; the routine arm + lag-1 arms + controls are
    reported alongside (a buying-pressure artifact must not count as information).

Order of operations (each gate aborts via sys.exit, spending NO trial):
  registration gate -> machinery gate -> DATA GATE (both sources concrete:
  SurvivorshipSafeSECSource for universe+prices, BulkInsiderSource for the signal)
  -> assemble real panels (cik->member re-key) -> POWER GATE (abort-without-N if
  thin) -> opportunistic arm + routine arm + lag-1 arms -> controls (label-shuffle
  placebo, -30% price-survivorship bound) -> PBO/MDE -> 7-gate verdict. Does NOT
  auto-bump N, does NOT auto-log, does NOT fetch real data on the import path.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))   # to import run_h10_trial helpers

import numpy as np
import pandas as pd

from quantlab import insider, metrics, universe as uni
from quantlab.insider_bulk import BulkInsiderSource
from quantlab.registry import require_runnable_registration

# REUSE the quantile-AGNOSTIC H10 helpers by import (no network at H10 import).
# Reused VERBATIM: equal_weight_baseline, momentum_baseline (the baselines keep
# their own decile per the spec — "SAME as H10"), and _survivorship_bounded_sr
# (operates on already-built weights, so it is quantile-free). run_arm and
# _shuffle_label_placebo are RE-DEFINED below because the H10 versions close over
# H10's module-level QUANTILE=0.10; H12 needs quantile=1.0 (the whole point).
import run_h10_trial as h10
from run_h10_trial import (
    equal_weight_baseline,
    momentum_baseline,
    _survivorship_bounded_sr,
)

# --------------------------------------------------------------------------- #
# FROZEN module constants (the H12 config; pinned by tests). IDENTICAL to H10
# EXCEPT QUANTILE = 1.0 (long ALL eligible, not the top decile).
# --------------------------------------------------------------------------- #
WINDOW_DAYS = 90              # trailing signal window (calendar days)
CLUSTER_K = 2                 # k>=2 distinct opportunistic buyers = a cluster
QUANTILE = 1.0               # LONG ALL eligible (k>=2) cluster names (NOT a decile)
REBALANCE_FREQ = "ME"        # MONTHLY, month-end as-of dates
PERIODS_PER_YEAR = 12        # monthly book -> sqrt(12)
COST_BPS_PER_SIDE = 10.0     # 10 bps/side on realized turnover
LABEL_HORIZON = "1M"         # forward 1-month total return (non-overlapping)
SURVIVORSHIP_DOWN = -0.30    # trial-#2 delisting terminal return (down-scenario)
N_TRIALS_DEFAULT = 13        # the DSR uses N=13 (this is trial #13)
MIN_N_OBS = 60               # POWER floor (a): monthly periods w/ non-empty basket
MIN_BASKET = 5               # POWER floor (b): median per-date long-basket size


# --------------------------------------------------------------------------- #
# Panel assembly — bulk Form 4 buy/sell panels, re-keyed CIK -> PIT member symbol.
# --------------------------------------------------------------------------- #

def assemble_bulk_panels(
    source, bulk, members: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, str]]:
    """Fetch bulk Form 4 ``transactions`` for the universe's issuer CIK set and
    re-key each row's ``ticker`` onto the PIT MEMBER symbol via ``issuer_cik``.

    The bulk source carries the ISSUER trading symbol in ``ticker`` (the symbol the
    issuer reported on the form, which may be stale / a renamed name). The graded
    book must key the signal cross-section on the PIT UNIVERSE MEMBER symbol so it
    aligns with the price panel (which is indexed by member symbol). We therefore:
      1. resolve each PIT member -> CIK via the SURVIVORSHIP-SAFE path
         (``source._cik_for``: current SEC map first, then the dead-name crosswalk);
      2. pull buys/sells for that CIK set with ``with_cik=True`` (one bulk call per
         leg, not one per member — the bulk source's whole cost advantage);
      3. build the inverse map ``member_by_cik`` (issuer_cik -> member symbol) and
         OVERWRITE each row's ``ticker`` with the member symbol;
      4. drop any row whose ``issuer_cik`` is not in the universe.

    Form 4s persist by CIK after a ticker dies/renames, so resolving by CIK (never
    by the current ticker map alone, which drops dead names — the survivorship sin)
    keeps the SIGNAL side survivorship-safe, exactly H10's claim.

    Returns ``(purchases_panel, sells_panel, cik_by_member, member_by_cik)``."""
    cik_by_member: dict[str, str] = {}
    for tkr in members:
        cik = source._cik_for(tkr)
        if cik is not None:
            cik_by_member[tkr] = cik

    # Inverse map issuer_cik -> member symbol. If two members map to one CIK
    # (rare: e.g. a dual-class ticker resolving to the same issuer), keep EITHER
    # (the first seen) and log the collision — the spec authorizes this.
    member_by_cik: dict[str, str] = {}
    for tkr, cik in cik_by_member.items():
        if cik in member_by_cik:
            print(f"[panels] CIK collision: {cik} maps to both "
                  f"{member_by_cik[cik]} and {tkr}; keeping {member_by_cik[cik]}.")
            continue
        member_by_cik[cik] = tkr

    ciks = sorted(set(cik_by_member.values()))
    cols = ["owner_name", "role", "shares", "value", "transaction_date",
            "ticker", "accession"]
    empty = pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="filed_date"))

    def _rekey(kind: str) -> pd.DataFrame:
        if not ciks:
            return empty.copy()
        raw = bulk.transactions(ciks, kind=kind, with_cik=True)
        if raw.empty:
            return empty.copy()
        df = raw.copy()
        # member symbol via issuer_cik; rows whose cik isn't in the universe drop.
        df["ticker"] = df["issuer_cik"].map(member_by_cik)
        df = df[df["ticker"].notna()]
        df = df.drop(columns=["issuer_cik"])
        df.index.name = "filed_date"
        return df[cols] if not df.empty else empty.copy()

    purchases = _rekey("P")
    sells = _rekey("S")
    return purchases, sells, cik_by_member, member_by_cik


# --------------------------------------------------------------------------- #
# One arm: net signal -> long-ALL-eligible-vs-EW book -> net SR / NW t / DSR.
# (Re-defined from H10's run_arm because H12 longs quantile=1.0, not 0.10 — the
#  metric computation is otherwise IDENTICAL: net SR @ periods=12, NW t lags=1,
#  DSR @ N=13, turnover, n_obs, median_basket.)
# --------------------------------------------------------------------------- #

def run_arm(
    signal: pd.DataFrame, n_buyers_mask: pd.DataFrame, prices: pd.DataFrame,
    n_trials: int, label: str,
) -> dict:
    """Build the long-ALL-eligible-vs-EW book from a (signal, mask) pair, price it
    net of cost, and return net SR / NW t / DSR / turnover / basket sizes.

    QUANTILE=1.0 -> ``insider.long_vs_ew_weights`` longs EVERY k>=2 cluster name
    (not the top decile). Metric computation mirrors ``run_h10_trial.run_arm``."""
    weights = insider.long_vs_ew_weights(
        signal, n_buyers_mask, prices.reindex(signal.index), quantile=QUANTILE)
    fwd = prices.pct_change(fill_method=None).shift(-1).reindex_like(weights)
    gross = (weights * fwd).sum(axis=1, min_count=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    net = (gross - turnover * COST_BPS_PER_SIDE / 1e4).dropna()
    sr = metrics.sharpe(net, periods=PERIODS_PER_YEAR)        # monthly -> sqrt(12)
    t_nw = metrics.newey_west_tstat(net, lags=1)              # 1m non-overlapping
    dsr = metrics.deflated_sharpe_ratio(net, n_trials=n_trials)
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
    each date, then the SAME (quantile=1.0) book priced on them. |SR| must be ~0
    (else leakage). PIT-safe: only the label is permuted, never the signal.

    Re-defined from H10 only to use H12's QUANTILE=1.0; logic is identical."""
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


def _run_trial(source, bulk, n_trials: int) -> dict:
    """The graded long-ALL-eligible-vs-EW run — reachable only with a
    survivorship-safe price source + the bulk insider signal source."""
    # The graded run consumes a richer price interface than the bare slot:
    # universe()/prices()/prices_monthly()/start/end. Surface a missing leg as a
    # clear "slot not wired" refusal (not a raw AttributeError), spending NO trial.
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
    # source.start/end are STRINGS (e.g. "2010-01-01"); pd.date_range parses them.
    asof = pd.date_range(source.start, source.end, freq=REBALANCE_FREQ)

    # ---- bulk Form 4 panels: buys + sells, re-keyed CIK -> member symbol ----- #
    purchases, sells, cik_by_member, member_by_cik = assemble_bulk_panels(
        source, bulk, members)
    n_resolved = len(cik_by_member)
    print(f"[panels] {n_resolved}/{len(members)} members resolved to a CIK; "
          f"{len(member_by_cik)} distinct CIKs; "
          f"{purchases.shape[0]} buy rows, {sells.shape[0]} sell rows "
          "(ticker re-keyed to PIT member symbol via issuer_cik).")

    # ---- prices on the rebalance grid (delisting-inclusive) ----------------- #
    prices = source.prices(members, asof)
    monthly_px = source.prices_monthly(members)

    # ---- OPPORTUNISTIC arm (the graduation arm), QUANTILE=1.0 --------------- #
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
            "free-data-limitation finding. (H12 was EXPECTED to pass: the probe "
            "measured median ~24 cluster-eligible firms/month over 197 months.)")

    # ---- ROUTINE control arm (long-ALL-cluster, built on ROUTINE clusters) -- #
    rou_sig, rou_mask = insider.net_cluster_buy_signal(
        purchases, sells, asof, tickers=list(prices.columns),
        window_days=WINDOW_DAYS, sector_map=sectors, classify="routine")
    routine = run_arm(rou_sig, rou_mask, prices, n_trials, "ROUTINE")

    # ---- entry-lag arms (signal lagged +1 period) --------------------------- #
    opp_lag = run_arm(opp_sig.shift(1), opp_mask.shift(1).fillna(0.0), prices,
                      n_trials, "OPPORTUNISTIC-lag1")
    rou_lag = run_arm(rou_sig.shift(1), rou_mask.shift(1).fillna(0.0), prices,
                      n_trials, "ROUTINE-lag1")

    # ---- baselines (SAME as H10: EW long-only + 12-1 momentum decile L/S) ---- #
    ew = equal_weight_baseline(prices, asof)
    mom = momentum_baseline(prices, asof, monthly_px)
    sr_ew = metrics.sharpe(ew, periods=PERIODS_PER_YEAR)
    sr_mom = metrics.sharpe(mom, periods=PERIODS_PER_YEAR)

    # ---- controls: label-shuffle placebo + -30% survivorship bound ---------- #
    shuffle_sr = _shuffle_label_placebo(opp_sig, opp_mask, prices)
    bounded_sr = _survivorship_bounded_sr(opp, prices)      # quantile-free (reused)

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
    print("\n=== H12 long-ALL-cluster-vs-EW result (opportunistic insider clusters) ===")
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
    # IDENTICAL conjunction to H10 (zero relaxation; the SAME bar).
    t_nw_ok = (opp["t_nw"] is not None and not np.isnan(opp["t_nw"])
               and opp["t_nw"] >= 2.0)
    sr_pos = opp["sharpe"] > 0
    beats_baselines = opp["sharpe"] > sr_ew and opp["sharpe"] > sr_mom
    dsr_ok = opp["dsr"] >= 0.95
    pbo_ok = (not pbo_blocked) and (pbo_out is not None) and (pbo_out["pbo"] <= 0.5)
    entry_lag_ok = (opp["sharpe"] > 0
                    and opp_lag["sharpe"] > 0
                    and opp_lag["sharpe"] >= 0.5 * opp["sharpe"])
    routine_t = routine["t_nw"]
    routine_t_ins = (routine_t is None or np.isnan(routine_t)
                     or abs(routine_t) < 2.0)
    routine_diff_ok = (opp["sharpe"] > routine["sharpe"]) and routine_t_ins
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
    print(f"  >>> H12 {'GRADUATES' if graduate else 'does NOT graduate'} "
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
        "cik_by_member": cik_by_member, "member_by_cik": member_by_cik,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypothesis", default="H12")
    ap.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    args = ap.parse_args()

    # 1) Registration gate (law #3): H12 must be PROPOSED.
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

    # 3) DATA GATE: survivorship-safe prices (SurvivorshipSafeSECSource) +
    #    BulkInsiderSource for the SIGNAL (the crawl is recent-only/incomplete and
    #    must NOT be used). Both are concrete — no "source not wired" slot here.
    from quantlab.sec_xwalk_source import SurvivorshipSafeSECSource
    source = SurvivorshipSafeSECSource()
    if not source.survivorship_safe:
        sys.exit(
            "\nDATA GATE: the price source is not survivorship-safe — a graded "
            "H12 trial on it would re-commit the survivorship sin; REFUSED, no "
            "trial spent (N unchanged).")
    bulk = BulkInsiderSource(start=source.start, end=source.end)

    _run_trial(source, bulk, args.n_trials)


if __name__ == "__main__":
    main()
