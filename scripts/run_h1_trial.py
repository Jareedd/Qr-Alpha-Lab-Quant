"""H1 graded run (trial #12) — the PRE-REGISTERED two-arm quality book.

Executes the registration in writeup/preregistered_hypotheses.md (H1 + the
2026-06-16 and 2026-06-24 amendments), FAITHFULLY — this is the machinery that
runs WHAT WAS REGISTERED, not a placeholder and not a tuned variant. It composes
existing, separately-tested pieces; it adds no new strategy logic.

What the registration requires (all enforced below):
  * Profitability = CBOP/A (cash-based operating profitability, NET of accruals)
    -> RAW signal = cbop_signal(z).
  * Universe = PIT S&P EXCLUDING GICS Financials AND Real Estate (universe.sector_map).
  * Book = VALUE-WEIGHTED QUINTILE long-short (quality_weights_vw; market_cap =
    price x shares), quarterly rebalance, current-assets denominator (lagged
    assets is the declared robustness leg, run by the data layer when wired).
  * TWO ARMS on the same universe / dates / costs:
      RAW     = cbop_signal
      NEUTRAL = value_neutralized_signal vs a trailing past-only HML loading
                (rolling_factor_betas on monthly FF HML aligned to month-ends,
                fit_intercept=True) + a ones column for dollar-neutrality.
    H1 GRADUATES ON THE NEUTRAL ARM (a quality claim that is merely value
    re-labeled must not count); the RAW arm is reported alongside.
  * Adjudication: NEUTRAL arm t_NW >= 2 AND net SR > both baselines (EW, 12-1
    momentum) AND DSR >= 0.95 at N=12 AND PBO <= 0.5 (cscv_pbo on the {raw,neutral}
    net-return matrix). MDE = the N=12 net-annual-Sharpe hurdle for the realized
    n_obs (expected_max_sharpe), printed BEFORE the verdict.
  * The synthetic machinery gate (planted_quality recovered / null_quality
    rejected, paired) runs FIRST, in-env (law #4).
  * The synthetic NEUTRALIZATION gate (amendment 2026-06-24 clause C: a value-
    disguised edge COLLAPSES when neutralized, a value-orthogonal edge SURVIVES,
    SR-matched + placebo-controlled) runs NEXT, in-env. Because H1 graduates ONLY
    on the NEUTRAL arm, neutralization must be proven able to discriminate TODAY
    or no raw-vs-neutral number is trusted -> ABORT, no trial spent.

Order of operations: registration gate -> machinery gate -> NEUTRALIZATION gate
-> DATA GATE (refuse a survivorship-blocked source, spending NO trial) -> assemble
panels -> two arms -> PBO/MDE -> verdict. Does NOT auto-bump N, does NOT auto-log,
does NOT fetch real data on the import path.

HML-absent policy (M2): if the monthly FF HML file is absent, the runner ABORTS
the NEUTRAL grade rather than substituting a market+sector proxy — that proxy is
NOT validated by the two-world (quality_is_value / quality_orthogonal) gate, and
H1 graduates ONLY on the NEUTRAL arm. A RAW-only report is acceptable; a graded
NEUTRAL verdict on an unvalidated neutralization is not. The FF file IS present in
this environment, so the live HML path is the default.
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from quantlab import fundamentals, metrics, risk_model, universe as uni
from quantlab.fundamentals_data import CompustatSource, FreeSECSource
from quantlab.registry import require_runnable_registration

# Sectors the registration excludes (no CoGS line; large-cap-by-construction cut).
EXCLUDED_SECTORS = ("Financials", "Real Estate")
REBALANCE_FREQ = "BQE"     # quarterly business-quarter-end (slow, low turnover)
# Annualization factor for the H1 BOOK: REBALANCE_FREQ == "BQE" => ~4 periods/yr,
# so SR annualizes by sqrt(4) and the MDE by sqrt(4). This is DELIBERATELY NOT
# fundamentals.PERIODS_PER_YEAR (== 12), which is correct only for the MONTHLY
# synthetic machinery gate; using 12 on a quarterly book inflates SR ~sqrt(3)
# (~1.73x) and the MDE alike (B4 fix). The book's cadence sets its annualizer.
PERIODS_PER_YEAR = 4
COST_BPS_PER_SIDE = 10.0
# Fundamentals staleness cap (post-trial-#12 robustness fix). A filing's value is
# carried forward at most this many days. WITHOUT it, reindex(ffill) carried a dead
# company's last fundamental FOREVER and paired it with prices from a REASSIGNED
# ticker (Monsanto's 2017 financials x the 2021 "MON" entity, + 12 others). 18
# months exceeds the annual filing cycle, so live names are never dropped; a name
# that stops filing for >18mo goes NaN -> drops out (the reassigned tail is unreachable).
STALENESS_DAYS = 548
QUANTILE = 0.2             # quintiles (frozen — NOT deciles)
# Registered N=12 graduation hurdle from the 2026-06-16 amendment (success
# criterion 6): ~0.90 net annual SR (daily, ~15-yr). Printed next to the
# recomputed MDE so the two are visibly reconciled, not silently divergent (M5).
REGISTERED_HURDLE_ANN = 0.90
HML_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "scratch_refute",
    "F-F_Research_Data_5_Factors_2x3.csv",
)


# --------------------------------------------------------------------------- #
# Panel assembly — CBOP/A + market-cap, PIT (pit_feature_panels-style).
# --------------------------------------------------------------------------- #

def cbop_and_cap_panels(
    source, tickers: list[str], asof: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    """Assemble (asof x ticker) CBOP/A and market-cap panels, filing-date PIT.

    Each cell is the latest filing on or before that date. Flow numerators (GP,
    NI, CFO) are ANNUAL-ONLY before division by point-in-time assets (the
    2026-06-16 annualization fix); GP falls back to Revenue - CoGS via the same
    private helper the GP/A path uses. Market cap = price x shares: price comes
    from the source's delisting-inclusive grid, shares are the PIT filed count.
    Names missing any required leg drop out (NaN) rather than being imputed.

    Two CBOP/A panels are returned (B5 PBO family):
      * ``cbop_a``        — current-assets denominator (the registration's PRIMARY).
      * ``cbop_a_lagged`` — assets LAGGED one ANNUAL filing (each date divides the
        flow numerators by the PRIOR annual Assets; the Hou-Xue-Zhang 2020
        insignificant cut, the declared robustness leg). The lag is on the
        annual-only assets series, then forward-filled to ``asof``; strictly
        past-only (an earlier filing's assets) so it adds no lookahead."""
    cbop_a, cbop_a_lagged, shares = {}, {}, {}
    for t in tickers:
        assets = source.field_series(t, "assets")
        if assets.empty:
            continue
        gp = fundamentals._gross_profit(source, t)
        ni = source.field_series(t, "net_income", annual_only=True)
        cfo = source.field_series(t, "cfo", annual_only=True)
        if gp.empty or ni.empty or cfo.empty:
            continue
        cbop_a[t] = fundamentals.cbop_over_assets(gp, ni, cfo, assets).reindex(
            asof, method="ffill", tolerance=pd.Timedelta(days=STALENESS_DAYS))
        # Lagged-assets leg: divide the SAME flow numerators by the PRIOR annual
        # filing's assets. Lag the annual-only assets series by one filing; if no
        # annual assets exist, the name simply has no lagged leg (NaN, dropped).
        assets_annual = source.field_series(t, "assets", annual_only=True)
        if not assets_annual.empty:
            assets_lagged = assets_annual.sort_index().shift(1).dropna()
            if not assets_lagged.empty:
                cbop_a_lagged[t] = fundamentals.cbop_over_assets(
                    gp, ni, cfo, assets_lagged).reindex(
                        asof, method="ffill",
                        tolerance=pd.Timedelta(days=STALENESS_DAYS))
        sh = source.field_series(t, "shares")           # any form (stock value)
        if not sh.empty:
            shares[t] = sh.reindex(
                asof, method="ffill", tolerance=pd.Timedelta(days=STALENESS_DAYS))
    cbop_df = pd.DataFrame(cbop_a, index=asof)
    cbop_lag_df = pd.DataFrame(cbop_a_lagged, index=asof)
    shares_df = pd.DataFrame(shares, index=asof)
    prices = source.prices(list(cbop_df.columns), asof)
    cols = cbop_df.columns.intersection(prices.columns).intersection(shares_df.columns)
    market_cap = (prices[cols] * shares_df[cols]).reindex(asof)
    return {"cbop_a": cbop_df, "cbop_a_lagged": cbop_lag_df,
            "market_cap": market_cap, "prices": prices}


# --------------------------------------------------------------------------- #
# Value loading — trailing past-only HML beta (law #1), with declared fallback.
# --------------------------------------------------------------------------- #

def hml_loading_panel(
    monthly_px: pd.DataFrame, asof: pd.DatetimeIndex,
) -> tuple[pd.DataFrame | None, str]:
    """Trailing past-only HML beta of each name, aligned to the rebalance grid.

    Takes the genuine (month-end x ticker) MONTHLY price grid (B3 fix: the daily
    grid resampled to month-end, NOT the quarterly ``asof`` grid resampled — that
    produced an all-NaN beta panel and silently degenerated the NEUTRAL arm to a
    plain demean). Returns (loading_panel_on_asof, mode).

    If the monthly FF file exists: monthly returns are regressed on monthly HML via
    rolling_factor_betas (lookback 36, min 18 months, fit_intercept=True — REQUIRED,
    law #1 / the K=1 reduction), then the per-month HML beta is as-of aligned to
    ``asof`` (last beta <= date). Each beta_t uses only returns through t
    (point-in-time safe). If the file is absent, returns (None, 'fallback') so the
    caller switches to market+sector neutralization and declares it."""
    if not os.path.exists(HML_FILE):
        return None, "fallback_market_sector"
    from quantlab import ff_factors
    ff = ff_factors.load_ff_factors_monthly(HML_FILE)
    hml = ff[["HML"]]
    monthly_ret = monthly_px.pct_change(fill_method=None)
    # Restrict the factor to the price window; rolling_factor_betas reindexes the
    # factor onto the asset index (asset index authoritative) internally.
    fb = risk_model.rolling_factor_betas(
        monthly_ret, hml.reindex(monthly_ret.index), lookback=36, min_periods=18,
        fit_intercept=True,
    )
    beta_monthly = fb["HML"]                      # (month-end x ticker)
    # As-of align to the rebalance grid: the latest month-end beta <= each asof.
    panel = beta_monthly.reindex(beta_monthly.index.union(asof)).sort_index()
    panel = panel.ffill().reindex(asof)
    return panel, "hml_monthly"


# --------------------------------------------------------------------------- #
# Baselines (the registration requires beating BOTH: EW quality, 12-1 momentum).
# --------------------------------------------------------------------------- #

def equal_weight_baseline(prices: pd.DataFrame, asof: pd.DatetimeIndex) -> pd.Series:
    """Long-only equal-weight book on the available universe (the EW baseline).

    pandas 2.x note: DataFrame.mean has NO ``min_count`` argument (that lives on
    .sum); calling it crashes on pandas >= 2.2. ``.mean(axis=1).dropna()`` is the
    correct equivalent — a row with at least one finite return yields its mean,
    an all-NaN row yields NaN and is dropped."""
    fwd = prices.pct_change(fill_method=None).shift(-1).reindex(asof)
    return fwd.mean(axis=1).dropna()


def momentum_baseline(
    prices: pd.DataFrame, asof: pd.DatetimeIndex, market_cap: pd.DataFrame,
    monthly_px: pd.DataFrame,
) -> pd.Series:
    """12-1 momentum rank, VALUE-WEIGHTED quintile L/S on the quarterly grid /
    cost convention (CLAUDE.md baseline #5).

    M1 fix: the signal is 12-MONTH-minus-1-month momentum — trailing 12-month total
    return skipping the most recent month — computed on the genuine MONTHLY price
    series (``monthly_px``), NOT 12 shifts on the quarterly grid (which was a
    12-QUARTER / 3-year lookback, the wrong object). The monthly momentum is then
    as-of aligned onto the quarterly rebalance grid and ranked cross-sectionally
    there, so the book trades at the registered quarterly cadence."""
    mom_m = (monthly_px.shift(1) / monthly_px.shift(12) - 1.0)  # 12-1 on months
    # As-of align the monthly momentum onto the quarterly rebalance grid: the
    # latest month-end momentum <= each asof (past-only).
    mom = mom_m.reindex(mom_m.index.union(asof)).sort_index().ffill().reindex(asof)
    sig = fundamentals._zscore_rows(mom)
    res = fundamentals.quality_backtest(
        sig, prices, quantile=QUANTILE, cost_bps_per_side=COST_BPS_PER_SIDE,
        market_cap=market_cap.reindex(asof),
    )
    return res["net"]


# --------------------------------------------------------------------------- #
# One arm: signal -> VW quintile L/S -> backtest -> net SR / NW t / DSR.
# --------------------------------------------------------------------------- #

def run_arm(
    signal: pd.DataFrame, prices: pd.DataFrame, market_cap: pd.DataFrame,
    n_trials: int, label: str,
) -> dict:
    # m3: reindex market_cap onto the signal's grid at the call site so the VW
    # weighting and the alpha membership are aligned by construction (defensive —
    # quality_weights_vw also reindex_like's internally).
    res = fundamentals.quality_backtest(
        signal, prices, quantile=QUANTILE, cost_bps_per_side=COST_BPS_PER_SIDE,
        market_cap=market_cap.reindex(signal.index),
    )
    net = res["net"]
    sr = metrics.sharpe(net, periods=PERIODS_PER_YEAR)  # quarterly book -> sqrt(4)
    # NW lags: quarterly rebalance held one period -> lag 1 (period == horizon).
    t_nw = metrics.newey_west_tstat(net, lags=1)
    t_nw2 = metrics.newey_west_tstat(net, lags=2)  # robustness print (M4)
    dsr = metrics.deflated_sharpe_ratio(net, n_trials=n_trials)
    return {"label": label, "net": net, "gross": res["gross"],
            "sharpe": sr, "t_nw": t_nw, "t_nw2": t_nw2, "dsr": dsr,
            "turnover": res["annual_turnover"], "n_obs": int(net.shape[0])}


def _run_trial(source, n_trials: int) -> None:
    """The graded two-arm run — reachable only with a survivorship-safe source."""
    # The graded run consumes a richer interface than bare FundamentalsSource:
    # universe()/prices()/start/end. CompustatSource is a survivorship-safe SLOT
    # that has not implemented these (its field_series raises NotImplementedError
    # by design) — surface that as a clear "slot not wired" refusal, not a raw
    # AttributeError, and spend NO trial.
    for attr in ("universe", "prices", "prices_monthly", "start", "end"):
        if not hasattr(source, attr):
            sys.exit(
                f"\nSOURCE NOT WIRED: {type(source).__name__} passes the DATA GATE "
                f"(survivorship_safe) but does not implement '{attr}' — it is a slot. "
                "Connect WRDS/Compustat (filing-date-PIT fundamentals + delisting-"
                "inclusive prices: universe()/prices()/start/end + field_series) here. "
                "The two-arm harness above is proven; no trial spent (N unchanged).")
    # ---- universe: PIT members, EXCLUDING Financials + Real Estate ---------- #
    members = source.universe()
    current, _ = uni.fetch_sp500_tables()
    sectors = uni.sector_map(current, members)
    universe = [t for t in members if sectors.get(t, "UNKNOWN") not in EXCLUDED_SECTORS]
    n_excl = len(members) - len(universe)
    print(f"[universe] {len(members)} PIT members -> {len(universe)} after excluding "
          f"{EXCLUDED_SECTORS} ({n_excl} dropped).")

    asof = pd.bdate_range(source.start, source.end, freq=REBALANCE_FREQ)

    # ---- panels: CBOP/A (+ lagged) + market cap (price x shares), filing PIT - #
    panels = cbop_and_cap_panels(source, universe, asof)
    cbop_a = panels["cbop_a"]
    cbop_a_lagged = panels["cbop_a_lagged"]
    market_cap, prices = panels["market_cap"], panels["prices"]
    print(f"[panels] CBOP/A coverage {cbop_a.notna().any().sum()} names; "
          f"lagged-assets CBOP/A coverage {cbop_a_lagged.notna().any().sum()} names; "
          f"market-cap coverage {market_cap.notna().any().sum()} names.")

    # The genuine MONTHLY price grid (daily->month-end, NOT the quarterly grid):
    # drives the HML value loading and the 12-1 momentum baseline (B3 / M1).
    monthly_px = source.prices_monthly(universe)

    # ---- RAW arm: z(CBOP/A) -> VW quintile L/S ------------------------------ #
    raw_sig = fundamentals.cbop_signal(cbop_a)
    raw = run_arm(raw_sig, prices, market_cap, n_trials, "RAW")

    # ---- NEUTRAL arm: residualize vs trailing HML loading (+dollar) --------- #
    # M2: the HML-absent market+sector FALLBACK is unvalidated by the two-world
    # gate, so it must NOT yield a graded NEUTRAL verdict. If the FF HML file is
    # absent, we ABORT the NEUTRAL grade (a RAW-only report is acceptable). The FF
    # file IS present in this environment, so the live HML path is the default.
    loading, mode = hml_loading_panel(monthly_px, asof)
    if loading is None:
        sys.exit(
            "\nNEUTRAL GRADE ABORTED (M2): the monthly FF HML file is absent at "
            f"{HML_FILE!r}. The market+sector fallback neutralization is NOT "
            "validated by the two-world (quality_is_value / quality_orthogonal) "
            "gate, so it cannot produce a graded NEUTRAL verdict — and H1 "
            "graduates ONLY on the NEUTRAL arm. Provide the FF 5-factor monthly "
            "file (the registered live path) or accept a RAW-only report. No "
            "graded verdict, no trial spent (N unchanged).")

    # B3 HARD-GUARD: an entirely-NaN HML loading panel means the value loading
    # never estimated (e.g. too few monthly returns) -> value_neutralized_signal
    # would degenerate to a plain demean and NEUTRAL would falsely equal RAW
    # ("not value-collinear"). Refuse to grade rather than report a degenerate
    # NEUTRAL arm.
    if loading.notna().to_numpy().sum() == 0:
        sys.exit(
            "\nNEUTRAL GRADE ABORTED (B3): the assembled HML loading panel is "
            "ENTIRELY NaN — the trailing monthly HML betas never estimated (check "
            "the monthly price grid / FF window overlap). A NEUTRAL arm built on "
            "an all-NaN loading silently degenerates to a demean (NEUTRAL == RAW, "
            "a false 'not value-collinear' read), so no graded NEUTRAL verdict is "
            "issued. No trial spent (N unchanged).")
    print(f"[neutral] HML-loading neutralization (monthly FF, trailing 36m, "
          f"fit_intercept=True; mode={mode}); loading non-NaN cells "
          f"{int(loading.notna().to_numpy().sum())}.")
    neutral_sig = fundamentals.value_neutralized_cbop(cbop_a, loading)
    neutral = run_arm(neutral_sig, prices, market_cap, n_trials, "NEUTRAL")

    # ---- baselines ---------------------------------------------------------- #
    ew = equal_weight_baseline(prices, asof)
    mom = momentum_baseline(prices, asof, market_cap, monthly_px)
    sr_ew = metrics.sharpe(ew, periods=PERIODS_PER_YEAR)   # quarterly book -> sqrt(4)
    sr_mom = metrics.sharpe(mom, periods=PERIODS_PER_YEAR)  # quarterly book -> sqrt(4)

    # ---- PBO across the REGISTERED 4-config family (B5) --------------------- #
    # {raw-current, neutral-current, raw-lagged, neutral-lagged}. The lagged-assets
    # leg (the registered robustness denominator) MUST exist; PBO on a degenerate
    # 2-config matrix is forbidden. If the lagged leg cannot be assembled cleanly,
    # REFUSE to grade (PBO-on-incomplete-family is a hard FAIL that BLOCKS the
    # NEUTRAL graduation verdict — never grade on a 2-config PBO).
    lagged_ok = bool(cbop_a_lagged.notna().to_numpy().sum() > 0)
    print(f"[lagged-assets leg] status: "
          f"{'PRESENT' if lagged_ok else 'ABSENT'} (m2) — the Hou-Xue-Zhang 2020 "
          "insignificant robustness cut; required for the 4-config PBO family.")
    pbo_out = None
    pbo_blocked = False
    if not lagged_ok:
        pbo_blocked = True
        print("[pbo] *** BLOCKED *** the lagged-assets leg is empty, so only a "
              "degenerate 2-config {raw,neutral} matrix is available. Per B5, "
              "PBO-on-an-incomplete-family is a hard FAIL that BLOCKS the NEUTRAL "
              "graduation verdict — never grade on a 2-config PBO.")
    else:
        raw_lag_sig = fundamentals.cbop_signal(cbop_a_lagged)
        raw_lag = run_arm(raw_lag_sig, prices, market_cap, n_trials, "RAW-lagged")
        neutral_lag_sig = fundamentals.value_neutralized_cbop(cbop_a_lagged, loading)
        neutral_lag = run_arm(neutral_lag_sig, prices, market_cap, n_trials,
                              "NEUTRAL-lagged")
        mat = pd.DataFrame({
            "raw_current": raw["net"], "neutral_current": neutral["net"],
            "raw_lagged": raw_lag["net"], "neutral_lagged": neutral_lag["net"],
        })
        # Contiguous, gap-free common slice (NEVER dropna across mismatched
        # warm-ups -- that produces non-contiguous "contiguous" blocks).
        valid = mat.dropna(how="any")
        if not valid.empty:
            mat = mat.loc[valid.index.min(): valid.index.max()]
        n_common = int(mat.dropna(how="any").shape[0])
        if n_common >= 4:
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

    # ---- MDE: the N=12 net-annual-Sharpe hurdle for the realized n_obs ------ #
    n_obs = neutral["n_obs"]
    var_sr = 1.0 / max(n_obs, 1)
    sr_star_pp = metrics.expected_max_sharpe(n_trials, var_sr, n_obs)
    mde_ann = sr_star_pp * np.sqrt(PERIODS_PER_YEAR)  # quarterly book -> sqrt(4)

    # ---- report ------------------------------------------------------------- #
    print("\n=== H1 two-arm result (VALUE-WEIGHTED quintile L/S, CBOP/A) ===")
    for arm in (raw, neutral):
        t2 = arm.get("t_nw2")
        t2s = f"{t2:+.2f}" if (t2 is not None and not np.isnan(t2)) else "n/a"
        print(f"  {arm['label']:>7}: net SR {arm['sharpe']:+.3f}  t_NW {arm['t_nw']:+.2f}  "
              f"(lags=2 {t2s})  DSR {arm['dsr']:.3f}  turnover {arm['turnover']:.2f}/yr  "
              f"n_obs {arm['n_obs']}")
    print(f"  baselines: EW SR {sr_ew:+.3f} | 12-1 mom SR {sr_mom:+.3f} "
          "(12-MONTH minus 1-month, M1)")
    if pbo_out is not None:
        print(f"  PBO {pbo_out['pbo']:.3f}  (n_configs {pbo_out['n_configs']}, "
              f"n_obs {pbo_out['n_obs']}, splits {pbo_out['n_splits']})")
    else:
        print("  PBO: not graded (4-config family incomplete -> verdict BLOCKED).")
    print(f"  MDE @ N={n_trials}, n_obs={n_obs}: net annual SR hurdle ~{mde_ann:.3f} "
          f"(DSR>=0.95 deflation benchmark, quarterly sqrt(4) annualization).")
    print(f"  registered hurdle (2026-06-16 amendment): ~{REGISTERED_HURDLE_ANN:.2f} net "
          "annual SR (daily, ~15-yr) — printed beside the recomputed MDE (M5).")

    raw_minus_neutral = raw["sharpe"] - neutral["sharpe"]
    print(f"  raw - neutral SR gap: {raw_minus_neutral:+.3f} "
          "(large gap => edge was value-collinear; declared in advance).")

    # ---- pre-registered verdict (graduate iff ALL hold on the NEUTRAL arm) -- #
    # PBO must be computable on the COMPLETE 4-config family AND <= 0.5. If the
    # family was incomplete / uncomputable, the verdict is BLOCKED (B5).
    beats_baselines = neutral["sharpe"] > sr_ew and neutral["sharpe"] > sr_mom
    pbo_ok = (not pbo_blocked) and (pbo_out is not None) and (pbo_out["pbo"] <= 0.5)
    t_nw_ok = (neutral["t_nw"] is not None and not np.isnan(neutral["t_nw"])
               and neutral["t_nw"] >= 2.0)
    graduate = (
        t_nw_ok
        and beats_baselines
        and (neutral["dsr"] >= 0.95)
        and pbo_ok
    )
    print("\n=== PRE-REGISTERED VERDICT (NEUTRAL arm) ===")
    print(f"  t_NW >= 2 ............ {neutral['t_nw']:+.2f}  -> "
          f"{'PASS' if t_nw_ok else 'FAIL'}")
    print(f"  net SR > both basel .. {neutral['sharpe']:+.3f} vs EW {sr_ew:+.3f}, "
          f"mom {sr_mom:+.3f}  -> {'PASS' if beats_baselines else 'FAIL'}")
    print(f"  DSR >= 0.95 ......... {neutral['dsr']:.3f}  -> "
          f"{'PASS' if neutral['dsr'] >= 0.95 else 'FAIL'}")
    pbo_str = f"{pbo_out['pbo']:.3f}" if pbo_out is not None else "BLOCKED"
    print(f"  PBO <= 0.5 (4-config) {pbo_str}  -> {'PASS' if pbo_ok else 'FAIL'}")
    print(f"  >>> H1 {'GRADUATES' if graduate else 'does NOT graduate'} "
          "(log the row whatever it says — N becomes 12 here; do this by hand).")

    # Structured result for programmatic adjudication / regression tests. The
    # verdict is EXACTLY the 4-gate conjunction printed above (no hidden gate).
    return {
        "raw": raw, "neutral": neutral,
        "sr_ew": sr_ew, "sr_mom": sr_mom,
        "loading": loading, "loading_mode": mode,
        "pbo": pbo_out, "pbo_blocked": pbo_blocked,
        "gates": {
            "t_nw": t_nw_ok, "beats_baselines": beats_baselines,
            "dsr": bool(neutral["dsr"] >= 0.95), "pbo": pbo_ok,
        },
        "graduate": bool(graduate),
        "universe": universe, "excluded_sectors": EXCLUDED_SECTORS,
        "n_obs": n_obs, "mde_ann": mde_ann,
    }


def build_source(name: str, sources: dict, asof_end: str | None = None):
    """Construct the chosen FundamentalsSource, threading an optional FIXED as-of
    END date to sources that accept one. ``asof_end=None`` preserves the prior
    behavior EXACTLY (end -> today inside the source). Pinning the end makes a run
    reproducible (law #8) AND lets it reuse a date-keyed price cache; the pin is
    passed only to sources whose ctor accepts ``end`` (a no-op otherwise)."""
    cls = sources[name]
    if asof_end and "end" in inspect.signature(cls).parameters:
        return cls(end=asof_end)
    return cls()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypothesis", default="H1")
    ap.add_argument("--n-trials", type=int, default=12)
    ap.add_argument(
        "--source",
        choices=["free_sec", "compustat", "free_xwalk"],
        default="free_sec",
    )
    ap.add_argument(
        "--asof-end", default=None, metavar="YYYY-MM-DD",
        help="Pin the data-window END date for reproducibility and to reuse a "
             "date-keyed price cache. Default: today (UTC) — the prior behavior.",
    )
    args = ap.parse_args()

    # 1) Registration gate (law #3): H1 must be PROPOSED.
    try:
        require_runnable_registration(args.hypothesis)
    except RuntimeError as exc:
        sys.exit(f"REGISTRATION GATE: {exc}")
    print(f"[registration] {args.hypothesis} verified PROPOSED.")

    # 2) Machinery gate (law #4): synthetic planted must beat null, paired.
    print("[gate] synthetic quality world: planted must beat null (paired)...")
    gate = fundamentals.machinery_gate()
    for s, p, n in zip((7, 11, 23), gate["planted_sr"], gate["null_sr"]):
        print(f"  seed {s}: planted SR {p:+.2f} | null SR {n:+.2f}")
    if not gate["passed"]:
        sys.exit(f"MACHINERY GATE FAILED: differential {min(gate['diffs']):.2f} "
                 "<= 0.5 — harness cannot tell quality from its absence; abort.")
    print(f"[gate] PASS (min paired differential {min(gate['diffs']):.2f})")

    # 2b) NEUTRALIZATION GATE (registration amendment 2026-06-24, clause C) — the
    # B3 integrity check. H1 graduates ONLY on the NEUTRAL arm, so before trusting
    # any real raw-vs-neutral number we must prove IN-ENV that neutralization can
    # tell a value-DISGUISED edge (must collapse) from a value-ORTHOGONAL one (must
    # survive), SR-matched on the raw arm, with the collapse requiring the TRUE
    # value factor (placebo control). If it cannot today -> ABORT, no trial spent.
    print("[gate] neutralization two-world: value-disguised must COLLAPSE, "
          "value-orthogonal must SURVIVE (paired)...")
    ngate = fundamentals.neutralization_gate()
    for s, ni, no in zip((7, 11, 23), ngate["isvalue_neutral_sr"],
                         ngate["orthogonal_neutral_sr"]):
        print(f"  seed {s}: is-value neutral SR {ni:+.2f} (must < 0.30) | "
              f"orthogonal neutral SR {no:+.2f} (must > 1.00)")
    if not ngate["passed"]:
        sys.exit(
            "NEUTRALIZATION GATE FAILED (B3 / amendment clause C): "
            f"collapse_ok={ngate['collapse_ok']} survive_ok={ngate['survive_ok']} "
            f"sr_matched={ngate['sr_matched']} placebo_ok={ngate['placebo_ok']}. "
            "Neutralization cannot reliably tell a value-disguised edge from a "
            "genuine one today, so a NEUTRAL-arm verdict is untrustworthy — abort, "
            "no trial spent (N unchanged).")
    print("[gate] PASS (value-disguised collapses, orthogonal survives, "
          "SR-matched, placebo-controlled).")

    # 3) DATA GATE: a graded trial requires a survivorship-safe source.
    if args.source in ("free_sec", "free_xwalk"):
        # SurvivorshipSafeSECSource is the free survivorship-safe path; importing
        # it (and CompustatSource) is deferred to here so the script's import path
        # touches no network / API key.
        from quantlab.sec_xwalk_source import SurvivorshipSafeSECSource
        sources = {"free_sec": FreeSECSource, "free_xwalk": SurvivorshipSafeSECSource}
    else:
        sources = {"compustat": CompustatSource}
    if args.asof_end:
        print(f"[data] as-of END pinned to {args.asof_end} (reproducible window; "
              "stable price-cache key).")
    source = build_source(args.source, sources, args.asof_end)
    if not source.survivorship_safe:
        sys.exit(
            "\nDATA GATE: the free SEC source is SURVIVORSHIP-BLOCKED -- its "
            "ticker->CIK map is current-only (~73% coverage; dead/renamed names "
            "dropped, audit 2026-06-14). A graded H1 trial on it would re-commit "
            "trial #1's survivorship sin, so this run is REFUSED and spends no "
            "trial (N unchanged).\n  -> Use --source free_xwalk (free, "
            "survivorship-safe via the name->CIK crosswalk) or --source compustat "
            "(WRDS). The two-arm harness above is proven; this is one command "
            "from a clean trial #12 — which still requires explicit sign-off.")

    _run_trial(source, args.n_trials)


if __name__ == "__main__":
    main()
