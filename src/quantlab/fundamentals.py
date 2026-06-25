"""H1 quality-tilt harness — features, PIT assembly, signal, and machinery gate.

Features (Novy-Marx 2013 profitability; Sloan 1996 accruals):
- GP/A         = gross profit / assets        (high = good)
- accruals/A   = (net income − CFO) / assets   (high = bad; accrual reversal)

All filing-date point-in-time: feature series are indexed by ``filed`` and
as-of-aligned to rebalance dates (latest filing ≤ date). On the free SEC path,
FLOW numerators use annual-only filings before division by point-in-time assets,
which avoids the confirmed bug of mixing 10-Q quarterly flows with stock values.
The quality signal is cross-sectional ``z(GP/A) − z(accruals/A)``; the book
longs the high-quality quintile, shorts the low. Slow rebalance keeps cost
mortality low — the failure mode that killed the price-feature trials.

The synthetic machinery gate (``machinery_gate``) must pass before any real run:
``planted_quality`` recovered, ``null_quality`` rejected, paired per seed. Mirrors
the carry/CEF harnesses so H1 is the same machine pointed at fundamentals.

The NEUTRAL arm has its OWN gate (``neutralization_gate``, registration amendment
2026-06-24 clause C): a value-DISGUISED edge must COLLAPSE when HML-neutralized and
a value-ORTHOGONAL edge must SURVIVE (SR-matched + placebo-controlled). H1
graduates only on the NEUTRAL arm, so this must pass in-env before any graded run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab import metrics, risk_model
from quantlab.fundamentals_data import FundamentalsSource

PERIODS_PER_YEAR = 12


def gp_over_assets(gross_profit: pd.Series, assets: pd.Series) -> pd.Series:
    """Filing-date GP/A. Inputs are ``filed``-indexed; aligned on the union of
    filing dates and forward-filled (the latest known value applies until the
    next filing — point-in-time safe)."""
    idx = gross_profit.index.union(assets.index)
    gp = gross_profit.reindex(idx).ffill()
    a = assets.reindex(idx).ffill()
    return (gp / a.replace(0.0, np.nan)).dropna()


def accruals_over_assets(net_income: pd.Series, cfo: pd.Series,
                         assets: pd.Series) -> pd.Series:
    """Filing-date total-accruals/A = (NI − CFO)/A (Sloan). Higher = more
    accrual-heavy earnings = worse forward returns."""
    idx = net_income.index.union(cfo.index).union(assets.index)
    ni, c, a = (s.reindex(idx).ffill() for s in (net_income, cfo, assets))
    return ((ni - c) / a.replace(0.0, np.nan)).dropna()


def cbop_over_assets(
    gross_profit: pd.Series,
    net_income: pd.Series,
    cfo: pd.Series,
    assets: pd.Series,
) -> pd.Series:
    """Filing-date cash-based operating profitability over assets (CBOP/A).

    The H1 profitability leg per the 2026-06-16 registration amendment: gross
    profit NET of total accruals, divided by point-in-time assets. Accruals
    (NI − CFO) are SUBTRACTED so profitability already booked as accruals (which
    reverse) does not count — this is profitability net of accruals, so the
    separate accruals leg is subsumed (Ball, Gerakos, Linnainmaa & Nikolaev 2016)
    and must NOT be blended on top (double-counting).

        CBOP/A = (GP − (NI − CFO)) / Assets

    DOCUMENTED PROXY: this is the AVAILABLE-FIELDS CBOP. Ball et al. (2016)'s full
    construction starts from gross profit and adds back the working-capital and
    operating-accrual deltas explicitly (ΔAR, ΔInventory, ΔAP, ΔDeferred revenue,
    minus SG&A timing), none of which are in FIELD_TAGS. (NI − CFO) is the
    total-accruals proxy for that adjustment using only tagged lines; it captures
    the same net-of-accruals intent but is coarser than the textbook measure. This
    limitation is stated, not hidden (CLAUDE.md style).

    Point-in-time: all inputs are ``filed``-indexed; the union index is forward-
    filled (latest known filing applies until the next — no lookahead). Flow
    numerators (GP, NI, CFO) must already be annualized on the same basis (the
    annual-only / TTM convention the runner enforces) BEFORE this divides by
    stock assets — the 2026-06-16 annualization fix."""
    idx = (gross_profit.index.union(net_income.index)
           .union(cfo.index).union(assets.index))
    gp, ni, c, a = (
        s.reindex(idx).ffill() for s in (gross_profit, net_income, cfo, assets)
    )
    cbop = gp - (ni - c)
    return (cbop / a.replace(0.0, np.nan)).dropna()


def cbop_signal(cbop_a: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional quality score for the H1 RAW arm: per-date z-score of
    CBOP/A (high = profitable net of accruals = good). Mirrors ``quality_signal``
    but on the single CBOP leg the registration freezes (no GP/A − accruals
    blend). Demeaned and unit-scaled per row, so it composes directly with the
    value-neutralization and VW-quintile construction downstream."""
    return _zscore_rows(cbop_a)


def _gross_profit(source: FundamentalsSource, ticker: str) -> pd.Series:
    """GP = GrossProfit if tagged, else Revenue − CoGS (the audit's finding:
    direct GrossProfit is ~0% tagged; the subtraction caps ~59% on non-financials)."""
    gp = source.field_series(ticker, "gross_profit", annual_only=True)
    if not gp.empty:
        return gp
    rev, cogs = (
        source.field_series(ticker, f, annual_only=True) for f in ("revenue", "cogs")
    )
    if rev.empty or cogs.empty:
        return pd.Series(dtype=float)
    idx = rev.index.union(cogs.index)
    return (rev.reindex(idx).ffill() - cogs.reindex(idx).ffill()).dropna()


def pit_feature_panels(
    source: FundamentalsSource, tickers: list[str], asof_dates: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    """Assemble (asof_date x ticker) GP/A and accruals/A panels, PIT: each cell
    is the latest filing on or before that date. Flow numerators are annual-only
    on the free SEC path, so GP/A and accruals/A are not understated by quarter
    values divided by stock assets. Unmapped/blank tickers drop out (NaN),
    surfaced honestly rather than imputed."""
    gp_a, acc_a = {}, {}
    for t in tickers:
        assets = source.field_series(t, "assets")
        if assets.empty:
            continue
        gp = _gross_profit(source, t)
        if not gp.empty:
            gp_a[t] = gp_over_assets(gp, assets).reindex(asof_dates, method="ffill")
        ni = source.field_series(t, "net_income", annual_only=True)
        cfo = source.field_series(t, "cfo", annual_only=True)
        if not ni.empty and not cfo.empty:
            acc_a[t] = accruals_over_assets(ni, cfo, assets).reindex(asof_dates, method="ffill")
    return {"gp_a": pd.DataFrame(gp_a, index=asof_dates),
            "accruals_a": pd.DataFrame(acc_a, index=asof_dates)}


def _zscore_rows(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.sub(panel.mean(axis=1), axis=0).div(panel.std(axis=1) + 1e-12, axis=0)


def quality_signal(gp_a: pd.DataFrame, accruals_a: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cross-sectional quality score per date: ``z(GP/A) − z(accruals/A)`` (high
    profitability good, high accruals bad). If accruals are absent, profitability
    alone (the cleanly-coverable, sector-agnostic reduced signal)."""
    sig = _zscore_rows(gp_a)
    if accruals_a is not None and not accruals_a.empty:
        sig = sig.sub(_zscore_rows(accruals_a.reindex_like(gp_a)), fill_value=0.0)
    return sig


def quality_weights(signal: pd.DataFrame, quantile: float = 0.2) -> pd.DataFrame:
    """Dollar-neutral equal-weight quintiles: LONG highest quality, SHORT lowest,
    per rebalance date (full reset each period — slow rebalance, so turnover is
    low and net ≈ gross)."""
    target = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    for d in signal.index:
        row = signal.loc[d].dropna()
        n = int(len(row) * quantile)
        if n < 2:
            continue
        longs, shorts = row.nlargest(n).index, row.nsmallest(n).index
        target.loc[d, longs] = 0.5 / n
        target.loc[d, shorts] = -0.5 / n
    return target


def quality_weights_vw(
    signal: pd.DataFrame,
    market_cap: pd.DataFrame,
    quantile: float = 0.2,
) -> pd.DataFrame:
    """Dollar-neutral, VALUE-WEIGHTED quintile long-short — the H1-frozen book
    (2026-06-16 amendment: value-weighted QUINTILE long-short, not equal-weight,
    not decile).

    Per rebalance date: LONG the top-``quantile`` of ``signal``, SHORT the
    bottom-``quantile``. WITHIN each side, weight by ``market_cap`` (= price ×
    shares, passed in) normalized so the long side sums to +0.5 and the short side
    to −0.5 — i.e. dollar-neutral, with bigger-cap names in a side carrying
    proportionally more of that side's notional. This is the institutional book a
    large-cap quality claim must actually trade; the equal-weight ``quality_weights``
    is kept unchanged for the synthetic machinery gate and the EW comparison.

    Conventions (documented):
    - Membership uses ``signal`` (the alpha); the weighting uses ``market_cap``.
      A name in a quintile with missing/non-positive cap is dropped from that
      side BEFORE normalization (it cannot be value-weighted), so each side still
      sums to exactly ±0.5 over its weightable names.
    - If a side has no weightable name (all caps missing) that date is skipped —
      surfaced as a flat row, not silently mis-normalized.
    - With EQUAL caps within a side this reduces to the EW book (each name gets
      0.5/n), so the VW and EW constructions agree in that limiting case — pinned
      in the tests."""
    target = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    mc = market_cap.reindex_like(signal)
    for d in signal.index:
        row = signal.loc[d].dropna()
        n = int(len(row) * quantile)
        if n < 2:
            continue
        longs, shorts = row.nlargest(n).index, row.nsmallest(n).index
        caps_d = mc.loc[d]
        for side, names in ((+0.5, longs), (-0.5, shorts)):
            w = caps_d.reindex(names)
            w = w.where(w > 0.0).dropna()            # only positively-cap'd names
            tot = float(w.sum())
            if w.empty or tot <= 0.0:
                continue                              # no weightable name this side
            target.loc[d, w.index] = side * (w / tot).to_numpy()
    return target


def quality_backtest(signal: pd.DataFrame, prices: pd.DataFrame,
                     quantile: float = 0.2, cost_bps_per_side: float = 10.0,
                     market_cap: pd.DataFrame | None = None) -> dict:
    """Period book: weights at t earn the t→t+1 return. Returns net/gross series
    and annual turnover.

    Construction: EQUAL-weight quintiles by default (``market_cap=None`` — the
    synthetic-gate / EW-baseline book). Pass ``market_cap`` (a (date x ticker)
    panel = price × shares) to switch to the registered VALUE-WEIGHTED quintiles
    (``quality_weights_vw``) — the book a large-cap H1 claim actually trades."""
    fwd = prices.pct_change(fill_method=None).shift(-1).reindex_like(signal)
    w = (quality_weights(signal, quantile=quantile) if market_cap is None
         else quality_weights_vw(signal, market_cap, quantile=quantile))
    gross = (w * fwd).sum(axis=1, min_count=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    net = (gross - turnover * cost_bps_per_side / 1e4).dropna()
    return {"net": net, "gross": gross.dropna(),
            "annual_turnover": float(turnover.sum() / max(len(w), 1) * PERIODS_PER_YEAR)}


def machinery_gate(seeds=(7, 11, 23), n_firms: int = 200, n_periods: int = 180,
                   min_differential: float = 0.5) -> dict:
    """Falsification gate (law #4): planted_quality must beat null_quality,
    paired per seed, before any real H1 run. Imports synthetic lazily so the
    data layer has no synthetic dependency."""
    from quantlab.synthetic import make_quality_panel
    diffs, planted, null = [], [], []
    for s in seeds:
        p = make_quality_panel(n_firms, n_periods, mode="planted_quality", seed=s)
        n = make_quality_panel(n_firms, n_periods, mode="null_quality", seed=s)
        sr_p = metrics.sharpe(quality_backtest(quality_signal(p.attrs["gp_a"]), p,
                                               cost_bps_per_side=0.0)["net"],
                              periods=PERIODS_PER_YEAR)
        sr_n = metrics.sharpe(quality_backtest(quality_signal(n.attrs["gp_a"]), n,
                                               cost_bps_per_side=0.0)["net"],
                              periods=PERIODS_PER_YEAR)
        diffs.append(sr_p - sr_n); planted.append(sr_p); null.append(sr_n)
    return {"passed": min(diffs) > min_differential, "diffs": diffs,
            "planted_sr": planted, "null_sr": null}


def _static_loading_panel(price: pd.DataFrame) -> pd.DataFrame:
    """(period x firm) panel broadcasting the STATIC ground-truth value loading
    of a quality_is_value / quality_orthogonal synthetic world. Estimation-error-
    free: this is the loading the registration's two-world gate uses ('static-
    loading neutral SR'), so a gate failure indicts the NEUTRALIZATION ALGEBRA,
    not the rolling-beta estimator."""
    vl = price.attrs["value_loading"]
    return pd.DataFrame(
        np.tile(vl.to_numpy(), (len(price.index), 1)),
        index=price.index, columns=price.columns,
    )


def neutralization_gate(
    seeds=(7, 11, 23),
    collapse_max: float = 0.3,
    survive_min: float = 1.0,
    sr_match_tol: float = 1.0,
    placebo_factor: float = 0.5,
) -> dict:
    """Falsification gate for the NEUTRAL arm (registration amendment 2026-06-24,
    clause C) — the B3 integrity check. Run in-env IMMEDIATELY BEFORE any real H1
    run; if neutralization cannot tell a value-disguised edge from a genuine one
    TODAY, no real raw-vs-neutral number is trusted -> ABORT, no trial spent.

    machinery_gate proves the RAW arm can tell quality from its absence; this
    proves the NEUTRAL arm can tell a value-DISGUISED edge (which must COLLAPSE
    when neutralized) from a value-ORTHOGONAL edge (which must SURVIVE). Four
    paired per-seed checks, all on the STATIC ground-truth loading:

      1. ``quality_is_value`` -> static-loading neutral SR < ``collapse_max`` (0.3):
         the 'alpha' was the value tilt, so HML-neutralizing kills it.
      2. ``quality_orthogonal`` -> static-loading neutral SR > ``survive_min`` (1.0):
         a genuinely value-orthogonal edge survives neutralization.
      3. SR-MATCHED on the RAW arm (MEAN |raw_isvalue - raw_orthogonal| across
         seeds <= ``sr_match_tol``): the discrimination is attributable to
         neutralization, not a Sharpe-level gap between the two worlds — the same
         "raw SR alone cannot separate the worlds" property pinned (across 33 seeds)
         by ``test_raw_sharpe_alone_does_not_separate_worlds``. The MEAN is used (a
         single-seed raw-SR gap is noisy on a 3-seed sample); a world that were
         genuinely raw-SR-separable would show a large mean gap and fail here.
      4. PLACEBO control: in ``quality_is_value``, neutralizing against the TRUE
         value loading must collapse MORE than against a random placebo factor
         (neutral_true < ``placebo_factor`` x neutral_placebo) — the collapse
         REQUIRES the true value factor, it is not generic shrinkage.

    Returns a dict with ``passed`` (all four hold, every seed) and the per-seed
    diagnostics. Synthetic only (law #7); spends ZERO trials."""
    from quantlab.synthetic import make_quality_panel

    def _sr(net):
        return metrics.sharpe(net, periods=PERIODS_PER_YEAR)

    def _raw_neutral(price):
        gp_a = price.attrs["gp_a"]
        raw = _sr(quality_backtest(quality_signal(gp_a), price,
                                   cost_bps_per_side=0.0)["net"])
        neutral = _sr(quality_backtest(
            value_neutralized_signal(gp_a, _static_loading_panel(price)),
            price, cost_bps_per_side=0.0)["net"])
        return raw, neutral

    isvalue_neutral, orth_neutral, sr_gaps, placebo_ok = [], [], [], []
    for s in seeds:
        pa = make_quality_panel(mode="quality_is_value", seed=s)
        pb = make_quality_panel(mode="quality_orthogonal", seed=s)
        raw_a, neu_a = _raw_neutral(pa)
        raw_b, neu_b = _raw_neutral(pb)
        isvalue_neutral.append(neu_a)
        orth_neutral.append(neu_b)
        sr_gaps.append(abs(raw_a - raw_b))

        # placebo: neutralize the value-disguised world against a RANDOM factor.
        gp_a = pa.attrs["gp_a"]
        placebo = np.random.default_rng(10 * s + 1).standard_normal(pa.shape[1])
        vl_pl = pd.DataFrame(np.tile(placebo, (len(pa.index), 1)),
                             index=pa.index, columns=pa.columns)
        neu_pl = _sr(quality_backtest(
            value_neutralized_signal(gp_a, vl_pl), pa, cost_bps_per_side=0.0)["net"])
        placebo_ok.append(neu_a < placebo_factor * neu_pl)

    collapse_ok = max(isvalue_neutral) < collapse_max
    survive_ok = min(orth_neutral) > survive_min
    sr_matched = float(np.mean(sr_gaps)) <= sr_match_tol
    placebo_pass = all(placebo_ok)
    return {
        "passed": bool(collapse_ok and survive_ok and sr_matched and placebo_pass),
        "collapse_ok": bool(collapse_ok),
        "survive_ok": bool(survive_ok),
        "sr_matched": bool(sr_matched),
        "placebo_ok": bool(placebo_pass),
        "isvalue_neutral_sr": isvalue_neutral,
        "orthogonal_neutral_sr": orth_neutral,
        "raw_sr_gaps": sr_gaps,
    }


def value_neutralized_signal(
    signal_panel: pd.DataFrame,
    value_loading: pd.DataFrame,
    accruals_a: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """``quality_signal(signal_panel, accruals_a)`` then, per date, cross-sectionally
    residualize against the value loading (HML proxy) + a ones column (dollar-
    neutral demean), via ``risk_model.cross_sectional_neutralize``. The 'neutral'
    arm of the H1 raw-vs-neutral test.

    ``signal_panel`` is the raw quality panel to neutralize (renamed from ``gp_a``,
    m1): the H1 registration passes the single CBOP/A leg here, NOT a GP/A panel —
    so the name no longer implies a profitability source or an accruals blend.
    ``accruals_a`` stays an OPTIONAL legacy hook (default None) used only by the
    older GP/A − accruals synthetic gate; for CBOP (accruals already subsumed) it
    is never passed — see ``value_neutralized_cbop`` which asserts that.

    Point-in-time: ``value_loading`` must be known at t — a synthetic ground-truth
    attr in the lab, or a trailing past-only ``rolling_factor_betas`` HML loading
    on real data. A date whose value_loading row is all-NaN degenerates to a plain
    demean (no value-neutralization that date) rather than crashing — the
    conservative, documented convention. A date absent from ``value_loading.index``
    is left as the raw signal."""
    sig = quality_signal(signal_panel, accruals_a)
    out = sig.copy()
    for d in sig.index:
        if d not in value_loading.index:
            continue  # leave as-is (no loading known)
        L = pd.DataFrame(
            {"value": value_loading.loc[d].reindex(sig.columns), "dollar": 1.0},
            index=sig.columns,
        )
        out.loc[d] = risk_model.cross_sectional_neutralize(sig.loc[d], L).reindex(
            sig.columns
        )
    return out


def value_neutralized_cbop(
    cbop_a: pd.DataFrame, value_loading: pd.DataFrame
) -> pd.DataFrame:
    """H1 NEUTRAL arm, explicit (m1): residualize the CBOP/A quality signal against
    the value loading (+ dollar) with NO accruals leg — CBOP already subsumes
    accruals (Ball, Gerakos, Linnainmaa & Nikolaev 2016), so blending one on top
    double-counts. A thin, self-documenting wrapper over ``value_neutralized_signal``
    that hard-asserts the no-accruals invariant the registration freezes."""
    return value_neutralized_signal(cbop_a, value_loading, accruals_a=None)
