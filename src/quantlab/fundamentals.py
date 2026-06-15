"""H1 quality-tilt harness — features, PIT assembly, signal, and machinery gate.

Features (Novy-Marx 2013 profitability; Sloan 1996 accruals):
- GP/A         = gross profit / assets        (high = good)
- accruals/A   = (net income − CFO) / assets   (high = bad; accrual reversal)

All filing-date point-in-time: feature series are indexed by ``filed`` and
as-of-aligned to rebalance dates (latest filing ≤ date). The quality signal is
cross-sectional ``z(GP/A) − z(accruals/A)``; the book longs the high-quality
quintile, shorts the low. Slow rebalance keeps cost mortality low — the failure
mode that killed the price-feature trials.

The synthetic machinery gate (``machinery_gate``) must pass before any real run:
``planted_quality`` recovered, ``null_quality`` rejected, paired per seed. Mirrors
the carry/CEF harnesses so H1 is the same machine pointed at fundamentals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab import metrics
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


def _gross_profit(source: FundamentalsSource, ticker: str) -> pd.Series:
    """GP = GrossProfit if tagged, else Revenue − CoGS (the audit's finding:
    direct GrossProfit is ~0% tagged; the subtraction caps ~59% on non-financials)."""
    gp = source.field_series(ticker, "gross_profit")
    if not gp.empty:
        return gp
    rev, cogs = (source.field_series(ticker, f) for f in ("revenue", "cogs"))
    if rev.empty or cogs.empty:
        return pd.Series(dtype=float)
    idx = rev.index.union(cogs.index)
    return (rev.reindex(idx).ffill() - cogs.reindex(idx).ffill()).dropna()


def pit_feature_panels(
    source: FundamentalsSource, tickers: list[str], asof_dates: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    """Assemble (asof_date x ticker) GP/A and accruals/A panels, PIT: each cell
    is the latest filing on or before that date. Unmapped/blank tickers drop out
    (NaN), surfaced honestly rather than imputed."""
    gp_a, acc_a = {}, {}
    for t in tickers:
        assets = source.field_series(t, "assets")
        if assets.empty:
            continue
        gp = _gross_profit(source, t)
        if not gp.empty:
            gp_a[t] = gp_over_assets(gp, assets).reindex(asof_dates, method="ffill")
        ni, cfo = source.field_series(t, "net_income"), source.field_series(t, "cfo")
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


def quality_backtest(signal: pd.DataFrame, prices: pd.DataFrame,
                     quantile: float = 0.2, cost_bps_per_side: float = 10.0) -> dict:
    """Period book: weights at t earn the t→t+1 return. Returns net/gross series
    and annual turnover."""
    fwd = prices.pct_change(fill_method=None).shift(-1).reindex_like(signal)
    w = quality_weights(signal, quantile=quantile)
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
