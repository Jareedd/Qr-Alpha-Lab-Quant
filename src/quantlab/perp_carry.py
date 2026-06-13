"""H2 carry backtest harness (trial #8) — the registered strategy, exactly.

Consumes the panels from ``perp_data.build_panels`` and implements the
H2 registration verbatim:

- Universe at t: top-30 by trailing 30d dollar volume among contracts
  TRADING at t (PIT; delisted contracts simply stop being eligible when
  their data ends — no hand list, no survivorship).
- Signal: trailing 7d mean daily funding (monotone-equivalent to the
  21-settlement mean; ranking is all that matters). Uses funding through
  t only.
- Label/return: funding-INCLUSIVE total return, ``mark_return - funding``
  (the funding flows ARE the premium; a price-only label measures the
  wrong object — pinned in tests/test_synthetic_carry.py).
- Book: dollar-neutral, equal-weight quartiles — SHORT the top funding
  quartile (they pay), LONG the bottom (they receive), weekly rebalance.
- Costs: taker 5 bps + spread 2 bps = 7 bps per side on turnover (linear
  headline, matching the equity pipeline's convention; sqrt impact is the
  capacity dimension, reported separately if requested).

Everything here is point-in-time by construction: weights formed from
data through t earn from t+1. The registered paired control
(``shuffle_funding``) and the funding-income/price-drag decomposition
live here too, so the single registered run emits every number the
registration demands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def total_returns(price: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """Funding-inclusive daily total return for a LONG: mark return minus
    the day's funding (positive funding = longs pay)."""
    return price.pct_change(fill_method=None) - funding.reindex_like(price).fillna(0.0)


def pit_universe(
    dollar_volume: pd.DataFrame,
    top_n: int = 30,
    lookback: int = 30,
    min_names: int = 10,
) -> pd.DataFrame:
    """Boolean (date x symbol): the top-``top_n`` by trailing dollar volume
    among symbols trading at t. Past-only (rolling mean of ADV through t).
    Dates with fewer than ``min_names`` trading names get no universe (too
    thin to rank). ``top_n`` larger than the trading count selects all of
    them — which is how the synthetic gate reuses this exact code path."""
    adv = dollar_volume.rolling(lookback, min_periods=max(5, lookback // 2)).mean()
    mask = pd.DataFrame(False, index=adv.index, columns=adv.columns)
    arr = adv.to_numpy()
    for i in range(len(adv.index)):
        row = pd.Series(arr[i], index=adv.columns).dropna()
        if len(row) < min_names:
            continue
        keep = row.nlargest(min(top_n, len(row))).index
        mask.iloc[i, mask.columns.get_indexer(keep)] = True
    return mask


def carry_signal(funding: pd.DataFrame, lookback: int = 7) -> pd.DataFrame:
    """Trailing mean daily funding through t (the rank signal)."""
    return funding.rolling(lookback, min_periods=lookback).mean()


def carry_weights(
    signal: pd.DataFrame,
    universe: pd.DataFrame,
    quantile: float = 0.25,
    rebalance: int = 7,
) -> pd.DataFrame:
    """Dollar-neutral equal-weight quartile book, rebalanced every
    ``rebalance`` days and held between rebalances. Short high funding,
    long low funding, within the PIT universe only."""
    sig = signal.where(universe)
    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    for i in range(0, len(signal.index), rebalance):
        row = sig.iloc[i].dropna()
        n_side = int(len(row) * quantile)
        if n_side < 2:
            continue
        shorts = row.nlargest(n_side).index
        longs = row.nsmallest(n_side).index
        weights.iloc[i, weights.columns.get_indexer(shorts)] = -0.5 / n_side
        weights.iloc[i, weights.columns.get_indexer(longs)] = 0.5 / n_side
    held = weights.replace(0.0, np.nan).ffill(limit=rebalance - 1).fillna(0.0)
    # a name that delists mid-hold drops out (its return goes NaN) -- zero it
    return held


def carry_backtest(
    panels: dict[str, pd.DataFrame],
    cost_bps_per_side: float = 7.0,
    quantile: float = 0.25,
    rebalance: int = 7,
    sig_lookback: int = 7,
    top_n: int = 30,
) -> dict:
    """Run the registered carry book. Returns daily gross/net total
    returns, turnover, and the funding-income vs price-drag decomposition.
    Weights at t earn from t+1 (PIT)."""
    price, vol, funding = panels["price"], panels["dollar_volume"], panels["funding"]
    universe = pit_universe(vol, top_n=top_n)
    signal = carry_signal(funding, lookback=sig_lookback)
    weights = carry_weights(signal, universe, quantile=quantile, rebalance=rebalance)

    tot = total_returns(price, funding)
    price_ret = price.pct_change(fill_method=None)
    held = weights.shift(1)

    gross = (held * tot).sum(axis=1, min_count=1)
    # decomposition: where did the P&L come from?
    price_pnl = (held * price_ret).sum(axis=1, min_count=1)
    funding_pnl = (held * (-funding.reindex_like(price).fillna(0.0))).sum(
        axis=1, min_count=1
    )
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (cost_bps_per_side / 1e4)
    net = (gross - cost).dropna()

    ann_turnover = float(turnover.sum() / len(weights) * 252)
    return {
        "net": net,
        "gross": gross.dropna(),
        "price_pnl": price_pnl.dropna(),
        "funding_pnl": funding_pnl.dropna(),
        "weights": weights,
        "universe": universe,
        "signal": signal,
        "annual_turnover": ann_turnover,
    }


def forward_total_return(
    price: pd.DataFrame, funding: pd.DataFrame, horizon: int = 7
) -> pd.DataFrame:
    """Forward ``horizon``-day funding-inclusive total return per symbol
    (the IC label). Sum of daily total returns over (t, t+h]."""
    tot = total_returns(price, funding)
    return tot.shift(-1).rolling(horizon).sum().shift(-(horizon - 1))


def shuffle_funding(funding: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Registered paired control: permute funding ACROSS symbols within
    each date (destroys the cross-sectional carry structure, preserves
    marginal distributions and the price panel). A book built on this must
    earn ~nothing; if it earns, the harvest was structure, not carry."""
    rng = np.random.default_rng(seed)
    out = funding.copy()
    arr = out.to_numpy()
    for i in range(arr.shape[0]):
        row = arr[i]
        finite = np.where(np.isfinite(row))[0]
        if len(finite) > 1:
            perm = rng.permutation(finite)
            arr[i, finite] = row[perm]
    return pd.DataFrame(arr, index=out.index, columns=out.columns)
