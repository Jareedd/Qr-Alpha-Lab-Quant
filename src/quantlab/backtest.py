"""Cost-aware long-short backtester.

Monthly-rebalanced decile long-short portfolio built from out-of-sample
predictions. Transaction costs are charged on turnover at ``cost_bps`` per
side. No shorting rebates, no leverage, no lookahead: weights formed from
predictions at rebalance date t are applied to returns from t+1 onward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def predictions_to_weights(
    preds: pd.Series,
    quantile: float = 0.1,
    rebalance_every: int = 21,
) -> pd.DataFrame:
    """Long top quantile, short bottom quantile, equal weight, dollar neutral.

    Vectorized across rebalance dates: ``rank(method="first")`` selects the
    same names as ``nlargest``/``nsmallest`` (both break ties by first
    occurrence). Dates with fewer than 10 valid predictions get zero weights.
    """
    wide = preds.unstack("ticker")
    sub = wide.iloc[::rebalance_every]
    n_valid = sub.count(axis=1)
    k = (n_valid * quantile).astype(int).clip(lower=1)

    rank_desc = sub.rank(axis=1, ascending=False, method="first")
    rank_asc = sub.rank(axis=1, ascending=True, method="first")
    longs = rank_desc.le(k, axis=0)
    shorts = rank_asc.le(k, axis=0)
    per_name = (0.5 / k).where(n_valid >= 10, 0.0)
    weights = longs.mul(per_name, axis=0) - shorts.mul(per_name, axis=0)
    return weights.fillna(0.0)


def run_backtest(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    cost_bps: float = 10.0,
) -> dict:
    """Apply weights to daily returns with linear transaction costs.

    Returns dict with daily net/gross return series and turnover.
    """
    # fill_method=None: dead names produce NaN returns (excluded from the
    # daily P&L sum), never pad-filled phantom zeros. The true delisting
    # return is still missing -- that residual bias is documented, not hidden.
    rets = prices.pct_change(fill_method=None).reindex(columns=weights.columns)
    # Daily weights: hold each rebalance's weights until the next one;
    # shift(1) so weights chosen at t earn returns from t+1 (no lookahead).
    daily_w = weights.reindex(rets.index).ffill().shift(1).fillna(0.0)

    gross = (daily_w * rets).sum(axis=1)

    w_change = daily_w.diff().abs().sum(axis=1)
    costs = w_change * (cost_bps / 1e4)
    net = gross - costs

    # Annualized one-way turnover.
    years = max(len(rets) / 252, 1e-9)
    ann_turnover = float(w_change.sum() / 2 / years)

    start = weights.index[0]
    return {
        "gross": gross.loc[start:],
        "net": net.loc[start:],
        "annual_turnover": ann_turnover,
    }
