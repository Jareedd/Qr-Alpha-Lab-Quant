"""Benchmark strategies every model must beat (research law #5: baselines first).

Two baselines, evaluated over the SAME out-of-sample dates as the model and
run through the same cost-aware backtester:

- ``momentum_baseline_weights``: a decile long-short ranked by the raw 12-1
  momentum feature alone (Jegadeesh & Titman 1993). Zero machine learning --
  this is the one-line strategy the ML pipeline has to justify itself against.
- ``equal_weight_returns``: long-only 1/N over the universe -- what the market
  hands you for free, with essentially no turnover.

If the model cannot beat the momentum rank net of costs out-of-sample, that
is a reportable finding, not a failure to hide.
"""

from __future__ import annotations

import pandas as pd

from quantlab.backtest import predictions_to_weights


def momentum_baseline_weights(
    feats: dict[str, pd.DataFrame],
    pred_index: pd.MultiIndex,
    quantile: float = 0.1,
    rebalance_every: int = 21,
) -> pd.DataFrame:
    """Decile long-short weights from the mom_12_1 feature alone.

    Restricted to ``pred_index`` (the model's out-of-sample (date, ticker)
    pairs) so model and baseline are compared on identical dates and names --
    point-in-time safe because mom_12_1 uses only past prices.
    """
    mom = feats["mom_12_1"].stack()
    mom.index.names = ["date", "ticker"]
    mom = mom.reindex(pred_index).dropna()
    if mom.empty:
        raise ValueError("no overlap between momentum feature and prediction index")
    return predictions_to_weights(mom, quantile=quantile, rebalance_every=rebalance_every)


def equal_weight_returns(prices: pd.DataFrame, start=None) -> pd.Series:
    """Daily returns of a long-only 1/N portfolio (rebalanced daily).

    Daily 1/N rebalancing has negligible turnover cost at this universe size,
    so gross ~= net; reported as-is for context, not as a tradable claim.
    """
    ew = prices.pct_change().mean(axis=1)
    return ew.loc[start:] if start is not None else ew
