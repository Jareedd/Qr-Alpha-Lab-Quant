"""Cost-aware execution planning for the engine: target weights → integer-share
orders with a per-name cap, plus turnover/impact cost estimates.

PLANNING ONLY. This produces an order list and a cost estimate; it does NOT
submit anything — order submission stays in the (frozen) live Alpaca path
(live.py / scripts/live_trade.py), which consumes exactly this kind of
integer-share delta. Keeping planning pure means the whole engine is testable
offline and the live trading path is never touched by engine development.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab import limits


def target_shares(weights: pd.Series, equity: float, prices: pd.Series,
                  max_weight: float | None = None) -> pd.Series:
    """Integer share targets for ``weights`` at ``equity`` dollars. Optional
    per-name weight cap applied first. Names with no price → 0."""
    w = weights if max_weight is None else limits.cap_position(weights, max_weight)
    dollars = w * float(equity)
    shares = dollars / prices.reindex(w.index).replace(0.0, np.nan)
    return shares.fillna(0.0).round().astype(int)


def orders_from_targets(current_shares: pd.Series, target_shares: pd.Series) -> pd.Series:
    """Delta orders (buy > 0 / sell < 0), nonzero only. Aligns on the target
    index, treating absent current holdings as flat."""
    cur = current_shares.reindex(target_shares.index).fillna(0)
    delta = (target_shares - cur).astype(int)
    return delta[delta != 0]


def turnover_cost(prev_weights: pd.Series, target_weights: pd.Series,
                  cost_bps: float = 10.0) -> float:
    """Linear cost of rebalancing prev → target: sum |Δw| × cost_bps."""
    tt = float(target_weights.sub(prev_weights, fill_value=0.0).abs().sum())
    return tt * cost_bps / 1e4


def execution_plan(
    weights: pd.Series, equity: float, prices: pd.Series,
    current_shares: pd.Series | None = None,
    prev_weights: pd.Series | None = None,
    max_weight: float = 0.05, cost_bps: float = 10.0,
) -> dict:
    """One rebalance's plan: capped integer-share targets, the delta orders to
    get there, gross/net dollar exposure, and the estimated rebalance cost. The
    ``orders`` series is what the live client would submit."""
    tgt = target_shares(weights, equity, prices, max_weight=max_weight)
    cur = current_shares if current_shares is not None else pd.Series(0, index=tgt.index)
    orders = orders_from_targets(cur, tgt)
    px = prices.reindex(tgt.index).fillna(0.0)
    capped_w = limits.cap_position(weights, max_weight)
    est_cost = (turnover_cost(prev_weights, capped_w, cost_bps)
                if prev_weights is not None else float("nan"))
    return {
        "target_shares": tgt,
        "orders": orders,
        "gross_notional": float((tgt.abs() * px).sum()),
        "net_notional": float((tgt * px).sum()),
        "n_orders": int(len(orders)),
        "est_rebalance_cost_frac": est_cost,
    }
