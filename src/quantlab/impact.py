"""Square-root market impact and capacity analysis.

Linear per-share costs answer "is there an edge at all?" -- this module
answers the question trading firms actually ask: **at what AUM does the edge
die?** A strategy that works on $1M and dies at $50M is a different (and much
less interesting) object than one that carries $1B.

Model: executing Q dollars in a name with daily dollar volume V and daily
volatility sigma moves the price by roughly

    impact ~= k * sigma * sqrt(Q / V)        (the "square-root law")

which is remarkably stable empirically across markets and decades (Almgren et
al. 2005; Toth et al. 2011). We charge each rebalance trade its half-spread
(the old linear cost) PLUS the square-root impact on the traded notional.
k defaults to 1.0, the literature's order-of-magnitude; capacity conclusions
should be read as order-of-magnitude too.

Assumptions stated plainly:
- Each rebalance's trade in a name executes within one day (participation =
  trade size / one day's volume). Slower execution lowers impact but adds
  tracking risk; modeling that is out of scope.
- ADV is a trailing 63-day median of dollar volume, shifted one day (point-
  in-time: today's trade uses yesterday's known liquidity).
- Impact is paid in full on each trade (no netting across names/days, no
  propagation/decay model).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def dollar_adv(
    prices: pd.DataFrame, volumes: pd.DataFrame, window: int = 63
) -> pd.DataFrame:
    """Trailing median daily dollar volume, shifted 1 day (known at trade time)."""
    dv = (prices * volumes.reindex_like(prices)).where(lambda x: x > 0)
    return dv.rolling(window, min_periods=window // 3).median().shift(1)


def impact_costs(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    adv: pd.DataFrame,
    aum: float,
    k: float = 1.0,
    spread_bps: float = 10.0,
    vol_window: int = 63,
) -> pd.Series:
    """Daily cost drag (as a return) of trading ``weights`` at ``aum`` dollars.

    cost_t = sum_i |dw_it| * [spread + k * sigma_it * sqrt(|dw_it| * AUM / ADV_it)]

    where |dw| is the day's weight change. Names with unknown ADV fall back to
    the cross-sectional median ADV that day (conservative-ish; flagged in the
    capacity report as adv_coverage).
    """
    # Same convention as backtest.run_backtest: weights chosen at t take
    # effect (and are traded, and pay costs) at t+1, the day they start
    # earning returns.
    daily_w = weights.reindex(prices.index).ffill().shift(1).fillna(0.0)
    dw = daily_w.diff().abs()

    rets = prices.pct_change(fill_method=None)
    sigma = rets.rolling(vol_window, min_periods=vol_window // 3).std().shift(1)

    adv_aligned = adv.reindex(prices.index).reindex(columns=daily_w.columns)
    row_median = adv_aligned.median(axis=1)
    adv_filled = adv_aligned.apply(lambda col: col.fillna(row_median))

    participation = (dw * aum).div(adv_filled)
    impact = k * sigma * np.sqrt(participation)
    per_name = dw * (spread_bps / 1e4 + impact.fillna(0.0))
    return per_name.sum(axis=1)


def capacity_curve(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    adv: pd.DataFrame,
    gross_returns: pd.Series,
    aums: tuple[float, ...] = (1e6, 1e7, 5e7, 1e8, 5e8, 1e9),
    k: float = 1.0,
    spread_bps: float = 10.0,
) -> pd.DataFrame:
    """Net Sharpe / return / cost drag as a function of AUM.

    ``gross_returns``: the strategy's cost-free daily returns (impact replaces
    the linear cost model entirely here -- spread is inside impact_costs).
    """
    rows = []
    start = gross_returns.index[0]
    for aum in aums:
        costs = impact_costs(
            weights, prices, adv, aum=aum, k=k, spread_bps=spread_bps
        ).loc[start:]
        net = gross_returns - costs
        ann_cost = float(costs.mean() * TRADING_DAYS)
        sr = float(net.mean() / net.std() * np.sqrt(TRADING_DAYS)) if net.std() > 0 else 0.0
        rows.append(
            {
                "aum": aum,
                "sharpe_net": sr,
                "ann_return_net": float(net.mean() * TRADING_DAYS),
                "ann_cost_drag": ann_cost,
            }
        )
    out = pd.DataFrame(rows).set_index("aum")
    out.attrs["adv_coverage"] = float(
        adv.reindex(prices.index).notna().mean().mean()
    )
    return out


def capacity_at_zero(curve: pd.DataFrame) -> float | None:
    """Smallest AUM in the sweep where net Sharpe <= 0 (None if it never dies).

    Linear interpolation between sweep points would imply more precision than
    a k~1 order-of-magnitude impact model can support, so we report the grid
    point instead.
    """
    dead = curve[curve["sharpe_net"] <= 0]
    return float(dead.index.min()) if len(dead) else None
