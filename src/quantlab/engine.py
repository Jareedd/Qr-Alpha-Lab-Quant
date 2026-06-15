"""Top-level execution/risk engine: signals → book → orders.

Composes the tested engine pieces into the pipeline a multi-manager book runs:

    combine signals → neutralize factor exposure → SIZE (vol-target × confidence
    from the uncertainty-shrunk Sharpe) → LIMITS (position / gross / drawdown) →
    EXECUTION (integer-share orders, cost estimate).

The honest property, demonstrated in tests: feed it a real edge and it sizes up
as evidence accumulates; feed it noise and it sizes to ~ZERO, because the
confidence haircut is driven by the LOWER confidence bound of the trailing
Sharpe (same logic as sizing.py). It is deliberately slow to commit capital —
the counterweight to the H6 over-confidence. It does NOT fetch data or submit
orders; orders feed the (frozen) live Alpaca client.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from quantlab import combine, execution, limits, risk_model

TRADING_DAYS = 252


class PortfolioEngine:
    def __init__(
        self, target_vol: float = 0.10, conf: float = 0.95,
        confidence_full_sharpe: float = 0.5, max_weight: float = 0.05,
        max_gross: float = 2.0, dd_max: float = 0.15, dd_floor: float = 0.25,
        lookback: int = 252, periods: int = TRADING_DAYS,
    ):
        self.target_vol = target_vol
        self.max_weight = max_weight
        self.max_gross = max_gross
        self.dd_max = dd_max
        self.dd_floor = dd_floor
        self.lookback = lookback
        self.periods = periods
        self.sr_full = confidence_full_sharpe
        self._z = float(stats.norm.ppf(conf))

    # --- pipeline steps ----------------------------------------------------- #
    def _raw_weights(self, composite: pd.DataFrame) -> pd.DataFrame:
        """Dollar-neutral, gross-1 proportional book from the composite signal."""
        demeaned = composite.sub(composite.mean(axis=1), axis=0)
        gross = demeaned.abs().sum(axis=1).replace(0.0, np.nan)
        return demeaned.div(gross, axis=0).fillna(0.0)

    def _neutralize(self, weights: pd.DataFrame, loadings: pd.DataFrame | None):
        if loadings is None:
            return weights
        L = loadings.reindex(weights.columns)
        rows = {d: risk_model.neutralize_weights(weights.loc[d], L) for d in weights.index}
        return pd.DataFrame(rows).T.reindex(columns=weights.columns).fillna(0.0)

    def _leverage(self, book_ret: pd.Series) -> pd.Series:
        """Past-only leverage = vol-target scale × confidence ∈ [0,1]. Confidence
        ramps 0→1 as the annualized LOWER-bound Sharpe goes 0→sr_full; a book
        whose edge isn't confidently positive gets ~zero leverage."""
        win, n = self.lookback, self.lookback
        mean = book_ret.rolling(win, min_periods=win // 2).mean()
        std = book_ret.rolling(win, min_periods=win // 2).std()
        rv_ann = std * np.sqrt(self.periods)
        vol_scale = (self.target_vol / rv_ann).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        sr_pp = (mean / std).replace([np.inf, -np.inf], 0.0).fillna(0.0)   # per-period
        se = np.sqrt((1.0 + 0.5 * sr_pp**2) / n)                # Lo-2002 SE (cf. sizing)
        sr_lb_ann = (sr_pp - self._z * se) * np.sqrt(self.periods)
        conf = (sr_lb_ann / self.sr_full).clip(lower=0.0, upper=1.0)
        return (vol_scale * conf).shift(1).fillna(0.0)

    # --- public API --------------------------------------------------------- #
    def build(self, signals: dict[str, pd.DataFrame], prices: pd.DataFrame,
              loadings: pd.DataFrame | None = None) -> pd.DataFrame:
        """Signals → final target-weight panel (after sizing, limits, drawdown)."""
        composite = combine.combine_signals(signals)
        raw = self._neutralize(self._raw_weights(composite), loadings)
        capped = limits.cap_position(raw, self.max_weight)
        rets = prices.pct_change(fill_method=None).reindex(columns=capped.columns)
        book_ret = (capped.shift(1).fillna(0.0) * rets).sum(axis=1, min_count=1).fillna(0.0)
        sized = limits.cap_gross(capped.mul(self._leverage(book_ret), axis=0), self.max_gross)
        sized_ret = (sized.shift(1).fillna(0.0) * rets).sum(axis=1, min_count=1).fillna(0.0)
        return sized.mul(limits.drawdown_scale(sized_ret, self.dd_max, self.dd_floor), axis=0)

    def backtest(self, signals, prices, loadings=None, cost_bps: float = 10.0) -> dict:
        """Run the engine end-to-end and return its net/gross P&L and turnover."""
        w = self.build(signals, prices, loadings=loadings)
        rets = prices.pct_change(fill_method=None).reindex(columns=w.columns)
        gross = (w.shift(1).fillna(0.0) * rets).sum(axis=1, min_count=1)
        turnover = w.diff().abs().sum(axis=1).fillna(0.0)
        net = (gross - turnover * cost_bps / 1e4).dropna()
        return {"weights": w, "net": net, "gross": gross.dropna(),
                "avg_gross_exposure": float(w.abs().sum(axis=1).mean()),
                "ann_turnover": float(turnover.sum() / max(len(w), 1) * self.periods)}

    def latest_orders(self, signals, prices, equity: float,
                      loadings=None, current_shares=None) -> dict:
        """The execution plan for the most recent date — integer-share orders
        ready for the live client."""
        w = self.build(signals, prices, loadings=loadings)
        prev = w.iloc[-2] if len(w) > 1 else None
        return execution.execution_plan(
            w.iloc[-1], equity, prices.reindex(columns=w.columns).iloc[-1],
            current_shares=current_shares, prev_weights=prev, max_weight=self.max_weight)
