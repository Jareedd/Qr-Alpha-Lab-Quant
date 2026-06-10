"""Cross-sectional feature engineering.

All features are computed from past data only and z-scored cross-sectionally
per date (so the model learns relative, not absolute, predictions).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _zscore_cs(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each row (date) across assets."""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-12, axis=0)


def build_features(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return dict of feature name -> (date x ticker) z-scored frames."""
    rets = prices.pct_change()
    log_p = np.log(prices)

    feats = {
        # 12-1 momentum: trailing year excluding most recent month (JT 1993).
        "mom_12_1": log_p.shift(21) - log_p.shift(252),
        # 6-month momentum, skip a month.
        "mom_6_1": log_p.shift(21) - log_p.shift(126),
        # Short-term reversal: last month's return, sign flips in literature.
        "rev_1m": log_p - log_p.shift(21),
        # Realized volatility, 3 months (low-vol anomaly proxy).
        "vol_3m": rets.rolling(63).std(),
        # Distance from 52-week high (George & Hwang 2004).
        "pct_52w_high": prices / prices.rolling(252).max() - 1.0,
    }
    return {name: _zscore_cs(f) for name, f in feats.items()}


def build_labels(prices: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """Forward ``horizon``-day return, cross-sectionally z-scored.

    Label at date t uses prices (t, t+horizon] -- strictly future information,
    aligned so a model trained on rows <= t never sees beyond t + horizon
    (handled by the embargo in validation).
    """
    fwd = prices.shift(-horizon) / prices - 1.0
    return _zscore_cs(fwd)


def stack_panel(
    features: dict[str, pd.DataFrame], labels: pd.DataFrame
) -> pd.DataFrame:
    """Long-format panel: index (date, ticker), columns = features + 'label'."""
    parts = {name: f.stack() for name, f in features.items()}
    parts["label"] = labels.stack()
    panel = pd.DataFrame(parts).dropna()
    panel.index.names = ["date", "ticker"]
    return panel
