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


def build_features(
    prices: pd.DataFrame,
    member_mask: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Return dict of feature name -> (date x ticker) z-scored frames.

    ``member_mask`` (date x ticker booleans): raw features are computed from
    each name's full price history (pre-membership prices are real, public
    data -- a new entrant's momentum is legitimate), but the cross-sectional
    z-score is taken over index members only, so non-members can't shift the
    mean/std the model normalizes against.
    """
    # fill_method=None: a halted or delisted name's missing price must yield a
    # NaN return, not a pad-filled phantom 0% -- phantom zeros deflate measured
    # vol and corrupt every downstream statistic for point-in-time universes.
    rets = prices.pct_change(fill_method=None)
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
    if member_mask is not None:
        feats = {name: f.where(member_mask) for name, f in feats.items()}
    return {name: _zscore_cs(f) for name, f in feats.items()}


def build_labels(
    prices: pd.DataFrame,
    horizon: int = 21,
    residualize: bool = False,
    member_mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Forward ``horizon``-day return, cross-sectionally z-scored.

    Label at date t uses prices (t, t+horizon] -- strictly future information,
    aligned so a model trained on rows <= t never sees beyond t + horizon
    (handled by the embargo in validation).

    ``residualize=True`` subtracts each name's beta-scaled market move over
    the same window: label = r_fwd - beta_t * mkt_fwd, with beta estimated
    from PAST returns only (rolling 252d as of t). The model then predicts
    idiosyncratic return -- the only part a dollar-neutral portfolio can
    actually harvest -- instead of wasting capacity on who-has-more-beta
    during whatever the market does next.
    """
    fwd = prices.shift(-horizon) / prices - 1.0
    if residualize:
        from quantlab.risk import rolling_beta

        rets = prices.pct_change(fill_method=None)
        mkt = rets.where(member_mask).mean(axis=1) if member_mask is not None else rets.mean(axis=1)
        beta = rolling_beta(rets, mkt)  # past-only, known at t
        mkt_prices = (1 + mkt.fillna(0)).cumprod()
        mkt_fwd = mkt_prices.shift(-horizon) / mkt_prices - 1.0
        fwd = fwd - beta.mul(mkt_fwd, axis=0)
    if member_mask is not None:
        fwd = fwd.where(member_mask)
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
