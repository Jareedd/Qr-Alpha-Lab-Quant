"""Factor-neutral risk model for the execution/risk engine.

Composes with sizing.py: sizing decides HOW MUCH to lever a unit-vol book; this
module decides the book's SHAPE — neutralize ex-ante exposure to chosen factors
(market, sectors, size…) and estimate the book's risk so the vol target in
sizing.py is honest rather than asserted. Source-agnostic: operates on aligned
returns / weight / loadings frames, pinned by known-answer tests before any real
book.

Ex-ante and past-only by construction: betas use trailing windows; the
projection uses loadings known at t; nothing peeks forward. This is the
engine-level, reusable form of the neutralization the equity backtest does
inline — built so a graduated strategy plugs straight in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def rolling_market_beta(
    asset_returns: pd.DataFrame,
    market_returns: pd.Series,
    lookback: int = 252,
    min_periods: int = 126,
) -> pd.DataFrame:
    """Past-only rolling beta of each asset to the market: cov(asset, mkt) /
    var(mkt) over the trailing window. (date x asset); NaN until min_periods."""
    m = market_returns.reindex(asset_returns.index)
    cov = asset_returns.rolling(lookback, min_periods=min_periods).cov(m)
    var = m.rolling(lookback, min_periods=min_periods).var()
    return cov.div(var, axis=0)


def neutralize_weights(weights: pd.Series, loadings: pd.DataFrame) -> pd.Series:
    """Project ``weights`` to ZERO net exposure across every column of
    ``loadings`` (assets x factors), with the smallest change to the book:
    ``w_n = w - L (LᵀL)⁻¹ Lᵀ w``  ⇒  ``Lᵀ w_n = 0``. Uses the pseudo-inverse so
    collinear/duplicate factors are handled. Include a column of ones in
    ``loadings`` to also force dollar-neutrality."""
    aligned = weights.reindex(loadings.index).fillna(0.0)
    w = aligned.to_numpy(dtype=float)
    L = loadings.fillna(0.0).to_numpy(dtype=float)
    coef = np.linalg.pinv(L.T @ L) @ (L.T @ w)
    return pd.Series(w - L @ coef, index=loadings.index)


def sample_covariance(returns: pd.DataFrame, shrinkage: float = 0.0) -> pd.DataFrame:
    """Per-period sample covariance, optionally shrunk toward its diagonal
    (Ledoit–Wolf-style, fixed intensity): ``(1-δ)·S + δ·diag(S)``. ``δ=0`` is the
    raw sample cov; ``δ=1`` zeros every off-diagonal. Shrinkage keeps the matrix
    well-conditioned when assets ≫ history — the regime a real book lives in."""
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError(f"shrinkage must be in [0,1], got {shrinkage}")
    s = returns.cov()
    d = pd.DataFrame(np.diag(np.diag(s.to_numpy())), index=s.index, columns=s.columns)
    return (1.0 - shrinkage) * s + shrinkage * d


def predicted_vol(
    weights: pd.Series, cov: pd.DataFrame, periods: int = TRADING_DAYS
) -> float:
    """Ex-ante portfolio volatility ``sqrt(wᵀ Σ w)``, annualized by ``periods``.
    ``cov`` is a per-period covariance (e.g. from ``sample_covariance``)."""
    w = weights.reindex(cov.index).fillna(0.0).to_numpy(dtype=float)
    var = float(w @ cov.to_numpy(dtype=float) @ w)
    return float(np.sqrt(max(var, 0.0)) * np.sqrt(periods))


def net_factor_exposure(weights: pd.Series, loadings: pd.DataFrame) -> pd.Series:
    """``Lᵀ w`` — the book's net exposure to each factor (zero after
    ``neutralize_weights``). The number you report every rebalance instead of
    asserting neutrality."""
    w = weights.reindex(loadings.index).fillna(0.0).to_numpy(dtype=float)
    return pd.Series(loadings.fillna(0.0).to_numpy(dtype=float).T @ w,
                     index=loadings.columns)
