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


def rolling_factor_betas(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    lookback: int = 252,
    min_periods: int = 126,
    fit_intercept: bool = True,
) -> dict[str, pd.DataFrame]:
    """Past-only rolling multivariate OLS of each asset on the K factor columns.

    Per trailing window ending at t (inclusive), solve beta = pinv(XᵀX)(XᵀY) with
    X = [1 | factors] when fit_intercept (the DEFAULT, REQUIRED for byte-identical
    K=1 reduction to rolling_market_beta on nonzero-mean factors). pinv (not inv)
    for collinearity-robustness, matching neutralize_weights.

    fit_intercept=True is load-bearing and must NOT be "simplified" to a
    through-origin solve: rolling_market_beta computes cov/var (ddof=1), which is
    the demeaned / with-intercept OLS slope. A through-origin solve differs by
    ~5e-3 on nonzero-mean returns and only coincidentally matches on noise-free
    test assets. (Verified: with-intercept matches rolling_market_beta to
    2.4e-15; through-origin diverges by 4.9e-3.)

    Returns dict keyed by factor_returns.columns -> (date x asset) DataFrame of
    that factor's loading (intercept NOT returned), indexed by
    asset_returns.index, columns asset_returns.columns, NaN until min_periods
    valid factor rows are present in the window.

    Trailing/causal: beta_t uses returns through t only (point-in-time safe).
    factor_returns is reindexed onto asset_returns.index first (asset index
    authoritative). factor_returns must be a DataFrame (one column for K=1)."""
    if lookback <= 0:
        raise ValueError(f"lookback must be > 0, got {lookback}")
    if not (0 < min_periods <= lookback):
        raise ValueError(
            f"need 0 < min_periods <= lookback, got min_periods={min_periods}, "
            f"lookback={lookback}"
        )
    if asset_returns.empty or factor_returns.empty:
        raise ValueError("asset_returns and factor_returns must be non-empty")

    F = factor_returns.reindex(asset_returns.index)
    names = list(F.columns)
    dates = asset_returns.index
    n = len(dates)
    A = asset_returns.shape[1]
    Y = asset_returns.to_numpy(dtype=float)
    Xf = F.to_numpy(dtype=float)
    out = {name: np.full((n, A), np.nan) for name in names}

    for t in range(n):
        lo = max(0, t - lookback + 1)
        Xw = Xf[lo : t + 1]
        Yw = Y[lo : t + 1]
        mask = ~np.isnan(Xw).any(axis=1)
        count = int(mask.sum())
        if count < min_periods:
            continue
        Xm = Xw[mask]
        Ym = Yw[mask]
        Xd = np.column_stack([np.ones(count), Xm]) if fit_intercept else Xm
        if np.isnan(Ym).any():
            coef = np.full((Xd.shape[1], A), np.nan)
            for a in range(A):
                ym = Ym[:, a]
                ok = ~np.isnan(ym)
                if int(ok.sum()) < min_periods:
                    continue
                Xa = Xd[ok]
                coef[:, a] = np.linalg.pinv(Xa.T @ Xa) @ (Xa.T @ ym[ok])
        else:
            coef = np.linalg.pinv(Xd.T @ Xd) @ (Xd.T @ Ym)
        betas_t = coef[1:, :] if fit_intercept else coef
        for k, name in enumerate(names):
            out[name][t, :] = betas_t[k, :]

    return {
        name: pd.DataFrame(out[name], index=dates, columns=asset_returns.columns)
        for name in names
    }


def loadings_at(
    factor_betas: dict[str, pd.DataFrame],
    date,
    assets: pd.Index | None = None,
    add_dollar: bool = False,
) -> pd.DataFrame:
    """Assemble the per-date L matrix (assets x factors) that neutralize_weights /
    net_factor_exposure consume, by slicing each factor frame as-of ``date``
    (last row at or before date) and column-stacking. Optionally append a
    'dollar' ones column for dollar-neutrality. Index = ``assets`` (default:
    union of the factor frames' columns); columns = list(factor_betas)
    [+ 'dollar']. A NaN/missing loading is LEFT AS NaN: the downstream
    neutralize_weights .fillna(0.0) then leaves that name un-neutralized that
    date (the conservative, documented convention — a coverage hole does not
    crash and does not masquerade as a neutral book)."""
    if assets is None:
        idx = None
        for f in factor_betas.values():
            idx = f.columns if idx is None else idx.union(f.columns)
        assets = idx
    cols = {}
    for name, f in factor_betas.items():
        sub = f.loc[:date]
        row = sub.iloc[-1] if len(sub) else pd.Series(np.nan, index=f.columns)
        cols[name] = row.reindex(assets)
    L = pd.DataFrame(cols, index=assets)
    if add_dollar:
        L["dollar"] = 1.0
    return L


def cross_sectional_neutralize(
    signal: pd.Series, loadings: pd.DataFrame
) -> pd.Series:
    """OLS-residualize a cross-sectional SIGNAL against factor loadings:
    r = s - L pinv(LᵀL)(Lᵀs). Same projection algebra as neutralize_weights but
    applied to an alpha signal, not weights. Include a ones column in
    ``loadings`` to also demean (dollar-neutral). signal is reindexed onto
    loadings.index and .fillna(0.0); loadings .fillna(0.0). The shared helper
    keeps the signal- and weight-neutralizers from drifting apart."""
    aligned = signal.reindex(loadings.index).fillna(0.0)
    s = aligned.to_numpy(dtype=float)
    L = loadings.fillna(0.0).to_numpy(dtype=float)
    coef = np.linalg.pinv(L.T @ L) @ (L.T @ s)
    return pd.Series(s - L @ coef, index=loadings.index)
