"""Position sizing for the execution/risk engine — honest geometric growth.

The "maximize geometric growth" instinct is Kelly sizing: lever in proportion to
the edge. The catch this whole project exists to respect is that you never KNOW
the edge — you estimate it, and full Kelly on an *estimated* edge over-levers
exactly when the estimate is luck (the DSR is the math of how often it is). So
the engine sizes on the LOWER confidence bound of the Sharpe, not the point
estimate, and applies a fractional-Kelly haircut on top.

The honest consequence — and the entire point — is that a near-zero or
statistically-uncertain edge (everything qr-alpha-lab has found so far) sizes to
**near zero**. Nothing here is alpha; this is the machine that turns a *verified*
edge into growth without blowing up on an unverified one. The day a strategy
graduates (clears its pre-registered bar at the true N), it plugs in here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def kelly_fraction(mu: float, sigma: float) -> float:
    """Full-Kelly leverage for a single edge: ``f* = mu / sigma**2`` (mu, sigma
    per-period). Returns 0 for nonpositive or degenerate inputs — the engine
    never *shorts* the Kelly bet on a negative estimated edge; it stands aside."""
    if sigma <= 0 or not np.isfinite(mu):
        return 0.0
    return max(0.0, float(mu) / float(sigma) ** 2)


def sharpe_standard_error(sr: float, n_obs: int) -> float:
    """Asymptotic SE of an estimated per-period Sharpe (Lo 2002):
    ``sqrt((1 + 0.5*SR**2) / n_obs)``. More data and a smaller Sharpe both
    tighten it; this is what makes a short, noisy track record size small."""
    if n_obs <= 1:
        return float("inf")
    return float(np.sqrt((1.0 + 0.5 * sr**2) / n_obs))


def kelly_under_uncertainty(
    sharpe_hat: float, n_obs: int, fraction: float = 0.5, conf: float = 0.95
) -> float:
    """Growth-optimal leverage sized on the LOWER confidence bound of the edge.

    Takes an estimated per-period Sharpe and the number of observations behind
    it, and returns a leverage multiplier for a book already scaled to unit
    volatility (see ``vol_target_scale``). Mechanics: compute the one-sided
    lower-confidence-bound Sharpe ``SR_lb = SR_hat - z(conf)*SE``; if it is not
    even confidently positive, size **zero**; otherwise lever ``fraction * SR_lb``
    (fractional Kelly, default half).

    Properties the tests pin: a zero/negative estimated edge -> 0 (the project's
    core lesson, encoded); size increases with n_obs (uncertainty shrinks) and
    with the fractional-Kelly knob; it never exceeds full Kelly on the point
    estimate.
    """
    if n_obs <= 1 or not np.isfinite(sharpe_hat):
        return 0.0
    se = sharpe_standard_error(sharpe_hat, n_obs)
    sr_lb = sharpe_hat - stats.norm.ppf(conf) * se
    return float(fraction * max(0.0, sr_lb))


def realized_vol(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    return float(r.std() * np.sqrt(periods))


def vol_target_scale(
    returns: pd.Series, target_vol: float, periods: int = TRADING_DAYS
) -> float:
    """Leverage that scales a book's realized vol to ``target_vol`` (annual).
    Zero if the book has no measurable vol. Doubling the target doubles the
    scale; a book running 2x the target gets scaled to 0.5x."""
    rv = realized_vol(returns, periods)
    return 0.0 if rv <= 0 else float(target_vol) / rv


def size_book(
    weights: pd.Series,
    book_returns: pd.Series,
    sharpe_hat: float,
    n_obs: int,
    target_vol: float = 0.10,
    fraction: float = 0.5,
    conf: float = 0.95,
    periods: int = TRADING_DAYS,
) -> pd.Series:
    """Compose the two steps: vol-target the raw dollar-neutral ``weights`` to
    ``target_vol``, then apply the uncertainty-shrunk Kelly leverage. A book with
    no confident edge collapses to ~zero gross exposure — by design.

    Returns the final position vector (same index as ``weights``). Pure: no
    market data is fetched, nothing is traded; this hands a target to the
    execution layer.
    """
    vt = vol_target_scale(book_returns, target_vol, periods)
    lev = kelly_under_uncertainty(sharpe_hat, n_obs, fraction=fraction, conf=conf)
    return weights * vt * lev
