"""Performance metrics, including the Deflated Sharpe Ratio.

The DSR (Bailey & Lopez de Prado, 2014) answers: "given that I tried N
strategy variants, what is the probability that this Sharpe ratio is real
rather than the lucky maximum of N draws of noise?" Any research process that
does not track N is, statistically, lying to itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329


def newey_west_tstat(series: pd.Series, lags: int) -> float:
    """t-statistic of the series mean with HAC (Newey-West 1987) standard error.

    Why it exists here: daily rank ICs of h-day forward-return labels overlap,
    so consecutive ICs share h-1 days of label information and are strongly
    autocorrelated. A naive t-stat (mean/sem) assumes independence and can
    overstate significance by roughly sqrt(h). Use ``lags`` >= label horizon.
    Bartlett-kernel weights guarantee a positive variance estimate.
    """
    x = series.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < lags + 2:
        return np.nan
    xc = x - x.mean()
    s = float(xc @ xc) / n
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        s += 2.0 * w * float(xc[:-lag] @ xc[lag:]) / n
    se = np.sqrt(s / n)
    return float(x.mean() / se) if se > 0 else np.nan


def sharpe(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods))


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns.fillna(0)).cumprod()
    return float((equity / equity.cummax() - 1).min())


def probabilistic_sharpe_ratio(
    returns: pd.Series, sr_benchmark: float = 0.0
) -> float:
    """P(true SR > sr_benchmark), adjusting for skew, kurtosis, sample size."""
    r = returns.dropna()
    n = len(r)
    if n < 30:
        return np.nan
    sr = r.mean() / r.std()  # per-period SR
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    z = (sr - sr_benchmark) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_sr: float, n_obs: int) -> float:
    """Expected maximum per-period SR among n_trials of pure noise."""
    if n_trials <= 1:
        return 0.0
    sd = np.sqrt(var_sr if var_sr > 0 else 1.0 / n_obs)
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(sd * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_ratio(
    returns: pd.Series, n_trials: int, var_sr: float | None = None
) -> float:
    """PSR against the expected-max-of-N-noise-trials benchmark.

    > 0.95: likely genuine. < 0.95: cannot reject that it's the luckiest of N.
    """
    r = returns.dropna()
    if var_sr is None:
        var_sr = 1.0 / len(r)
    sr_star = expected_max_sharpe(n_trials, var_sr, len(r))
    return probabilistic_sharpe_ratio(r, sr_benchmark=sr_star)


def summary(
    net: pd.Series,
    gross: pd.Series,
    annual_turnover: float,
    n_trials: int,
) -> dict:
    return {
        "ann_return_net": float(net.mean() * TRADING_DAYS),
        "ann_vol": float(net.std() * np.sqrt(TRADING_DAYS)),
        "sharpe_gross": sharpe(gross),
        "sharpe_net": sharpe(net),
        "max_drawdown": max_drawdown(net),
        "annual_turnover": annual_turnover,
        "skew": float(stats.skew(net.dropna())),
        "psr": probabilistic_sharpe_ratio(net),
        "dsr": deflated_sharpe_ratio(net, n_trials=n_trials),
        "n_trials_assumed": n_trials,
        "n_days": int(net.dropna().shape[0]),
    }
