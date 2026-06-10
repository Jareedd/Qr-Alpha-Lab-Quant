"""Risk neutralization and exposure reporting.

Why this module exists: a naive cross-sectional long-short is rarely the
"market-neutral stock-picking" it claims to be. Momentum loads on beta and
sectors (long winners = long whatever sector ran); low-vol loads short beta.
Factor exposure usually explains most of a naive signal's return, so Phase 3
asks: after removing it, is anything left?

Two neutralizations, both deliberately simple and point-in-time safe:

- Sector: demean predictions within (date, sector) before ranking, so longs
  and shorts spread across sectors instead of betting one sector vs another.
- Beta: at each rebalance, project weights onto the subspace satisfying
  {sum(w) = 0, w . beta = 0}, using rolling betas estimated from PAST returns
  only, then rescale to the original gross exposure.

Plus a risk report quantifying what the neutralization did (realized market
beta, sector tilts, market correlation) -- claims of neutrality are tested,
not asserted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_beta(
    asset_rets: pd.DataFrame,
    mkt_rets: pd.Series,
    window: int = 252,
    min_periods: int = 126,
) -> pd.DataFrame:
    """Per-asset rolling OLS beta vs the market, (date x ticker).

    The beta at date t uses returns through t only -- weights formed at t and
    applied from t+1 may use it without lookahead.
    """
    cov = asset_rets.rolling(window, min_periods=min_periods).cov(mkt_rets)
    var = mkt_rets.rolling(window, min_periods=min_periods).var()
    return cov.div(var, axis=0)


def neutralize_predictions_by_sector(
    preds: pd.Series, sectors: dict[str, str]
) -> pd.Series:
    """Demean predictions within (date, sector): scores become sector-relative.

    A decile portfolio built from sector-relative scores picks the best names
    *within* each sector rather than loading up on whichever sector's signal
    is hot, killing most net sector exposure at the source.
    """
    sec = preds.index.get_level_values("ticker").map(lambda t: sectors.get(t, "UNKNOWN"))
    grouped = preds.groupby([preds.index.get_level_values("date"), sec])
    return preds - grouped.transform("mean")


def beta_neutralize_weights(
    weights: pd.DataFrame, betas: pd.DataFrame
) -> pd.DataFrame:
    """Project each rebalance row onto {sum(w)=0, w.beta=0}, keep gross fixed.

    Gram-Schmidt on the two constraint directions (ones and beta). Missing
    betas (young listings inside the estimation window) fall back to the
    cross-sectional mean beta -- neutral, since projection removes the mean
    direction anyway. Rows where betas are degenerate (all equal) only get
    the dollar-neutrality projection.
    """
    out = weights.copy()
    for d, w in weights.iterrows():
        active = w[w != 0.0]
        if len(active) < 3:
            continue
        bt = betas.loc[:d]
        if bt.empty:
            continue
        b = bt.iloc[-1].reindex(active.index)
        b = b.fillna(b.mean())
        if b.isna().all():
            continue

        v = active.to_numpy(dtype=float)
        ones = np.ones(len(v)) / np.sqrt(len(v))
        v = v - (v @ ones) * ones  # enforce sum(w)=0 (a no-op if already)
        b_orth = b.to_numpy(dtype=float)
        b_orth = b_orth - (b_orth @ ones) * ones
        norm = np.linalg.norm(b_orth)
        if norm > 1e-10:
            b_orth /= norm
            v = v - (v @ b_orth) * b_orth

        gross = np.abs(v).sum()
        if gross > 1e-12:
            v *= active.abs().sum() / gross
        out.loc[d, :] = 0.0
        out.loc[d, active.index] = v
    return out


def risk_report(
    strategy_rets: pd.Series,
    mkt_rets: pd.Series,
    daily_weights: pd.DataFrame,
    betas: pd.DataFrame,
    sectors: dict[str, str],
    window: int = 63,
) -> dict:
    """Measure (not assert) the portfolio's realized factor exposure."""
    df = pd.DataFrame({"strat": strategy_rets, "mkt": mkt_rets}).dropna()

    roll_cov = df["strat"].rolling(window).cov(df["mkt"])
    roll_var = df["mkt"].rolling(window).var()
    roll_beta = (roll_cov / roll_var).dropna()

    # Ex-ante portfolio beta at each date: w . beta with what was known then.
    aligned_b = betas.reindex(daily_weights.index).reindex(
        columns=daily_weights.columns
    )
    port_beta = (daily_weights * aligned_b).sum(axis=1)
    port_beta = port_beta[daily_weights.abs().sum(axis=1) > 0]

    sec_series = pd.Series(
        {t: sectors.get(t, "UNKNOWN") for t in daily_weights.columns}
    )
    sector_net = daily_weights.T.groupby(sec_series).sum().T
    active = sector_net[daily_weights.abs().sum(axis=1) > 0]

    return {
        "market_corr": float(df["strat"].corr(df["mkt"])),
        "realized_beta_mean": float(roll_beta.mean()),
        "realized_beta_p95_abs": float(roll_beta.abs().quantile(0.95)),
        "ex_ante_beta_mean_abs": float(port_beta.abs().mean()),
        "sector_net_mean_abs": float(active.abs().mean().mean()),
        "sector_net_max_abs": float(active.abs().max().max()),
        "n_sectors": int(sec_series.nunique()),
    }
