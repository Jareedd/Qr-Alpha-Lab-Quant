"""Live beta-leak diagnosis (research menu §5.0): where does ~20% annualized
vol on a "neutral" live book come from?

The live monitor marks the INTENDED book (committed ``weights_*.csv`` x
public prices); the broker equity in ``summary_*.json`` is what the account
actually did. The two streams diverge exactly where execution failed —
orders rejected at the broker leave the HELD book different from the LOGGED
book — so decomposing each against the market separates three mutually
exclusive explanations for the vol:

  construction leak   the intended book carries realized beta the
                      at-construction projection failed to remove (beta
                      estimation error, or drift between rebalances);
  execution leak      the intended book is clean but broker equity is not
                      (e.g. a failed short-leg rebalance leaves a long tilt);
  idio concentration  both streams are ~beta-flat and the vol is simply what
                      a ~100-name daily decile L/S costs in idiosyncratic
                      terms — neutrality never promised low vol.

Pure functions only (no IO beyond the two thin loaders); the companion
script ``scripts/live_beta_diagnosis.py`` feeds them the committed audit
trail. This is monitoring-class analysis: read-only over committed
artifacts, no registration, no trial spent, the frozen live config
untouched.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Loaders (thin IO; the frame builder is pure and tested)
# ---------------------------------------------------------------------------

def load_summaries(live_dir: str) -> pd.DataFrame:
    """Read every ``summary_YYYY-MM-DD.json`` under ``live_dir``."""
    records = []
    for fn in sorted(os.listdir(live_dir)):
        if fn.startswith("summary_") and fn.endswith(".json"):
            with open(os.path.join(live_dir, fn)) as f:
                records.append(json.load(f))
    return summaries_frame(records)


def summaries_frame(records: list[dict]) -> pd.DataFrame:
    """(cycle date) x {equity, orders_sent, n_failed, failed_sells,
    failed_buys}.

    ``n_failed`` is the true failure count; the per-side split comes from the
    RETAINED failure list, which live.py caps at 20 entries — so the side
    columns are a lower bound on mass-failure days, never a total.
    """
    rows = {}
    for r in records:
        fails = r.get("orders_failed") or []
        sides = pd.Series([f.get("side") for f in fails if isinstance(f, dict)])
        rows[pd.Timestamp(r["asof"])] = {
            "equity": float(r["equity"]) if r.get("equity") is not None else np.nan,
            "orders_sent": r.get("orders_sent", np.nan),
            "n_failed": r.get("n_failed", len(fails)),
            "failed_sells": int((sides == "sell").sum()),
            "failed_buys": int((sides == "buy").sum()),
        }
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


# ---------------------------------------------------------------------------
# Return streams
# ---------------------------------------------------------------------------

def broker_returns(equity: pd.Series) -> pd.Series:
    """Per-cycle simple returns of broker equity, indexed by cycle date.

    Consecutive cycles can span a weekend/holiday; at this resolution a
    per-cycle return IS the trading-day return (cycles are daily on NYSE
    days), so no calendar re-scaling is applied.
    """
    eq = equity.astype(float).sort_index().dropna()
    return eq.pct_change().dropna()


def execution_gap(broker: pd.Series, intended: pd.Series) -> pd.DataFrame:
    """Aligned per-cycle comparison: gap = broker − intended.

    The gap is the return of what the broker actually held MINUS the return
    of the logged target book; a string of near-zero gaps means orders
    tracked the target, a large one-sided run means the held book departed
    from it (failed legs, fills, rounding).
    """
    df = pd.concat([broker, intended], axis=1, keys=["broker", "intended"]).dropna()
    df["gap"] = df["broker"] - df["intended"]
    df["cum_gap"] = (1 + df["broker"]).cumprod() - (1 + df["intended"]).cumprod()
    return df


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

def beta_decomposition(strat: pd.Series, mkt: pd.Series) -> dict:
    """Full-window OLS of a strategy stream on the market + variance shares.

    Returns annualized vols, beta with its iid OLS standard error (declared:
    optimistic under autocorrelation — with ~40 obs a NW correction is
    decoration, so the caveat is stated instead), correlation, and the
    market-explained variance share beta² var(mkt)/var(strat). What the
    share leaves behind is sector+idio by construction.
    """
    df = pd.concat([strat, mkt], axis=1, keys=["s", "m"]).dropna()
    n = len(df)
    if n < 3:
        raise ValueError(f"need >= 3 overlapping observations, got {n}")
    s, m = df["s"].to_numpy(), df["m"].to_numpy()
    var_m, var_s = m.var(ddof=1), s.var(ddof=1)
    beta = float(np.cov(s, m, ddof=1)[0, 1] / var_m)
    resid = (s - s.mean()) - beta * (m - m.mean())
    beta_se = float(np.sqrt(resid.var(ddof=1) / ((n - 1) * var_m)))
    alpha_ann = float((s.mean() - beta * m.mean()) * TRADING_DAYS)
    return {
        "n_obs": int(n),
        "ann_vol_strat": float(np.sqrt(var_s * TRADING_DAYS)),
        "ann_vol_mkt": float(np.sqrt(var_m * TRADING_DAYS)),
        "beta": beta,
        "beta_se_iid": beta_se,
        "alpha_ann_gross": alpha_ann,
        "corr": float(np.corrcoef(s, m)[0, 1]),
        "mkt_var_share": float(beta**2 * var_m / var_s),
        "resid_ann_vol": float(np.sqrt(resid.var(ddof=1) * TRADING_DAYS)),
    }


def rolling_beta_series(strat: pd.Series, mkt: pd.Series, window: int = 15) -> pd.Series:
    """Rolling OLS beta of the strategy on the market (declared window).

    The live record is ~40 cycles, so the window must be far shorter than
    risk_report's 63-day default; 15 trades noise for coverage and the
    window is reported alongside every number it produces.
    """
    df = pd.concat([strat, mkt], axis=1, keys=["s", "m"]).dropna()
    cov = df["s"].rolling(window, min_periods=window).cov(df["m"])
    var = df["m"].rolling(window, min_periods=window).var()
    return (cov / var).dropna()


def exante_beta_path(
    weights_by_date: dict[pd.Timestamp, pd.Series], betas: pd.DataFrame
) -> pd.Series:
    """w·β at each cycle date — what construction BELIEVED its beta was.

    Mirrors ``risk.beta_neutralize_weights`` exactly: the latest beta row
    at-or-before the cycle date, reindexed to the active names, missing
    entries filled with the cross-sectional mean of the active names. On a
    book that went through that projection this is ~0 by construction; a
    nonzero value here means the SCRIPT's beta inputs differ from the ones
    construction saw (a reconstruction approximation, reported as such).
    """
    out = {}
    for d, w in sorted(weights_by_date.items()):
        active = w[w != 0.0]
        bt = betas.loc[:d]
        if bt.empty or len(active) < 3:
            continue
        b = bt.iloc[-1].reindex(active.index)
        b = b.fillna(b.mean())
        if b.isna().all():
            continue
        out[d] = float((active * b).sum())
    return pd.Series(out, dtype=float).sort_index()
