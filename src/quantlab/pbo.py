"""Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric
Cross-Validation (CSCV) — Bailey & López de Prado (2014), "The Probability of
Backtest Overfitting," Journal of Computational Finance.

The family-wise complement to the per-trial Deflated Sharpe Ratio in metrics.py.
DSR asks, of ONE strategy: is this Sharpe the lucky maximum of N noise draws?
PBO asks, of a FAMILY of N comparable variants backtested on the SAME data: if I
select the in-sample best, what is the probability it ranks BELOW the median
out-of-sample? PBO ≈ 0.5 means selection carries no OOS information (a coin
flip); PBO ≈ 0 means the in-sample winner reliably stays a winner OOS.

SCOPE — read before applying. CSCV needs a single (T × N) matrix whose columns
are the per-period returns of COMPARABLE variants over the SAME T observations.
It is a within-backtest, within-family diagnostic. It is NOT valid across
heterogeneous trials on different universes, frequencies, or asset classes — you
cannot rank those on a shared OOS slice, and pretending you can is exactly the
overfitting error this metric exists to catch. This project applies it to the
equity price-feature config family on the shared point-in-time universe (trials
#2/#3/#5/#6/#7 — same universe, same 21-day horizon), never across all 11 trials.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def _sharpe_columns(block: np.ndarray) -> np.ndarray:
    """Per-period Sharpe of each column (mean/std). Annualization is a positive
    monotone scaling that cancels in the within-split ranking, so it is omitted.
    A flat (zero-variance) column ranks last — it can never be the IS 'best'."""
    mu = block.mean(axis=0)
    sd = block.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, mu / sd, -np.inf)


def cscv_pbo(returns: pd.DataFrame, n_splits: int = 16) -> dict:
    """PBO via CSCV over a (T periods × N configs) return matrix.

    Split the T rows into ``n_splits`` contiguous equal blocks; for every way to
    choose half the blocks as in-sample (the complement is out-of-sample):
      1. pick the config with the best IS Sharpe (the selection a researcher makes);
      2. find that config's OOS Sharpe RANK among all configs;
      3. logit of its relative rank ω∈(0,1):  λ = ln(ω / (1−ω)).
    PBO = P(λ < 0) = the fraction of symmetric splits in which the IS-best config
    lands below the OOS median. Also returns the IS→OOS Sharpe degradation slope
    (of the selected config) and the probability it loses money OOS.
    """
    if n_splits % 2 != 0:
        raise ValueError(f"n_splits must be even (CSCV is symmetric), got {n_splits}")
    R = returns.dropna(how="any")
    T, N = R.shape
    if N < 2:
        raise ValueError(f"need >= 2 configs to rank, got {N}")
    if n_splits > T:
        raise ValueError(f"n_splits ({n_splits}) exceeds observations ({T})")
    M = R.to_numpy(dtype=float)
    block = T // n_splits
    blocks = [M[i * block:(i + 1) * block] for i in range(n_splits)]
    idx = list(range(n_splits))

    logits, sel_is_sr, sel_oos_sr = [], [], []
    for is_combo in combinations(idx, n_splits // 2):
        is_set = set(is_combo)
        IS = np.vstack([blocks[i] for i in is_combo])
        OOS = np.vstack([blocks[i] for i in idx if i not in is_set])
        is_perf = _sharpe_columns(IS)
        oos_perf = _sharpe_columns(OOS)
        n_star = int(np.argmax(is_perf))                       # the IS selection
        ranks = pd.Series(oos_perf).rank(method="average").to_numpy()  # ties averaged
        omega = ranks[n_star] / (N + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(float(np.log(omega / (1.0 - omega))))
        sel_is_sr.append(float(is_perf[n_star]))
        sel_oos_sr.append(float(oos_perf[n_star]))

    logits = np.asarray(logits)
    sel_is_sr = np.asarray(sel_is_sr)
    sel_oos_sr = np.asarray(sel_oos_sr)
    slope = (float(np.polyfit(sel_is_sr, sel_oos_sr, 1)[0])
             if len(sel_is_sr) > 1 and np.ptp(sel_is_sr) > 0 else float("nan"))
    return {
        "pbo": float((logits < 0).mean()),
        "n_configs": int(N),
        "n_obs": int(T),
        "n_splits": int(n_splits),
        "n_combinations": int(len(logits)),
        "median_logit": float(np.median(logits)),
        "perf_degradation_slope": slope,        # < 1 ⇒ OOS Sharpe decays vs IS
        "prob_oos_loss": float((sel_oos_sr < 0.0).mean()),
    }
