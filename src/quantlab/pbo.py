"""Combinatorially-Symmetric Cross-Validation (CSCV) and the Probability of
Backtest Overfitting (PBO), Bailey-Borwein-Lopez de Prado-Zhu 2015.

PBO complements the Deflated Sharpe Ratio (metrics.deflated_sharpe_ratio), it
does not duplicate it:
  - DSR takes ONE return series + a scalar trial count N and asks whether the
    final track record beats the luckiest of N noise draws.
  - PBO takes the full (T periods x N configs) matrix and asks whether the
    SELECTION RULE ("pick the in-sample best") generalizes out-of-sample.
Report both. DSR guards the track record against luck-of-N; PBO guards the
selection PROCESS that chose it.

Rows MUST be per-period RETURNS (not prices) on a common, time-ordered,
gap-free index. CSCV is symmetric under the IS<->OOS swap of complementary
block sets, but it is NOT invariant to row permutation: blocks are CONTIGUOUS,
so callers must pass a contiguous slice, never a dropna'd union of series with
different warm-up lengths.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _column_sharpes(matrix: np.ndarray) -> np.ndarray:
    """Per-config per-period Sharpe (mean / std, ddof=1) for every column of a
    (rows x N) block; returns shape (N,). A column with std exactly 0.0 maps to
    0.0 (no divide-by-zero); a bit-constant float column has ~1e-18 std (not
    exactly 0.0) and is NOT special-cased -- harmless because such columns tie
    and only relative rank order is used downstream. NOT annualized: only the
    RELATIVE ORDER is used for
    ranking, so the sqrt(periods) factor in metrics.sharpe is a positive
    monotone constant deliberately omitted. ddof=1 matches metrics.sharpe and
    the pandas convention. Assumes ``matrix`` is a contiguous row-slice of one
    config-return matrix."""
    mu = matrix.mean(axis=0)
    sd = matrix.std(axis=0, ddof=1)
    safe = np.where(sd > 0, sd, 1.0)  # avoid div-by-zero warning
    return np.where(sd > 0, mu / safe, 0.0)


def _make_blocks(n_periods: int, n_splits: int) -> list[np.ndarray]:
    """Partition row indices [0, n_periods) into n_splits disjoint CONTIGUOUS
    equal blocks of size n_periods // n_splits. Assumes n_periods % n_splits == 0
    (the caller drops the trailing remainder first). Returns n_splits int index
    arrays."""
    bs = n_periods // n_splits
    return [np.arange(i * bs, (i + 1) * bs) for i in range(n_splits)]


def cscv(returns: pd.DataFrame | np.ndarray, n_splits: int = 16) -> dict:
    """Full CSCV on a (T periods x N configs) per-period return matrix.

    Splits T into n_splits (S, must be EVEN) contiguous blocks. For each of the
    C(S, S/2) ways to choose S/2 blocks as in-sample (complement = out-of-sample):
    rank configs by IS Sharpe, take the IS-argmax n*, compute n*'s OOS Sharpe
    rank among all N configs (average ranks for ties, 1=worst..N=best), relative
    rank w = rank/(N+1), logit lambda = ln(w/(1-w)).

    Returns dict:
      'pbo'             : float, fraction of splits with lambda < 0 (STRICT).
      'logits'          : np.ndarray (n_combinations,)
      'n_splits'        : int (== S, the input n_splits)
      'n_combinations'  : int (== math.comb(S, S/2) == len(logits))
      'is_sharpe'       : np.ndarray (n_combinations,) IS-best config's IS Sharpe
      'oos_sharpe'      : np.ndarray (n_combinations,) IS-best config's OOS Sharpe
      'degradation'     : dict('slope','intercept','r_squared') OLS of oos_sharpe
                          on is_sharpe across all splits
      'prob_oos_loss'   : float, fraction of splits where IS-best OOS Sharpe < 0

    Assumes columns are independent candidate configs; rows are per-period
    returns on a contiguous time index. The last (T mod S) periods are DROPPED
    (trailing rows) to keep equal contiguous blocks (Bailey et al. assume
    divisible T); this keeps the OLDEST data. S and the equal-block partition
    are researcher degrees of freedom (law #3): S is a pinned, logged default."""
    if isinstance(returns, pd.DataFrame):
        M = returns.to_numpy(dtype=float)
    else:
        M = np.asarray(returns, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"returns must be 2-D (T x N), got shape {M.shape}")
    T, N = M.shape

    if n_splits % 2 != 0:
        raise ValueError(f"n_splits must be even, got {n_splits}")
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if N < 2:
        raise ValueError(f"need >= 2 configs to rank, got {N}")
    if T < n_splits:
        raise ValueError(f"need T >= n_splits, got T={T}, n_splits={n_splits}")
    if not np.isfinite(M).all():
        raise ValueError("returns contains non-finite values")

    usable = T - (T % n_splits)
    M = M[:usable]
    T = usable
    blocks = _make_blocks(T, n_splits)
    half = n_splits // 2

    logits: list[float] = []
    is_best_is: list[float] = []
    is_best_oos: list[float] = []
    for combo in itertools.combinations(range(n_splits), half):
        combo_set = set(combo)
        is_rows = np.concatenate([blocks[b] for b in combo])
        oos_rows = np.concatenate(
            [blocks[b] for b in range(n_splits) if b not in combo_set]
        )
        is_sh = _column_sharpes(M[is_rows])
        oos_sh = _column_sharpes(M[oos_rows])
        n_star = int(np.argmax(is_sh))  # lowest index on IS ties (deterministic)
        ranks = rankdata(oos_sh, method="average")  # 1=worst OOS .. N=best
        rank_star = ranks[n_star]
        w = rank_star / (N + 1)  # in (0,1) by construction -> finite logit, no clamp
        logits.append(math.log(w / (1.0 - w)))
        is_best_is.append(is_sh[n_star])
        is_best_oos.append(oos_sh[n_star])

    logits = np.asarray(logits)
    x = np.asarray(is_best_is)
    y = np.asarray(is_best_oos)
    pbo = float(np.mean(logits < 0))  # STRICT: lambda==0 (OOS median) is NOT overfit

    if len(x) >= 2 and np.var(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        ss_res = float(np.sum((y - yhat) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        slope = intercept = r_squared = float("nan")

    prob_oos_loss = float(np.mean(y < 0))
    return {
        "pbo": pbo,
        "logits": logits,
        "n_splits": int(n_splits),
        "n_combinations": math.comb(n_splits, half),
        "is_sharpe": x,
        "oos_sharpe": y,
        "degradation": {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_squared),
        },
        "prob_oos_loss": prob_oos_loss,
    }


def probability_of_backtest_overfitting(
    returns: pd.DataFrame | np.ndarray, n_splits: int = 16
) -> float:
    """Convenience wrapper returning cscv(returns, n_splits)['pbo']. PBO in
    [0,1]; high => the IS-selection rule does not generalize OOS."""
    return cscv(returns, n_splits)["pbo"]


def performance_degradation(
    returns: pd.DataFrame | np.ndarray, n_splits: int = 16
) -> dict:
    """Convenience wrapper returning cscv(returns, n_splits)['degradation'] =
    {'slope','intercept','r_squared'}: OLS of the IS-best config's OOS Sharpe on
    its IS Sharpe across all C(S, S/2) splits. Negative slope / low R^2 => IS
    performance does not carry to OOS (an overfitting signature)."""
    return cscv(returns, n_splits)["degradation"]
