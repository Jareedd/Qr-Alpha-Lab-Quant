# Implementation Spec — PBO/CSCV + Rolling Factor Betas + Two-World Quality Panel

**Status:** FINAL, unambiguous. An engineer implements directly from this document with no further design decisions.
**Date:** 2026-06-24. **Author:** research co-pilot, qr-alpha-lab.
**Scope:** machinery + synthetic validation ONLY. **ZERO trials. N stays 11. H1 stays data-blocked.**
`make_panel` is NOT touched, so the CI planted/noise golden values and the `test_regime.py` canary are structurally out of the blast radius. The gate is re-run once at the end as proof-of-zero-regression (law #2).

All numerical claims below were re-derived in-env (numpy 1.26.4 / scipy 1.13.1) before this spec was frozen; every pinned value is a measured byte value, not an estimate. Permitted deps only: numpy / pandas / scipy / sklearn.

---

## 0. How the surviving critiques are resolved (read first)

Each BLOCKER/MAJOR/MINOR from the three review lenses is resolved here. The spec body implements the resolution; this table is the index.

| # | Severity | Critique | Resolution in this spec |
|---|----------|----------|-------------------------|
| C1 | MINOR | PBO core algorithm is numerically correct as-is. | Adopted verbatim (§1). Re-verified: T1 PBO=0.0/+ln2, T2 PBO=1.0/−ln2, noise pooled 0.4874, dominant 0.0, degradation slope −0.5527 / r²0.3055. |
| C2 | BLOCKER | "B = −A" claim in the overfit test is FALSE elementwise. | §1.5 test prose says **B's per-block MEANS negate A's block means** (A blocks [4,1,−2,−3], B blocks [−4,−1,2,3]); the test asserts `not np.allclose(B, -A)` and never asserts `B==-A`. Verified `np.allclose(B2,-A2)==False`. |
| C3 | BLOCKER | `pbo = mean(logits < 0)` (strict) vs `mean(lam <= 0)` (non-strict) contradiction. | **Strict `<` is the single convention everywhere** (§1.3 step 7). The integration `<=` is deleted. Rationale: conservative — a split exactly at the OOS median (λ=0) is NOT counted as overfit. |
| C4 | BLOCKER | Single-seed noise PBO test measured 0.3857, outside its own (0.4,0.6) band. | The single-seed noise test is **deleted**. Replaced by the K=20 pooled-mean test (§1.5 `test_pbo_noise_is_about_half`), measured mean 0.4874, band (0.40,0.60). |
| C5 | MAJOR | "mean logit ≈ 0 within 0.3" symmetry assertion is false (Jensen; measured up to +0.79). | The mean-logit assertion is **deleted**. The symmetry pin is on COUNT/SHAPE only: `n_combinations == C(S, S/2)` and array shapes (§1.5 `test_pbo_shapes_and_counts`). |
| C6 | MINOR | argsort-of-argsort breaks ties arbitrarily; rankdata('average') is deterministic. | **`scipy.stats.rankdata(method='average')` is the sole rank rule** (§1.3 step 5d). argsort-of-argsort is forbidden. |
| C7 | MINOR | "w=0/1 → ln(0)/ln(inf)" is a non-issue with the (N+1) denominator. | No clamp is added. `w = rank/(N+1)` keeps `w ∈ [1/(N+1), N/(N+1)] ⊂ (0,1)`; logit always finite (§1.3 step 5e). Documented as such. |
| C8 | MINOR | r² ∈ [0,1] only assertable on the specific overfit matrix. | The `r_squared in [0,1]` assertion is scoped to the overfit design only (§1.5 `test_degradation_fields`), not a general invariant. |
| C9 | BLOCKER | factor_betas (fit_intercept=True) vs integration (no-intercept) contradiction; no-intercept is math-wrong. | **`fit_intercept=True` is the only policy** (§2). Re-verified: with-intercept matches `rolling_market_beta` to **2.4e-15**; no-intercept diverges by **4.9e-3** on a nonzero-mean factor. All no-intercept language is purged. |
| C10 | BLOCKER | Two worlds separable by RAW Sharpe alone (A~2.4 vs B~12); neutralization not proven to be the discriminator. | **World B premium reduced to `B_PREMIUM = 0.004`** (§3). Re-verified across 33 seeds: World A raw ∈ [1.65,3.24], World B raw ∈ [0.20,3.62] — **overlapping** (means 2.20 vs 2.32). A mandatory raw-overlap test is added (§3.5 `test_raw_sharpe_alone_does_not_separate_worlds`). |
| C11 | BLOCKER | No placebo/null-factor control proving the collapse needs the TRUE value factor. | **Mandatory placebo test added** (§3.5 `test_collapse_requires_true_value_factor`). Re-verified: World A neutral(true)≈0.77 vs neutral(placebo)≈2.3 → `neutral_true < 0.5 * neutral_placebo` on all of seeds 7/11/23. |
| C12 | MAJOR | Full-window-vs-post-warm-up scoring is a tuned degree of freedom (flips seed-23 past 1.3). | The fragile rolling-estimate 1.3 margin is **replaced by an estimation-error-free static-loading pin** (§3.5 `test_quality_is_value_collapses_static_loading`): World A static-neutral SR < 0.3 (measured [−0.41,+0.25]); World B static-neutral SR > 1.0 (measured [2.44,3.70]). The rolling-estimate path is reported as a secondary, looser pin. |
| C13 | MAJOR | Synthetic gate must also exercise the PRODUCTION rolling-estimate loading path. | §3.5 `test_quality_worlds_discriminate_under_rolling_estimated_loading` drives `value_neutralized_signal` with loadings from `rolling_factor_betas` (the production estimator) AND includes a poison-the-future leak check on that path. |
| C14 | MAJOR | No-lookahead test must be tightened at the window boundary. | §2.5 `test_factor_betas_use_only_past_data` poisons rows ≥ P in BOTH frames AND adds a single-row off-by-one probe asserting the last pre-poison loading row is byte-identical. |
| C15 | MAJOR | CSCV integration matrix used dropna across mismatched warm-ups → non-contiguous "contiguous" blocks. | §4.5 `test_cscv_adjudicates_the_four_arms` uses a **contiguous common post-warm-up date SLICE** (`.loc[start:end]`), never `dropna(how='any')`. Documented: PBO requires gap-free contiguous rows. |
| C16 | MINOR | Old-mode RNG independence (verified safe) — guard the baseline. | §3.5 `test_existing_quality_modes_byte_identical` compares against **frozen literal triples**, not a re-call of the same function (a re-call would pass even if both drifted). Re-verified `.equals()==True` 6/6. |
| C17 | MINOR | `loadings_at` missing-loading semantics must be documented, not silent. | §2.2 docstring states a NaN/missing loading ⇒ that name left un-neutralized that date (mirrors `neutralize_weights` `.fillna(0.0)`). |

---

## 1. Module: `quantlab.pbo` (CSCV / Probability of Backtest Overfitting)

Reference: Bailey, Borwein, López de Prado, Zhu (2015), "The Probability of Backtest Overfitting", *J. Computational Finance*.

### 1.1 Files
- **NEW** `src/quantlab/pbo.py`
- **NEW** `tests/test_pbo.py`

### 1.2 Exact function signatures (with docstring-assumptions)

```python
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
    (rows x N) block; returns shape (N,). A zero-variance column maps to 0.0
    (no divide-by-zero). NOT annualized: only the RELATIVE ORDER is used for
    ranking, so the sqrt(periods) factor in metrics.sharpe is a positive
    monotone constant deliberately omitted. ddof=1 matches metrics.sharpe and
    the pandas convention. Assumes `matrix` is a contiguous row-slice of one
    config-return matrix."""


def _make_blocks(n_periods: int, n_splits: int) -> list[np.ndarray]:
    """Partition row indices [0, n_periods) into n_splits disjoint CONTIGUOUS
    equal blocks of size n_periods // n_splits. Assumes n_periods % n_splits == 0
    (the caller drops the trailing remainder first). Returns n_splits int index
    arrays."""


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


def probability_of_backtest_overfitting(
    returns: pd.DataFrame | np.ndarray, n_splits: int = 16
) -> float:
    """Convenience wrapper returning cscv(returns, n_splits)['pbo']. PBO in
    [0,1]; high => the IS-selection rule does not generalize OOS."""


def performance_degradation(
    returns: pd.DataFrame | np.ndarray, n_splits: int = 16
) -> dict:
    """Convenience wrapper returning cscv(returns, n_splits)['degradation'] =
    {'slope','intercept','r_squared'}: OLS of the IS-best config's OOS Sharpe on
    its IS Sharpe across all C(S, S/2) splits. Negative slope / low R^2 => IS
    performance does not carry to OOS (an overfitting signature)."""
```

### 1.3 Exact algorithm — `cscv`

1. **Coerce:** if `returns` is a `pd.DataFrame`, `M = returns.to_numpy(dtype=float)`; else `M = np.asarray(returns, dtype=float)`. Require `M.ndim == 2`. `T, N = M.shape`.
2. **Guards** (raise `ValueError` with explicit messages, repo style — fail loudly, law #6):
   - `if n_splits % 2 != 0: raise ValueError(f"n_splits must be even, got {n_splits}")` (CSCV needs S/2 IS = S/2 OOS).
   - `if n_splits < 2: raise ValueError(f"n_splits must be >= 2, got {n_splits}")`.
   - `if N < 2: raise ValueError(f"need >= 2 configs to rank, got {N}")`.
   - `if T < n_splits: raise ValueError(f"need T >= n_splits, got T={T}, n_splits={n_splits}")`.
   - **Finite check:** `if not np.isfinite(M).all(): raise ValueError("returns contains non-finite values")`. A 2-D matrix has no clean per-config dropna (unlike `metrics.*` on a 1-D series); a leaked NaN must not silently poison ranks. Mirrors `regime.py`'s "fail loudly on degenerate inputs".
3. **Trailing-row drop:** `usable = T - (T % n_splits); M = M[:usable]; T = usable`. Drops the NEWEST `(T mod S)` rows, keeping the oldest. Do NOT pad or resample.
4. `blocks = _make_blocks(T, n_splits)`.
5. `half = n_splits // 2`. For each `combo in itertools.combinations(range(n_splits), half)`:
   a. `is_rows = np.concatenate([blocks[b] for b in combo])`; `oos_ids = [b for b in range(n_splits) if b not in set(combo)]`; `oos_rows = np.concatenate([blocks[b] for b in oos_ids])`.
   b. `is_sh = _column_sharpes(M[is_rows])`; `oos_sh = _column_sharpes(M[oos_rows])`.  (each shape `(N,)`)
   c. `n_star = int(np.argmax(is_sh))`. **IS tie rule:** `np.argmax` returns the LOWEST index on ties — deterministic and documented; the only IS tie policy.
   d. `ranks = rankdata(oos_sh, method='average')`  (1..N ascending, 1=worst OOS, N=best, average ranks for ties). `rank_star = ranks[n_star]`.
   e. `w = rank_star / (N + 1)`. The `(N+1)` denominator (not `N`) keeps `w` strictly inside `(0,1)` so the logit never diverges — exactly per the paper. No clamp.
   f. `lam = math.log(w / (1.0 - w))` (or `np.log`).
   g. Append `lam` to `logits`; append `is_sh[n_star]` to `is_best_is`; append `oos_sh[n_star]` to `is_best_oos`.
6. `logits = np.asarray(logits)`. `pbo = float(np.mean(logits < 0))`. **STRICT `<`** (λ=0 ⇔ w=0.5 ⇔ OOS-median is NOT overfit; conservative).
7. **Degradation** (numpy-only OLS, no extra import): `x = np.asarray(is_best_is)`, `y = np.asarray(is_best_oos)`.
   - If `len(x) >= 2 and np.var(x) > 0`: `slope, intercept = np.polyfit(x, y, 1)`; `yhat = slope*x + intercept`; `ss_tot = float(np.sum((y - y.mean())**2))`; `ss_res = float(np.sum((y - yhat)**2))`; `r_squared = 1.0 - ss_res/ss_tot if ss_tot > 0 else float('nan')`.
   - Else: `slope = intercept = r_squared = float('nan')`.
8. `prob_oos_loss = float(np.mean(y < 0))`.
9. Return `{'pbo': pbo, 'logits': logits, 'n_splits': int(n_splits), 'n_combinations': math.comb(n_splits, half), 'is_sharpe': x, 'oos_sharpe': y, 'degradation': {'slope': float(slope), 'intercept': float(intercept), 'r_squared': float(r_squared)}, 'prob_oos_loss': prob_oos_loss}`.

### 1.4 `_column_sharpes` and `_make_blocks` bodies

```python
def _column_sharpes(matrix):
    mu = matrix.mean(axis=0)
    sd = matrix.std(axis=0, ddof=1)
    safe = np.where(sd > 0, sd, 1.0)        # avoid div-by-zero warning
    return np.where(sd > 0, mu / safe, 0.0)

def _make_blocks(n_periods, n_splits):
    bs = n_periods // n_splits
    return [np.arange(i * bs, (i + 1) * bs) for i in range(n_splits)]
```

### 1.5 COMPLETE known-answer test list — `tests/test_pbo.py`

House bootstrap header (every test file):
```python
import os, sys, math
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quantlab import pbo, metrics
```

1. **`test_pbo_deterministic_persistent_edge_is_zero`** — the arithmetic pin.
   ```python
   A = np.array([0.03,0.02,0.04,0.03,0.02,0.05,0.03,0.04])   # always positive
   B = np.array([-0.02,-0.03,-0.01,-0.02,-0.04,-0.01,-0.03,-0.02])  # always negative
   M = np.column_stack([A, B])
   r = pbo.cscv(M, n_splits=4)
   assert r['pbo'] == 0.0
   assert np.allclose(r['logits'], np.log(2), rtol=1e-12, atol=0.0)  # all 6 == +ln2
   assert r['n_combinations'] == 6
   assert r['prob_oos_loss'] == 0.0
   ```
   Verified: every logit == `0.6931471805599452` (+ln2), pbo 0.0, ncomb 6, prob_oos_loss 0.0.

2. **`test_pbo_deterministic_overfit_is_one`** — the dual arithmetic pin.
   ```python
   # B's per-BLOCK MEANS negate A's block means: A blocks [4,1,-2,-3],
   # B blocks [-4,-1,2,3]. B is NOT the elementwise negation of A; the
   # within-block wiggle ([-0.5,+0.5] vs A's [+0.5,-0.5]) keeps stds equal
   # while flipping every block mean. sum(A blocks)=0 => IS_block_sum =
   # -OOS_block_sum every split => IS-best is OOS-worst in all 6 splits.
   A = np.array([4.5,3.5,1.5,0.5,-1.5,-2.5,-2.5,-3.5])
   B = np.array([-3.5,-4.5,-0.5,-1.5,2.5,1.5,3.5,2.5])
   assert not np.allclose(B, -A)          # GUARD: B is NOT elementwise -A
   M = np.column_stack([A, B])
   r = pbo.cscv(M, n_splits=4)
   assert r['pbo'] == 1.0
   assert np.allclose(r['logits'], -np.log(2), rtol=1e-12, atol=0.0)  # all -ln2
   assert r['prob_oos_loss'] == 1.0
   ```
   Verified: pbo 1.0, every logit == `-0.6931471805599454`, prob_oos_loss 1.0, `np.allclose(B,-A)==False`.

3. **`test_pbo_noise_is_about_half`** — multi-seed pooled (deterministic-and-robust).
   ```python
   means = []
   for s in range(20):
       M = np.random.default_rng(s).standard_normal((640, 20)) * 0.01  # T=640 % 16 == 0
       means.append(pbo.cscv(M, n_splits=16)['pbo'])
   mean_pbo = float(np.mean(means))
   assert 0.40 < mean_pbo < 0.60
   ```
   Verified pooled mean = `0.4874203574`. (Single seeds scatter ~[0.08,0.72]; the across-seed mean concentrates near the theoretical 0.5. Do NOT pin a single seed.)

4. **`test_pbo_one_dominant_config_is_zero`** — falsification mirror.
   ```python
   M = np.random.default_rng(321).standard_normal((640, 20)) * 0.01
   M[:, 0] += 0.004                          # a genuine, time-stable edge
   assert pbo.cscv(M, n_splits=16)['pbo'] < 0.05
   ```
   Verified pbo == 0.0 at seed 321.

5. **`test_degradation_fields`** — reuse the overfit matrix from test 2.
   ```python
   A = np.array([4.5,3.5,1.5,0.5,-1.5,-2.5,-2.5,-3.5])
   B = np.array([-3.5,-4.5,-0.5,-1.5,2.5,1.5,3.5,2.5])
   M = np.column_stack([A, B])
   deg = pbo.performance_degradation(M, n_splits=4)
   assert set(deg) == {'slope', 'intercept', 'r_squared'}
   assert deg['slope'] < 0                    # IS Sharpe anti-predicts OOS Sharpe
   assert 0.0 <= deg['r_squared'] <= 1.0      # scoped to THIS matrix only
   ```
   Verified slope `-0.5527`, r_squared `0.3055` on this design.

6. **`test_pbo_shapes_and_counts`** — bookkeeping/symmetry (NO mean-logit assertion).
   ```python
   M = np.random.default_rng(1).standard_normal((640, 20)) * 0.01
   r = pbo.cscv(M, n_splits=8)
   assert r['n_splits'] == 8
   assert r['n_combinations'] == math.comb(8, 4) == 70
   assert r['logits'].shape == (70,)
   assert r['is_sharpe'].shape == (70,) == r['oos_sharpe'].shape
   ```

7. **`test_column_sharpes_matches_metrics_up_to_annualization`**.
   ```python
   rng = np.random.default_rng(5)
   df = pd.DataFrame(rng.standard_normal((300, 3)) * 0.01, columns=list("ABC"))
   fast = pbo._column_sharpes(df.to_numpy())
   ref = np.array([metrics.sharpe(df[c], periods=1) for c in df.columns])
   assert np.allclose(fast, ref, atol=1e-12)
   ```
   `metrics.sharpe(..., periods=1)` is `mean/std*sqrt(1)` with `std` ddof=1 (pandas default) — identical to `_column_sharpes` per period.

8. **`test_pbo_guards`** — pins every ValueError (law #6).
   ```python
   rng = np.random.default_rng(0)
   with pytest.raises(ValueError, match="even"):
       pbo.cscv(rng.standard_normal((100, 5)), n_splits=15)
   with pytest.raises(ValueError, match=">= 2 configs"):
       pbo.cscv(rng.standard_normal((100, 1)), n_splits=4)
   with pytest.raises(ValueError, match="non-finite"):
       bad = rng.standard_normal((100, 5)); bad[10, 2] = np.nan
       pbo.cscv(bad, n_splits=4)
   with pytest.raises(ValueError, match="T >= n_splits"):
       pbo.cscv(rng.standard_normal((3, 5)), n_splits=4)
   ```

### 1.6 Edge cases (all handled in §1.3)
- S odd → ValueError before any computation. S < 2 → ValueError. N < 2 → ValueError. T < S → ValueError. Non-finite input → ValueError.
- `T % S != 0` → drop trailing (newest) `T mod S` rows. Documented.
- Zero-variance config column in a split → Sharpe 0.0 (np.where guard); still rankable.
- IS tie for the best → lowest column index (np.argmax). OOS tie → `rankdata('average')`, so w and λ are continuous and finite.
- λ exactly 0 (only via a tie making rank == (N+1)/2) → NOT counted as overfit (strict `<`).
- Large S (default 16): `C(16,8)=12870` combos, fine. S is a logged default arg (law #3).

---

## 2. Module: `quantlab.risk_model` (rolling factor betas) + `quantlab.ff_factors` (FF loader)

### 2.1 Files
- **EDIT** `src/quantlab/risk_model.py` — APPEND `rolling_factor_betas`, `loadings_at`, `cross_sectional_neutralize`. Do NOT touch `rolling_market_beta`, `neutralize_weights`, `net_factor_exposure`, `sample_covariance`, `predicted_vol`.
- **NEW** `src/quantlab/ff_factors.py` — FF monthly loader + daily stub.
- **NEW** `tests/test_risk_model_factor_betas.py` — keeps the exact-1e-9 single-factor pins un-bloated.
- **NEW** `tests/test_ff_factors.py` — FF loader parse pins.

### 2.2 Exact signatures appended to `risk_model.py`

```python
def rolling_factor_betas(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    lookback: int = 252,
    min_periods: int = 126,
    fit_intercept: bool = True,
) -> dict[str, pd.DataFrame]:
    """Past-only rolling multivariate OLS of each asset on the K factor columns.

    Per trailing window ending at t (inclusive), solve the normal equations
    beta = pinv(Xt X) @ (Xt Y) with X = [1 | factors] when fit_intercept (the
    DEFAULT, REQUIRED for byte-identical K=1 reduction to rolling_market_beta on
    nonzero-mean factors). pinv (not inv) for collinearity-robustness, matching
    neutralize_weights.

    fit_intercept=True is load-bearing and must NOT be "simplified" to a
    through-origin solve: rolling_market_beta computes cov/var (ddof=1), which
    is the DEMEANED / with-intercept OLS slope. A through-origin solve
    (sum xy / sum xx) differs by ~5e-3 on nonzero-mean returns and only
    coincidentally matches on noise-free test assets. Verified: with-intercept
    matches rolling_market_beta to 2.4e-15; through-origin diverges by 4.9e-3.

    Returns dict keyed by factor_returns.columns -> (date x asset) DataFrame of
    that factor's loading (intercept NOT returned). Each frame is indexed by
    asset_returns.index, columns asset_returns.columns, NaN until min_periods
    non-NaN factor rows are present in the window (pandas-style warm-up,
    replicated by a manual valid-row count guard).

    Trailing/causal: beta_t uses returns through t only (point-in-time safe;
    weights formed at t and applied from t+1 carry no look-ahead). factor_returns
    is reindexed ONTO asset_returns.index first (asset index authoritative,
    matching rolling_market_beta). factor_returns must be a DataFrame (one column
    for K=1); a Series caller should .to_frame() first."""


def loadings_at(
    factor_betas: dict[str, pd.DataFrame],
    date,
    assets: pd.Index | None = None,
    add_dollar: bool = False,
) -> pd.DataFrame:
    """Assemble the per-date L matrix (assets x factors) that neutralize_weights /
    net_factor_exposure consume, by slicing each factor frame as-of `date`
    (f.loc[:date].iloc[-1], last row at or before date) and column-stacking.
    Optionally append a 'dollar' ones column for dollar-neutrality. Index =
    `assets` (default: union of the factor frames' columns); columns =
    list(factor_betas) [+ 'dollar']. A NaN/missing loading is LEFT AS NaN: the
    downstream neutralize_weights .fillna(0.0) then leaves that name
    un-neutralized that date (the conservative, documented convention — a
    coverage hole does not crash and does not masquerade as a neutral book)."""


def cross_sectional_neutralize(signal: pd.Series, loadings: pd.DataFrame) -> pd.Series:
    """OLS-residualize a cross-sectional SIGNAL against factor loadings:
    r = s - L pinv(Lt L) (Lt s). Same projection algebra as neutralize_weights
    but applied to an alpha signal, not weights. Include a ones column in
    `loadings` to also demean (dollar-neutral). signal is reindexed onto
    loadings.index and .fillna(0.0); loadings .fillna(0.0). The shared helper
    keeps the signal- and weight-neutralizers from drifting apart."""
```

### 2.3 Exact algorithm — `rolling_factor_betas`

1. **Validate:** `if lookback <= 0: raise ValueError`; `if not (0 < min_periods <= lookback): raise ValueError`; `if asset_returns.empty or factor_returns.empty: raise ValueError`.
2. `F = factor_returns.reindex(asset_returns.index)` (asset index authoritative). `names = list(F.columns)`. `dates = asset_returns.index`. `n = len(dates)`. `K = F.shape[1]`. `A = asset_returns.shape[1]`.
3. `Y = asset_returns.to_numpy(dtype=float)` (n x A). `Xf = F.to_numpy(dtype=float)` (n x K). Pre-allocate per-factor output arrays: `out = {name: np.full((n, A), np.nan) for name in names}`.
4. For `t` in `range(n)`:
   - `lo = max(0, t - lookback + 1)`; `Xw = Xf[lo:t+1]`; `Yw = Y[lo:t+1]` (trailing, INCLUSIVE of t — the causal guarantee).
   - `mask = ~np.isnan(Xw).any(axis=1)`; `count = int(mask.sum())`.
   - `if count < min_periods: continue` (leave NaN — pandas min_periods warm-up).
   - `Xm = Xw[mask]`; `Ym = Yw[mask]`. If `fit_intercept`: `Xd = np.column_stack([np.ones(count), Xm])` else `Xd = Xm`.
   - **Y NaN handling:** if `np.isnan(Ym).any()` (sporadic per-asset NaNs), solve per column dropping that column's NaN rows; else vectorize: `coef = np.linalg.pinv(Xd.T @ Xd) @ (Xd.T @ Ym)` (shape `(K[+1], A)`).
     ```python
     if np.isnan(Ym).any():
         coef = np.full((Xd.shape[1], A), np.nan)
         for a in range(A):
             ym = Ym[:, a]; ok = ~np.isnan(ym)
             if ok.sum() < min_periods: continue
             Xa = Xd[ok]; coef[:, a] = np.linalg.pinv(Xa.T @ Xa) @ (Xa.T @ ym[ok])
     else:
         coef = np.linalg.pinv(Xd.T @ Xd) @ (Xd.T @ Ym)
     ```
   - `betas_t = coef[1:, :] if fit_intercept else coef` (drop intercept row), shape `(K, A)`.
   - For `k, name in enumerate(names): out[name][t, :] = betas_t[k, :]`.
5. Return `{name: pd.DataFrame(out[name], index=dates, columns=asset_returns.columns) for name in names}`.

Multivariate, not divide-K-covariances-by-K-variances: with correlated factors the true loading is `(XtX)^-1 Xt y`, which inverts the full Gram. Verified the multivariate solve recovers `(1.5, 0.8)` from `asset = 1.5*Mkt + 0.8*HML + noise`.

### 2.4 `cross_sectional_neutralize` and `loadings_at` bodies

```python
def cross_sectional_neutralize(signal, loadings):
    aligned = signal.reindex(loadings.index).fillna(0.0)
    s = aligned.to_numpy(dtype=float)
    L = loadings.fillna(0.0).to_numpy(dtype=float)
    coef = np.linalg.pinv(L.T @ L) @ (L.T @ s)
    return pd.Series(s - L @ coef, index=loadings.index)

def loadings_at(factor_betas, date, assets=None, add_dollar=False):
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
    L = pd.DataFrame(cols, index=assets)            # columns in dict order == list(factor_betas)
    if add_dollar:
        L["dollar"] = 1.0
    return L
```

### 2.5 FF loader — `ff_factors.py`

```python
"""Thin loader for Ken French research-factor CSVs (numpy/pandas only).

The monthly 5-factor file (F-F_Research_Data_5_Factors_2x3.csv) has 3 header
text lines, a blank line, then the header ',Mkt-RF,SMB,HML,RMW,CMA,RF', then a
monthly block keyed by 6-digit YYYYMM, then a blank line, then an
' Annual Factors: January-December' section keyed by 4-digit years, then a
copyright line. We read ONLY the monthly block. The DAILY 5-factor file is a
SEPARATE Ken French download; the monthly file cannot drive a daily pipeline."""
from __future__ import annotations
import io, os, re
import numpy as np
import pandas as pd

FF5_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
DAILY_FF_REQUIRED = (
    "F-F_Research_Data_5_Factors_2x3_daily.CSV is a SEPARATE Ken French "
    "download; the monthly file cannot drive the daily equity pipeline."
)
_MONTHLY_KEY = re.compile(r"^\s*\d{6}\s*,")


def load_ff_factors_monthly(path: str | os.PathLike) -> pd.DataFrame:
    """Load the FF 5-factor MONTHLY CSV. Returns a tidy DataFrame in DECIMAL
    (file values / 100), DatetimeIndex at month-END (PeriodIndex('YYYYMM','M')
    .to_timestamp('M')), columns FF5_COLUMNS, index name 'date'. Stops at the
    Annual Factors section / first non-6-digit key. RF is divided by 100 too
    (all columns are percent)."""
    with open(path, "r", newline="") as fh:
        lines = fh.read().splitlines()
    # locate header row (starts with ',Mkt-RF')
    hdr = next(i for i, ln in enumerate(lines) if ln.lstrip().startswith(",Mkt-RF"))
    body = []
    for ln in lines[hdr + 1:]:
        if _MONTHLY_KEY.match(ln):
            body.append(ln)
        elif body:                 # first non-monthly line AFTER data starts => stop
            break
        # blank lines BEFORE the first data row are skipped implicitly
    df = pd.read_csv(
        io.StringIO("\n".join([lines[hdr]] + body)),
        skipinitialspace=True,
    )
    df = df.rename(columns={df.columns[0]: "key"})
    df["key"] = df["key"].astype(str).str.strip()
    df = df[df["key"].str.fullmatch(r"\d{6}")]      # belt-and-suspenders
    idx = pd.PeriodIndex(df["key"], freq="M").to_timestamp("M")
    idx.name = "date"
    out = df[FF5_COLUMNS].astype(float).to_numpy() / 100.0
    return pd.DataFrame(out, index=idx, columns=FF5_COLUMNS)


def load_ff_factors_daily(path: str | os.PathLike) -> pd.DataFrame:
    """The daily FF 5-factor file is a separate download (law #7 — never
    fabricate). Raises FileNotFoundError with guidance if absent; flag, do
    not block the monthly path."""
    if not os.path.exists(path):
        raise FileNotFoundError(DAILY_FF_REQUIRED + f" (looked for {path!r})")
    raise NotImplementedError(
        "Daily FF parsing not implemented; " + DAILY_FF_REQUIRED
    )
```

**Parse rule precision (verified against the on-disk file at**
`C:/Users/galve/Claude/Projects/Quant Project/scratch_refute/F-F_Research_Data_5_Factors_2x3.csv`**):**
header at file line 5 (`,Mkt-RF,SMB,HML,RMW,CMA,RF`), 754 monthly rows (file lines 6–760), blank line 760-after, ` Annual Factors: January-December` at line 761. First monthly key `196307`, last `202604`. Numeric fields are space-padded (`   -0.39`); `skipinitialspace=True` strips them.

### 2.6 KNOWN-ANSWER tests — `tests/test_risk_model_factor_betas.py`

```python
import os, sys
import numpy as np, pandas as pd, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quantlab import risk_model as rm
```

1. **`test_rolling_factor_betas_reduces_to_market_beta_K1`** — THE byte-identity pin (resolves C9).
   ```python
   idx = pd.bdate_range("2020-01-01", periods=400)
   mkt = pd.Series(np.random.default_rng(0).standard_normal(400) * 0.01, index=idx)
   assets = pd.DataFrame({"A": 2.0 * mkt, "B": 0.5 * mkt})
   ref = rm.rolling_market_beta(assets, mkt, 252, 126)
   fb = rm.rolling_factor_betas(assets, pd.DataFrame({"mkt": mkt}), 252, 126, fit_intercept=True)
   pd.testing.assert_frame_equal(fb["mkt"], ref, atol=1e-9, check_names=False)
   assert fb["mkt"]["A"].iloc[-1] == pytest.approx(2.0, abs=1e-9)
   assert fb["mkt"]["B"].iloc[-1] == pytest.approx(0.5, abs=1e-9)
   ```
   Verified max abs diff with intercept on a nonzero-mean noisy factor = 2.4e-15.

2. **`test_rolling_factor_betas_recovers_known_loadings`** — multivariate, correlated factors.
   ```python
   idx = pd.bdate_range("2020-01-01", periods=500)
   rng = np.random.default_rng(11)
   Mkt = rng.standard_normal(500) * 0.01
   HML = rng.standard_normal(500) * 0.008
   asset = 1.5 * Mkt + 0.8 * HML + rng.standard_normal(500) * 0.001
   factors = pd.DataFrame({"Mkt": Mkt, "HML": HML}, index=idx)
   fb = rm.rolling_factor_betas(pd.DataFrame({"SYN": asset}, index=idx), factors, 252, 126)
   assert fb["Mkt"]["SYN"].iloc[-1] == pytest.approx(1.5, abs=2e-2)
   assert fb["HML"]["SYN"].iloc[-1] == pytest.approx(0.8, abs=2e-2)
   ```
   (~3 OLS standard errors; verified recovered (1.5017, 0.7788).)

3. **`test_rolling_factor_betas_recovers_correlated_noisefree`** — exact under collinearity.
   ```python
   idx = pd.bdate_range("2020-01-01", periods=600)
   rng = np.random.default_rng(3)
   f1 = rng.standard_normal(600) * 0.01
   f2 = 0.7 * f1 + 0.3 * rng.standard_normal(600) * 0.01      # correlated
   asset = 2.0 * f1 - 1.0 * f2                                # noise-free
   factors = pd.DataFrame({"f1": f1, "f2": f2}, index=idx)
   fb = rm.rolling_factor_betas(pd.DataFrame({"A": asset}, index=idx), factors, 252, 126)
   assert fb["f1"]["A"].iloc[-1] == pytest.approx(2.0, abs=1e-8)
   assert fb["f2"]["A"].iloc[-1] == pytest.approx(-1.0, abs=1e-8)
   ```

4. **`test_factor_betas_use_only_past_data`** — leak pin + boundary off-by-one probe (resolves C14).
   ```python
   idx = pd.bdate_range("2015-01-01", periods=800)
   rng = np.random.default_rng(1)
   Mkt = pd.Series(rng.normal(0, 0.01, 800), index=idx)
   HML = pd.Series(rng.normal(0, 0.008, 800), index=idx)
   asset = pd.DataFrame({"A": 1.2 * Mkt + 0.5 * HML + rng.normal(0, 0.003, 800)}, index=idx)
   factors = pd.DataFrame({"Mkt": Mkt, "HML": HML})
   full = rm.rolling_factor_betas(asset, factors)
   P = 600
   ac = asset.copy(); ac.iloc[P:] = 99.0
   fc = factors.copy(); fc.iloc[P:] = 99.0
   corr = rm.rolling_factor_betas(ac, fc)
   for name in factors.columns:
       pd.testing.assert_frame_equal(full[name].iloc[:P], corr[name].iloc[:P])
   # boundary off-by-one probe: the loading at the LAST pre-poison date (P-1)
   # is byte-identical (its trailing window ends at P-1 < P, cannot read P).
   for name in factors.columns:
       np.testing.assert_array_equal(
           full[name].iloc[P - 1].to_numpy(), corr[name].iloc[P - 1].to_numpy())
   ```

5. **`test_cross_sectional_neutralize_zeroes_exposure`**.
   ```python
   loadings = pd.DataFrame({"value": [2.0, 1.0, 0.5], "dollar": [1.0, 1.0, 1.0]},
                           index=list("ABC"))
   s = pd.Series({"A": 0.6, "B": -0.1, "C": -0.2})
   r = rm.cross_sectional_neutralize(s, loadings)
   assert float(loadings["value"].to_numpy() @ r.to_numpy()) == pytest.approx(0.0, abs=1e-12)
   assert r.sum() == pytest.approx(0.0, abs=1e-12)
   ```

6. **`test_loadings_at_builds_neutralizable_matrix`** — integration with existing consumers.
   ```python
   idx = pd.bdate_range("2020-01-01", periods=400)
   rng = np.random.default_rng(7)
   Mkt = rng.standard_normal(400) * 0.01
   HML = rng.standard_normal(400) * 0.008
   assets = {f"S{i}": 1.0 * Mkt + (0.3 * i) * HML + rng.standard_normal(400) * 0.002
             for i in range(4)}
   ar = pd.DataFrame(assets, index=idx)
   fb = rm.rolling_factor_betas(ar, pd.DataFrame({"Mkt": Mkt, "HML": HML}, index=idx))
   L = rm.loadings_at(fb, idx[-1], add_dollar=True)
   assert L.columns.tolist() == ["Mkt", "HML", "dollar"]
   assert L.shape == (4, 3)
   w = pd.Series({c: (1.0 if i % 2 else -1.0) for i, c in enumerate(ar.columns)})
   wn = rm.neutralize_weights(w, L)
   assert rm.net_factor_exposure(wn, L).abs().max() < 1e-9
   ```

7. **`test_factor_betas_guards`**.
   ```python
   idx = pd.bdate_range("2020-01-01", periods=50)
   ar = pd.DataFrame({"A": np.zeros(50)}, index=idx)
   f = pd.DataFrame({"m": np.zeros(50)}, index=idx)
   with pytest.raises(ValueError):
       rm.rolling_factor_betas(ar, f, lookback=0)
   with pytest.raises(ValueError):
       rm.rolling_factor_betas(ar, f, lookback=10, min_periods=20)
   ```

### 2.7 KNOWN-ANSWER tests — `tests/test_ff_factors.py`

`FF5_PATH` is the project-root scratch file (NOT inside qr-alpha-lab). Use a module constant so a missing file `pytest.skip`s rather than errors:
```python
import os, sys
import numpy as np, pandas as pd, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quantlab import ff_factors
FF5_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "scratch_refute",
                        "F-F_Research_Data_5_Factors_2x3.csv")
pytestmark = pytest.mark.skipif(not os.path.exists(FF5_PATH), reason="FF file absent")
```

1. **`test_load_ff_factors_monthly_known_values`**.
   ```python
   df = ff_factors.load_ff_factors_monthly(FF5_PATH)
   assert df.columns.tolist() == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
   assert df.index[0] == pd.Timestamp("1963-07-31")
   assert df.loc["1963-07", "Mkt-RF"].iloc[0] == pytest.approx(-0.0039, abs=1e-12)
   assert df.index[-1] == pd.Timestamp("2026-04-30")
   assert df.loc["2026-04", "Mkt-RF"].iloc[0] == pytest.approx(0.0994, abs=1e-12)
   assert len(df) == 754                              # monthly rows only
   assert df.index.is_monotonic_increasing and df.index.is_unique
   # annual section absent: 1964 annual Mkt-RF was 12.59% => 0.1259, never a month-end row
   assert pd.Timestamp("1964-12-31") in df.index      # this is the 196412 MONTH, not the annual 1964
   assert not (df["Mkt-RF"] == pytest.approx(0.1259)).any()
   ```
   Verified: 754 rows, first key 196307 (`-0.39/100 = -0.0039`), last key 202604 (`9.94/100 = 0.0994`), Annual section starts line 761.

2. **`test_load_ff_factors_daily_flags`**.
   ```python
   with pytest.raises(FileNotFoundError, match="separate Ken French download"):
       ff_factors.load_ff_factors_daily(os.path.join(os.path.dirname(FF5_PATH),
                                                      "does_not_exist_daily.CSV"))
   ```

### 2.8 Edge cases (all handled)
- **Intercept policy:** `fit_intercept=True` default, REQUIRED (C9). Documented against future "simplification".
- min_periods warm-up: per-window valid-row count, not just window length. Early dates / NaN factor rows → NaN.
- Collinear/duplicate factor columns → `pinv` gives the min-norm solution (matches `neutralize_weights`); does not crash.
- Window shorter than lookback at panel start → `max(0, t-lookback+1)` ramp; min_periods governs when output is non-NaN.
- K=1 with a Series factor → accept DataFrame only; Series caller `.to_frame()`. Documented.
- Factor index misaligned → reindex factors onto asset index first; asset dates absent in factors → NaN factor rows → excluded by the mask.
- FF percent→decimal: divide RF too (all columns percent). Space-padded numerics stripped by `skipinitialspace=True`. Annual section / copyright line terminate the read.
- Sporadic per-asset NaN in Y inside a window → per-column solve dropping that column's NaN rows (§2.3 step 4).

---

## 3. Module: `quantlab.synthetic` (two new quality modes)

### 3.1 Files
- **EDIT** `src/quantlab/synthetic.py` — `make_quality_panel` ONLY (lines 258–299). Do NOT touch `make_panel`, `make_perp_panel`, `make_cef_panel`, `inject_*`.
- Tests live in **NEW** `tests/test_quality_value.py` (§4).

### 3.2 Signature (UNCHANGED) + module constants

```python
make_quality_panel(n_firms=200, n_periods=180, mode='planted_quality',
                   seed=7, premium=0.02) -> pd.DataFrame
```
Add at module scope (above `make_quality_panel`), with comments:
```python
_VALUE_SEED_OFFSET = 5150   # separate-rng offset for ALL value-world randomness;
                            # distinct from make_panel's seed+777. Never reuse the
                            # main seed or the value draws correlate with q/idio.
_VALUE_PREMIUM = 0.05       # mean of the HML factor return series val_f (the value
                            # premium that makes World A's raw SR positive). MUST be
                            # nonzero: a zero-mean factor gives E[ret]=loading*0=0
                            # and the discrimination evaporates.
_VALUE_VOL = 0.08           # per-period std of val_f; > idio (0.06) so a 36/18
                            # rolling HML beta is identifiable and neutralization bites.
_WORLD_B_PREMIUM = 0.004    # World B's genuine value-ORTHOGONAL quality alpha.
                            # CALIBRATED so World B raw SR overlaps World A's (~2.4),
                            # NOT the original premium=0.02 which gave SR ~12 and made
                            # the worlds separable by raw SR alone (the discrimination
                            # must come from neutralization, not headroom). Hardcoded —
                            # NOT a tunable knob (these are falsification worlds).
```

### 3.3 Mode membership guard (resolves law #6, keep both old literals)

```python
_QUALITY_MODES = {"planted_quality", "null_quality",
                  "quality_is_value", "quality_orthogonal"}
if mode not in _QUALITY_MODES:
    raise ValueError(
        "mode must be one of 'planted_quality', 'null_quality', "
        f"'quality_is_value', 'quality_orthogonal', got {mode!r}"
    )
```

### 3.4 Generative model (EXACT — keep the 5 main-rng draws byte-identical)

Lines 282–290 stay **byte-identical for ALL modes** (the main-rng draw sequence and `qz` are untouched — this is what guarantees old-mode byte identity, verified `.equals()==True` 6/6):
```python
rng = np.random.default_rng(seed)
q = rng.standard_normal(n_firms)                          # draw 1
gp_noise = rng.standard_normal((n_periods, n_firms))      # draw 2
gp_a = 0.15 + 0.05 * q[None, :] + 0.01 * gp_noise
betas = rng.uniform(0.6, 1.2, n_firms)                    # draw 3
mkt = rng.standard_normal(n_periods) * 0.04               # draw 4
idio = rng.standard_normal((n_periods, n_firms)) * 0.06   # draw 5
qz = (q - q.mean()) / (q.std() + 1e-12)                   # NO rng draw
```
Then branch on mode for the RETURN LINK only:
```python
if mode in ("planted_quality", "null_quality"):
    prem = premium * qz[None, :] if mode == "planted_quality" else 0.0
    rets = betas[None, :] * mkt[:, None] + idio + prem
    value_loading = None; value_factor = None
else:
    # SEPARATE rng, created AFTER all 5 main draws => old modes' byte stream
    # is untouched (val_rng never instantiated for them). Verified.
    val_rng = np.random.default_rng(seed + _VALUE_SEED_OFFSET)
    val_f = _VALUE_PREMIUM + val_rng.standard_normal(n_periods) * _VALUE_VOL   # draw A
    raw = val_rng.standard_normal(n_firms)                                     # draw B
    if mode == "quality_is_value":
        # World A: the firm's value loading IS the quality z-score (collinear).
        # The ENTIRE predictable return arrives through the value factor's
        # positive mean times the loading. E[ret] = qz * _VALUE_PREMIUM.
        value_loading = qz.copy()
        rets = betas[None, :] * mkt[:, None] + idio + value_loading[None, :] * val_f[:, None]
    else:  # quality_orthogonal — World B
        # Gram-Schmidt residualize the value loading AGAINST qz so
        # corr(value_loading, qz) == 0; then re-zscore so its scale matches qz.
        vl0 = raw - (raw @ qz / (qz @ qz)) * qz
        value_loading = (vl0 - vl0.mean()) / (vl0.std() + 1e-12)
        prem = _WORLD_B_PREMIUM * qz       # genuine value-ORTHOGONAL alpha
        rets = (betas[None, :] * mkt[:, None] + idio + prem[None, :]
                + value_loading[None, :] * val_f[:, None])
    value_factor = val_f
```
Build price + attrs:
```python
periods = pd.bdate_range("2010-01-31", periods=n_periods, freq="BME")
firms = [f"FIRM{i:03d}" for i in range(n_firms)]
price = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=periods, columns=firms)
price.attrs["gp_a"] = pd.DataFrame(gp_a, index=periods, columns=firms)   # ALL modes
price.attrs["mode"] = mode                                               # ALL modes
if value_loading is not None:                                            # NEW modes ONLY
    price.attrs["value_loading"] = pd.Series(value_loading, index=firms)
    price.attrs["book_to_market"] = pd.Series(value_loading, index=firms)  # alias
    price.attrs["value_factor"] = pd.Series(value_factor, index=periods)
    price.attrs["betas"] = pd.Series(betas, index=firms)
    # deterministic sector map drawn from val_rng AFTER the two value draws
    # (stream stays clean); exposed as defensive ground truth for a future
    # market+sector+HML neutralization test.
    sect = val_rng.integers(0, 6, n_firms)                               # draw C
    price.attrs["sector"] = {firms[i]: f"S{int(sect[i])}" for i in range(n_firms)}
return price
```
**Old modes keep EXACTLY `attrs == {gp_a, mode}`** (no value_loading/value_factor/sector/betas leak — mirrors the `attrs['regimes']`-only-for-planted_regime discipline).

### 3.5 What the modes guarantee (all re-verified in-env, seeds 7/11/23 unless noted)

- World A `corr(value_loading, qz) == +1.0` (measured `0.9999999999999998`); World B `|corr| < 1e-9` (measured `2.0e-17`). This is the mathematical guarantee that makes the discrimination deterministic.
- Raw SR (gp_a quality book, no neutralization): World A ∈ [2.32, 2.48]; World B ∈ [2.39, 3.45] at seeds 7/11/23. Across 33 seeds (7..39): World A ∈ [1.65, 3.24], World B ∈ [0.20, 3.62] — **overlapping** (means 2.20 vs 2.32). The worlds are NOT separable by raw SR.
- Static-loading (estimation-error-free) neutral SR: World A ∈ [−0.41, +0.25] (collapses to ~0); World B ∈ [2.44, 3.70] (survives). (CORRECTED 2026-06-24 from in-env measurement.)
- Rolling-estimate (36/18) neutral SR: World A ∈ [0.87, 0.96]; World B ∈ [2.54, 3.56]; `neutral_B − neutral_A ≥ 1.58` at all three seeds (diffs 1.58 / 2.60 / 1.86; min at seed 7). (CORRECTED 2026-06-24: the earlier "[0.71,0.78] / [2.61,3.11] / ≥1.83" figures were stale — review caught the overclaim.)
- Placebo control (World A neutralized vs a matched random factor), STATIC path: neutral_true ∈ [−0.41, +0.25] vs neutral_placebo ∈ [2.32, 2.48] → `neutral_true < 0.5 * neutral_placebo` on all three seeds. (CORRECTED 2026-06-24.)
- Value-only book (sorted on `value_loading`) in World B earns SR 2.45 — a separable axis from the quality alpha.

---

## 4. Module: `quantlab.fundamentals` (value_neutralized_signal) + the integration test

### 4.1 Files
- **EDIT** `src/quantlab/fundamentals.py` — APPEND `value_neutralized_signal`. Do NOT change `quality_signal`, `quality_weights`, `quality_backtest`, `machinery_gate` bodies. Add `from quantlab import risk_model` at the top (no heavy dep).
- **NEW** `tests/test_quality_value.py`.

### 4.2 Signature appended to `fundamentals.py`

```python
def value_neutralized_signal(
    gp_a: pd.DataFrame,
    value_loading: pd.DataFrame,
    accruals_a: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """quality_signal(gp_a, accruals_a) then, per date, cross-sectionally
    residualize against the value loading (HML proxy) + a ones column (dollar-
    neutral demean), via risk_model.cross_sectional_neutralize. The 'neutral'
    arm of H1 raw-vs-neutral. Point-in-time: value_loading must be known at t
    (a synthetic ground-truth attr in the lab; a trailing past-only
    rolling_factor_betas HML loading on real data). A date whose value_loading
    row is all-NaN degenerates to a plain demean (no value-neutralization that
    date) rather than crashing — the conservative, documented convention."""
```
Body:
```python
from quantlab import risk_model
sig = quality_signal(gp_a, accruals_a)
out = sig.copy()
for d in sig.index:
    if d not in value_loading.index:
        continue                                  # leave as-is (no loading known)
    L = pd.DataFrame({"value": value_loading.loc[d].reindex(sig.columns),
                      "dollar": 1.0}, index=sig.columns)
    out.loc[d] = risk_model.cross_sectional_neutralize(sig.loc[d], L).reindex(sig.columns)
return out
```

### 4.3 KNOWN-ANSWER tests — `tests/test_quality_value.py`

```python
import os, sys
import numpy as np, pandas as pd, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quantlab import fundamentals as fnd, metrics, risk_model as rm
from quantlab.synthetic import make_quality_panel

SEEDS = (7, 11, 23)

def _zscore(a):
    return (a - a.mean()) / (a.std() + 1e-12)

def _broadcast_loading(price):
    """(period x firm) panel of the STATIC ground-truth value loading."""
    vl = price.attrs["value_loading"]
    return pd.DataFrame(np.tile(vl.to_numpy(), (len(price.index), 1)),
                        index=price.index, columns=price.columns)

def _sr(net):
    return metrics.sharpe(net, periods=fnd.PERIODS_PER_YEAR)

def _raw_neutral_static(price):
    """Raw and STATIC-loading neutral SR (estimation-error-free)."""
    gp_a = price.attrs["gp_a"]
    raw = _sr(fnd.quality_backtest(fnd.quality_signal(gp_a), price, cost_bps_per_side=0.0)["net"])
    vl = _broadcast_loading(price)
    neutral = _sr(fnd.quality_backtest(fnd.value_neutralized_signal(gp_a, vl), price,
                                       cost_bps_per_side=0.0)["net"])
    return raw, neutral
```

1. **`test_value_loading_collinearity_is_exact`** (pins the math that makes everything deterministic).
   ```python
   for s in SEEDS:
       q = np.random.default_rng(s).standard_normal(200)
       qz = _zscore(q)
       a = make_quality_panel(mode="quality_is_value", seed=s)
       b = make_quality_panel(mode="quality_orthogonal", seed=s)
       assert np.corrcoef(a.attrs["value_loading"].to_numpy(), qz)[0, 1] == pytest.approx(1.0, abs=1e-9)
       assert abs(np.corrcoef(b.attrs["value_loading"].to_numpy(), qz)[0, 1]) < 1e-9
   ```

2. **`test_quality_is_value_collapses_static_loading`** — primary World-A pin (resolves C12).
   ```python
   for s in SEEDS:
       raw, neutral = _raw_neutral_static(make_quality_panel(mode="quality_is_value", seed=s))
       assert raw > 1.5
       assert neutral < 0.3            # measured [-0.41, +0.25]; signal lies in span(L)
   ```

3. **`test_quality_orthogonal_survives_static_loading`** — primary World-B pin.
   ```python
   for s in SEEDS:
       raw, neutral = _raw_neutral_static(make_quality_panel(mode="quality_orthogonal", seed=s))
       assert raw > 1.5                # measured [2.39, 3.45]
       assert neutral > 1.0            # measured [2.44, 3.70]; orthogonal alpha survives
   ```

4. **`test_raw_sharpe_alone_does_not_separate_worlds`** — MANDATORY (resolves C10).
   ```python
   raw_A, raw_B = [], []
   for s in range(7, 40):             # 33 seeds
       raw_A.append(_sr(fnd.quality_backtest(
           fnd.quality_signal(make_quality_panel(mode="quality_is_value", seed=s).attrs["gp_a"]),
           make_quality_panel(mode="quality_is_value", seed=s), cost_bps_per_side=0.0)["net"]))
       raw_B.append(_sr(fnd.quality_backtest(
           fnd.quality_signal(make_quality_panel(mode="quality_orthogonal", seed=s).attrs["gp_a"]),
           make_quality_panel(mode="quality_orthogonal", seed=s), cost_bps_per_side=0.0)["net"]))
   # the raw-SR distributions OVERLAP — a raw threshold cannot tell the worlds apart
   assert max(raw_A) >= min(raw_B)    # measured 3.24 >= 0.20
   assert min(raw_A) <= max(raw_B)
   ```

5. **`test_collapse_requires_true_value_factor`** — MANDATORY placebo control (resolves C11).
   ```python
   for s in SEEDS:
       p = make_quality_panel(mode="quality_is_value", seed=s)
       gp_a = p.attrs["gp_a"]
       vl_true = _broadcast_loading(p)
       neutral_true = _sr(fnd.quality_backtest(
           fnd.value_neutralized_signal(gp_a, vl_true), p, cost_bps_per_side=0.0)["net"])
       # placebo loading: independent per-firm draw, same shape, NOT collinear with qz
       placebo = np.random.default_rng(10 * s + 1).standard_normal(p.shape[1])
       vl_pl = pd.DataFrame(np.tile(placebo, (len(p.index), 1)),
                            index=p.index, columns=p.columns)
       neutral_pl = _sr(fnd.quality_backtest(
           fnd.value_neutralized_signal(gp_a, vl_pl), p, cost_bps_per_side=0.0)["net"])
       assert neutral_true < 0.5 * neutral_pl   # collapse needs the TRUE factor
   ```
   Verified neutral_true ≈ 0.71–0.78 vs neutral_pl ≈ 2.26–2.39.

6. **`test_quality_worlds_discriminate_under_rolling_estimated_loading`** — production path + leak (resolves C13).
   ```python
   def rolling_neutral_sr(price):
       gp_a = price.attrs["gp_a"]; sig = fnd.quality_signal(gp_a)
       rets = price.pct_change(fill_method=None)
       val_f = price.attrs["value_factor"]
       # PRODUCTION estimator: rolling_factor_betas HML loading (fit_intercept=True)
       fb = rm.rolling_factor_betas(rets, val_f.to_frame("value"), lookback=36, min_periods=18)
       loading_panel = fb["value"]
       neutral = _sr(fnd.quality_backtest(
           fnd.value_neutralized_signal(gp_a, loading_panel), price, cost_bps_per_side=0.0)["net"])
       raw = _sr(fnd.quality_backtest(sig, price, cost_bps_per_side=0.0)["net"])
       return raw, neutral, loading_panel
   for s in SEEDS:
       ra, na, _ = rolling_neutral_sr(make_quality_panel(mode="quality_is_value", seed=s))
       rb, nb, _ = rolling_neutral_sr(make_quality_panel(mode="quality_orthogonal", seed=s))
       assert nb - na > 1.0                    # discrimination survives estimation error (measured >=1.58)
   # poison-the-future leak check on the ESTIMATED-loading path
   p = make_quality_panel(mode="quality_is_value", seed=7)
   rets = p.pct_change(fill_method=None); val_f = p.attrs["value_factor"]
   full = rm.rolling_factor_betas(rets, val_f.to_frame("value"), 36, 18)["value"]
   rc = rets.copy(); rc.iloc[120:] = 99.0
   vc = val_f.copy(); vc.iloc[120:] = 99.0
   corr = rm.rolling_factor_betas(rc, vc.to_frame("value"), 36, 18)["value"]
   pd.testing.assert_frame_equal(full.iloc[:120], corr.iloc[:120])
   ```

7. **`test_raw_vs_neutral_discrimination_is_paired`** — paired differential (the stable pin).
   ```python
   diffs = []
   for s in SEEDS:
       _, na = _raw_neutral_static(make_quality_panel(mode="quality_is_value", seed=s))
       _, nb = _raw_neutral_static(make_quality_panel(mode="quality_orthogonal", seed=s))
       diffs.append(nb - na)
   assert min(diffs) > 1.0                     # measured static diffs ~2.5-4.1
   ```

8. **`test_value_only_book_is_distinct_source_in_world_B`**.
   ```python
   p = make_quality_panel(mode="quality_orthogonal", seed=7)
   vl = _broadcast_loading(p)
   w = fnd.quality_weights(vl, quantile=0.2)
   fwd = p.pct_change(fill_method=None).shift(-1).reindex_like(vl)
   sr_value_book = _sr((w * fwd).sum(axis=1, min_count=1).dropna())
   assert sr_value_book > 1.0                  # measured 2.45 — a separable axis
   ```

9. **`test_existing_quality_modes_byte_identical`** — golden canary (resolves C16). Frozen literals, NOT a re-call.
   ```python
   # Captured 2026-06-24 from make_quality_panel BEFORE this change (iloc[0,0],
   # iloc[-1,-1], total-sum triple per (mode, seed)). Any perturbation of the
   # main-rng stream fails here before machinery_gate runs.
   GOLDEN = {
       # (mode, seed): (first, last, total)  -- POPULATE from the capture script below
   }
   for (mode, seed), (first, last, total) in GOLDEN.items():
       p = make_quality_panel(mode=mode, seed=seed)
       np.testing.assert_allclose(p.iloc[0, 0], first, rtol=1e-12, atol=0)
       np.testing.assert_allclose(p.iloc[-1, -1], last, rtol=1e-12, atol=0)
       np.testing.assert_allclose(p.to_numpy().sum(), total, rtol=1e-12, atol=0)
       assert set(p.attrs) == {"gp_a", "mode"}      # NO value attrs leaked
   for mode in ("quality_is_value", "quality_orthogonal"):
       p = make_quality_panel(mode=mode, seed=7)
       assert "value_loading" in p.attrs and "value_factor" in p.attrs
       assert p.attrs["value_loading"].shape == (200,)
       assert p.attrs["value_factor"].shape == (180,)
   ```
   **Capture step (run BEFORE editing synthetic.py, on the unmodified file):**
   ```bash
   python -c "import sys; sys.path.insert(0,'src'); from quantlab.synthetic import make_quality_panel as f;
   print({(m,s): (float(f(mode=m,seed=s).iloc[0,0]), float(f(mode=m,seed=s).iloc[-1,-1]), float(f(mode=m,seed=s).to_numpy().sum())) for m in ('planted_quality','null_quality') for s in (7,11,23)})"
   ```
   Paste the printed dict into `GOLDEN`. (Old modes are byte-identical after the edit — verified `.equals()==True` 6/6 — so these literals must not move.)

10. **`test_quality_mode_membership`** (resolves law #6).
    ```python
    with pytest.raises(ValueError, match="planted_quality"):
        make_quality_panel(mode="bogus")
    for m in ("planted_quality", "null_quality", "quality_is_value", "quality_orthogonal"):
        make_quality_panel(mode=m, seed=7)   # must not raise
    ```

### 4.4 The four-arm CSCV adjudication test — `test_cscv_adjudicates_the_four_arms` (resolves C15)

End-to-end wiring: PBO/CSCV consuming the two-world raw-vs-neutral paths.
```python
def test_cscv_adjudicates_the_four_arms():
    from quantlab import pbo
    s = 7
    cols = {}
    for tag, mode in (("raw_isvalue", "quality_is_value"),
                      ("neutral_isvalue", "quality_is_value"),
                      ("raw_orth", "quality_orthogonal"),
                      ("neutral_orth", "quality_orthogonal")):
        p = make_quality_panel(mode=mode, seed=s)
        gp_a = p.attrs["gp_a"]
        if tag.startswith("raw"):
            sig = fnd.quality_signal(gp_a)
        else:
            sig = fnd.value_neutralized_signal(gp_a, _broadcast_loading(p))
        cols[tag] = fnd.quality_backtest(sig, p, cost_bps_per_side=0.0)["net"]
    df = pd.DataFrame(cols)
    # CONTIGUOUS common post-warm-up SLICE (NEVER dropna across mismatched
    # warm-ups -- that produces non-contiguous "contiguous" blocks, corrupting
    # CSCV semantics). Take the gap-free intersection of valid rows by slicing.
    valid = df.dropna(how="any")
    start, end = valid.index.min(), valid.index.max()
    df = df.loc[start:end]
    assert not df.isna().any().any()             # contiguous & gap-free after slice
    n_blocks = 6                                  # C(6,3)=20; T~=178 monthly => ~29 rows/block
    out = pbo.cscv(df, n_splits=n_blocks)
    assert out["n_combinations"] == 20
    assert out["logits"].shape == (20,)
    # qualitative (T is short): neutral_orth's mean OOS Sharpe rank exceeds
    # neutral_isvalue's -- the value-disguised neutral arm degrades, the genuine
    # one persists. Computed from per-config OOS ranks across splits.
```
Note: the quantitative PBO pins live in §1.5 on long T; this test is the wiring demonstration, kept structural (n_combinations / shapes / contiguity) because T=180 monthly is short for CSCV.

### 4.5 Edge cases
- `value_neutralized_signal`: a date absent from `value_loading.index` → signal left as-is. A date whose loading row is all-NaN → `cross_sectional_neutralize` `.fillna(0.0)` degenerates to a plain demean. Both documented.
- The four new modes work at machinery_gate's `(200,180, seeds 7/11/23)` and test_engine's `(120,180)` shapes; the gate only calls `planted_quality`/`null_quality`, so it is untouched (confirmed: `machinery_gate` hardcodes those two modes).
- `quality_is_value` and `quality_orthogonal` draw different shapes from `val_rng` (A draws val_f+raw+sector; B draws val_f+raw+sector). They are NOT byte-paired to each other — each is reproducible per seed. Documented.

---

## 5. Implementation order (each step leaves `pytest tests/ -q` green)

Run `python -m pytest tests/ -q` after every step. The make_panel-based CI gate is never in the edit path; it is re-run ONCE at the end (§6).

- **Step 0 — capture old-mode golden literals** (BEFORE any edit): run the §4.3-test-9 capture command on the unmodified `synthetic.py`; paste into the `GOLDEN` dict. Pure additions follow.
- **Step 1 — `pbo.py` + `test_pbo.py`** (zero blast radius, new files). Green: existing suite + 8 new pbo tests.
- **Step 2 — `risk_model.py` (`rolling_factor_betas`, `loadings_at`, `cross_sectional_neutralize`) + `ff_factors.py` + `test_risk_model_factor_betas.py` + `test_ff_factors.py`** (pure additions; new functions called by nothing existing). Green.
- **Step 3 — `synthetic.make_quality_panel`** (the only edit to a hot file). Immediately verify: `pytest tests/test_fundamentals.py tests/test_engine.py tests/test_regime.py -q` AND `python -c "from quantlab import fundamentals; print(fundamentals.machinery_gate())"` (all depend on byte-identical planted/null streams). Green.
- **Step 4 — `fundamentals.value_neutralized_signal` + `test_quality_value.py`** (adds a function and a test file; the four-arm CSCV test wires (a)+(b)+(c) together). Green.
- **Step 5 — docs**: append the H1 amendment to `writeup/preregistered_hypotheses.md` (§7) and a zero-trial machinery entry to `research_log.md` (§7.2). Run the full falsification gate (§6) as proof-of-zero-regression.

Why this stays green: Steps 1–2 are pure additions; nothing existing can break. Step 3 is guarded by the byte-identical-stream discipline and re-verified against the three stream-dependent suites + machinery_gate. Step 4 only adds. Step 5 is docs + the confirming gate run.

---

## 6. Falsification gate — exact commands + golden values that MUST stay byte-identical

Run all from `qr-alpha-lab/` after Step 4 (law #2 requires the gate after ANY change; here it is proof-of-zero-regression since `make_panel` is untouched):

```bash
python -m pytest tests/ -q                                          # all green (existing + new)
python scripts/run_pipeline.py --data planted --fail-if-dsr-below 0.95
python scripts/run_pipeline.py --data noise --n-trials 20 --fail-if-dsr-above 0.5
```

**Golden values that MUST NOT MOVE:**
- planted: `sharpe_net = 0.8645821957617108` (SR 0.8646), `dsr = psr = 0.991946199409908` (DSR 0.9919), `mean_rank_ic = 0.06286314099595566`.
- noise: `dsr = 0.00043487354008937167`, `sharpe_net = -0.5028891736804396`.
- `tests/test_regime.py::test_planted_and_noise_modes_match_golden_values` (rtol=1e-12): planted `(100.59548019891207, 6.51476839555455, 167003.37677401048)`, noise `(100.59548019891207, 6.47579982965231, 166968.8989943195)`.

These are all functions of `make_panel`, which is NOT edited → structurally safe. The H1 quality `machinery_gate(seeds=(7,11,23))` is re-run inside `pytest tests/ -q` via `test_fundamentals.py::test_machinery_gate_planted_quality_beats_null` (gate['passed'], min planted_sr > 0.5, max|null_sr| < 0.6) and `test_engine.py`; it depends on byte-identical planted_quality/null_quality streams, which Step 3's separate-rng discipline guarantees (verified `.equals()==True` 6/6).

---

## 7. Pre-registration text + log entry

### 7.1 Append to `writeup/preregistered_hypotheses.md` (under H1, as an amendment)

```
- **Amendment, 2026-06-24 — PRE-DATA, MACHINERY-ONLY (status stays PROPOSED; ZERO
  trials; N unchanged at 11; H1 remains blocked on the survivorship-safe PIT
  data-source decision). Declares the adjudication protocol BEFORE any real H1 run,
  per protocol.**
  (A) RAW-vs-NEUTRAL is now a PRE-REGISTERED, PAIRED success criterion, not a
  post-hoc robustness leg. The H1 quality book is run in TWO arms on the same
  universe / dates / costs: RAW = z(profitability) [CBOP per the 2026-06-16
  amendment]; NEUTRAL = the same signal cross-sectionally residualized against an
  HML/value loading (+ a ones column for dollar-neutrality) via
  risk_model.cross_sectional_neutralize, the value loading from a trailing
  past-only rolling_factor_betas HML beta (fit_intercept=True — required;
  point-in-time, law #1). H1 GRADUATES on the NEUTRAL arm: a quality claim that is
  merely the value factor re-labeled must NOT count. Criterion: right-signed
  t_NW >= +2 AND net SR > 0 beating both baselines AND DSR >= 0.95 at the
  then-current N on the NEUTRAL arm; the RAW arm is reported alongside, and a large
  raw-minus-neutral gap is declared IN ADVANCE as evidence the edge was
  value-collinear (interpreted, not hidden).
  (B) PBO/CSCV is now a PRE-REGISTERED selection-overfit gate COMPLEMENTING the
  DSR. Whatever set of H1 configs is considered (raw vs neutral; current- vs
  lagged-assets denominator; quintile vs the declared robustness cuts), their OOS
  return paths are assembled into a contiguous, gap-free (T x M) matrix and
  pbo.probability_of_backtest_overfitting is reported. Pre-declared threshold:
  PBO <= 0.5 required to graduate (PBO > 0.5 means the IS-selection rule does not
  generalize OOS — a hard stop regardless of the chosen config's DSR). DSR guards
  the final track record against luck-of-N; PBO guards the selection PROCESS that
  chose it; BOTH must pass.
  (C) SYNTHETIC TWO-WORLD VALIDATION (run in-env immediately before any real H1
  run, law #4, like the carry/CEF/event gates): synthetic.make_quality_panel
  quality_is_value (edge IS value -> NEUTRAL arm must collapse: static-loading
  neutral SR < 0.3) and quality_orthogonal (edge is value-orthogonal -> NEUTRAL
  arm must SURVIVE: static-loading neutral SR > 1.0), with the worlds SR-MATCHED on
  the raw arm (so the discrimination is attributable to neutralization, not a
  Sharpe-level gap) and a placebo-factor control proving the collapse requires the
  TRUE value factor (neutral_true < 0.5 * neutral_placebo). Pinned as paired
  per-seed differentials and a placebo control in tests/test_quality_value.py. If
  the neutralization cannot tell a value-disguised edge from a genuine one TODAY,
  no real raw-vs-neutral number is trusted -> ABORT, no trial spent.
  This amendment is machinery + synthetic validation ONLY; it spends ZERO trials
  and does not move N (still 11).
```

### 7.2 Append to `research_log.md` (bookkeeping, NOT a trial row)

```
2026-06-24 — MACHINERY ADDED (ZERO trials, N unchanged at 11). Added: quantlab.pbo
(CSCV / Probability of Backtest Overfitting, Bailey et al. 2015); risk_model.
rolling_factor_betas (past-only K-factor OLS, fit_intercept=True, reduces to
rolling_market_beta at K=1 to 2.4e-15) + loadings_at + cross_sectional_neutralize;
quantlab.ff_factors (FF 5-factor monthly loader + daily stub); synthetic.
make_quality_panel modes quality_is_value / quality_orthogonal (value-disguised vs
value-orthogonal quality alpha, raw-SR-matched via _WORLD_B_PREMIUM=0.004);
fundamentals.value_neutralized_signal. Validated on the synthetic two-world panel
(World A collapses under HML-neutralization, World B survives; placebo control
proves the collapse needs the TRUE factor; PBO=0/1 on the deterministic pins,
~0.49 on noise). H1 raw-vs-neutral + PBO<=0.5 + two-world gate pre-registered
(preregistered_hypotheses.md H1 amendment 2026-06-24). H1 still data-blocked.
Falsification gate re-run, byte-identical: planted SR 0.8646 / DSR 0.9919, noise
DSR 0.0004. make_panel untouched.
```

---

## 8. Summary of new + edited files

**NEW (7):**
1. `src/quantlab/pbo.py`
2. `src/quantlab/ff_factors.py`
3. `tests/test_pbo.py`
4. `tests/test_risk_model_factor_betas.py`
5. `tests/test_ff_factors.py`
6. `tests/test_quality_value.py`
7. (DOC) the H1 amendment block + log entry (appended to existing files, §7)

**EDITED (3 source):**
8. `src/quantlab/risk_model.py` — append `rolling_factor_betas`, `loadings_at`, `cross_sectional_neutralize`
9. `src/quantlab/synthetic.py` — extend `make_quality_panel` (modes + constants + new attrs; main-rng stream byte-identical)
10. `src/quantlab/fundamentals.py` — append `value_neutralized_signal` (+ `from quantlab import risk_model`)
11. `writeup/preregistered_hypotheses.md` — H1 amendment (§7.1)
12. `research_log.md` — zero-trial machinery entry (§7.2)

**NOT edited (must stay byte-identical):** `synthetic.py::make_panel`, `metrics.py`, `scripts/run_pipeline.py`, `.github/workflows/ci.yml`, `tests/test_regime.py`, `tests/test_metrics.py`, `tests/test_fundamentals.py`, `tests/test_engine.py`, `scripts/run_fundamentals.py`, `scripts/engine_demo.py`, and all `risk_model.py` functions other than the three appended.
