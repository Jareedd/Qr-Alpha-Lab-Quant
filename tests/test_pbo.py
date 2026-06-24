"""Known-answer tests for CSCV PBO. The two anchoring properties:
  * a family of pure-noise configs -> PBO ~ 0.5 (selection is a coin flip);
  * one genuinely-skilled config among noise -> PBO ~ 0 (the winner stays a winner).
Seeded synthetic data; small n_splits keeps the combinatorics fast.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab.pbo import cscv_pbo


def _mean_pbo(make_matrix, seeds, n_splits=10):
    return float(np.mean([cscv_pbo(make_matrix(s), n_splits=n_splits)["pbo"] for s in seeds]))


def test_noise_family_pbo_is_high():
    # Selecting the in-sample best among indistinguishable noise configs does NOT
    # generalize: PBO is high (theoretical mean 0.5; single runs are noisy because
    # CSCV's combinations reuse blocks, so average over seeds and assert a floor).
    noise = _mean_pbo(lambda s: pd.DataFrame(np.random.default_rng(s).standard_normal((2400, 8))),
                      seeds=range(8))
    assert noise > 0.35
    one = cscv_pbo(pd.DataFrame(np.random.default_rng(0).standard_normal((2400, 8))), n_splits=10)
    assert one["n_configs"] == 8 and one["n_combinations"] == 252


def test_one_skilled_config_drives_pbo_low():
    rng = np.random.default_rng(1)
    R = pd.DataFrame(rng.standard_normal((2400, 8)) * 0.02)
    R[7] = R[7] + 0.01                                 # one config has a real edge
    out = cscv_pbo(R, n_splits=10)
    assert out["pbo"] < 0.10                           # the IS winner stays the OOS winner
    assert out["prob_oos_loss"] < 0.10                 # and it rarely loses OOS


def test_skill_separates_from_noise():
    # the defensible known-answer: noise-selection PBO sits far above
    # skilled-selection PBO (the metric distinguishes a real edge from a coin flip).
    def skilled(s):
        m = pd.DataFrame(np.random.default_rng(s).standard_normal((2000, 6)) * 0.02)
        m[0] = m[0] + 0.012
        return m
    noise_mean = _mean_pbo(lambda s: pd.DataFrame(np.random.default_rng(s).standard_normal((2000, 6))),
                           seeds=range(6))
    skilled_mean = _mean_pbo(skilled, seeds=range(6))
    assert skilled_mean < 0.15
    assert noise_mean - skilled_mean > 0.30


def test_validation_guards():
    R = pd.DataFrame(np.zeros((100, 3)))
    with pytest.raises(ValueError):
        cscv_pbo(R, n_splits=9)                         # odd -> not symmetric
    with pytest.raises(ValueError):
        cscv_pbo(pd.DataFrame(np.zeros((100, 1))), n_splits=10)  # need >= 2 configs
    with pytest.raises(ValueError):
        cscv_pbo(pd.DataFrame(np.zeros((8, 3))), n_splits=10)    # n_splits > T


def test_flat_column_never_selected_in_sample():
    # a zero-variance column has Sharpe -inf in-sample, so it is never the IS pick;
    # the result must still be well-defined.
    rng = np.random.default_rng(3)
    R = pd.DataFrame(rng.standard_normal((1200, 4)))
    R[2] = 0.0
    out = cscv_pbo(R, n_splits=10)
    assert 0.0 <= out["pbo"] <= 1.0 and np.isfinite(out["median_logit"])


# --- Deterministic arithmetic pins -------------------------------------------
# The statistical tests above anchor the *behaviour* (noise -> high, skill ->
# low). These two pin the exact rank/omega/logit arithmetic on hand-built
# matrices with a known closed-form answer, so a future refactor of the rank,
# the (N+1) denominator, or the lambda<0 convention is caught immediately.
# Both values were re-derived in-env before being frozen.


def test_pbo_deterministic_persistent_edge_is_zero():
    # Config A is positive every period, B negative every period -> A is the
    # IS best AND the OOS best in every one of the C(4,2)=6 symmetric splits.
    # OOS rank of the winner among 2 configs = 2 -> omega = 2/3 -> lambda =
    # ln(2) > 0 for all splits -> PBO = 0 exactly.
    A = np.array([0.03, 0.02, 0.04, 0.03, 0.02, 0.05, 0.03, 0.04])
    B = np.array([-0.02, -0.03, -0.01, -0.02, -0.04, -0.01, -0.03, -0.02])
    out = cscv_pbo(pd.DataFrame(np.column_stack([A, B])), n_splits=4)
    assert out["pbo"] == 0.0
    assert out["prob_oos_loss"] == 0.0
    assert out["n_combinations"] == 6 and out["n_configs"] == 2
    assert out["median_logit"] == pytest.approx(np.log(2), abs=1e-12)


def test_pbo_deterministic_overfit_is_one():
    # B's per-BLOCK means negate A's (A blocks [4,1,-2,-3], B blocks
    # [-4,-1,2,3]); B is NOT the elementwise negation of A (the within-block
    # wiggle keeps stds equal while flipping every block mean). Because the
    # block means sum to zero, whichever config is IS-best is OOS-worst in all
    # 6 splits -> omega = 1/3 -> lambda = -ln(2) < 0 -> PBO = 1 exactly.
    A = np.array([4.5, 3.5, 1.5, 0.5, -1.5, -2.5, -2.5, -3.5])
    B = np.array([-3.5, -4.5, -0.5, -1.5, 2.5, 1.5, 3.5, 2.5])
    assert not np.allclose(B, -A)  # guard: NOT elementwise negation
    out = cscv_pbo(pd.DataFrame(np.column_stack([A, B])), n_splits=4)
    assert out["pbo"] == 1.0
    assert out["prob_oos_loss"] == 1.0
    assert out["median_logit"] == pytest.approx(-np.log(2), abs=1e-12)
    assert out["perf_degradation_slope"] < 0  # IS Sharpe anti-predicts OOS
