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
