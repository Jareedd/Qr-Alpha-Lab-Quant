import os
import sys
import math

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import pbo, metrics


def test_pbo_deterministic_persistent_edge_is_zero():
    # Config A always positive, B always negative -> A is IS-best AND OOS-best in
    # every split -> best OOS rank -> w = N/(N+1) = 2/3 -> lambda = +ln2 always.
    A = np.array([0.03, 0.02, 0.04, 0.03, 0.02, 0.05, 0.03, 0.04])
    B = np.array([-0.02, -0.03, -0.01, -0.02, -0.04, -0.01, -0.03, -0.02])
    M = np.column_stack([A, B])
    r = pbo.cscv(M, n_splits=4)
    assert r["pbo"] == 0.0
    assert np.allclose(r["logits"], np.log(2), rtol=1e-12, atol=0.0)
    assert r["n_combinations"] == 6
    assert r["prob_oos_loss"] == 0.0


def test_pbo_deterministic_overfit_is_one():
    # B's per-BLOCK MEANS negate A's block means: A blocks [4,1,-2,-3],
    # B blocks [-4,-1,2,3]. B is NOT the elementwise negation of A; the
    # within-block wiggle keeps stds equal while flipping every block mean.
    # sum(A blocks)=0 => IS-best is OOS-worst in all 6 splits.
    A = np.array([4.5, 3.5, 1.5, 0.5, -1.5, -2.5, -2.5, -3.5])
    B = np.array([-3.5, -4.5, -0.5, -1.5, 2.5, 1.5, 3.5, 2.5])
    assert not np.allclose(B, -A)  # GUARD: B is NOT elementwise -A
    M = np.column_stack([A, B])
    r = pbo.cscv(M, n_splits=4)
    assert r["pbo"] == 1.0
    assert np.allclose(r["logits"], -np.log(2), rtol=1e-12, atol=0.0)
    assert r["prob_oos_loss"] == 1.0


def test_pbo_noise_is_about_half():
    means = []
    for s in range(20):
        M = np.random.default_rng(s).standard_normal((640, 20)) * 0.01
        means.append(pbo.cscv(M, n_splits=16)["pbo"])
    mean_pbo = float(np.mean(means))
    assert 0.40 < mean_pbo < 0.60


def test_pbo_one_dominant_config_is_zero():
    M = np.random.default_rng(321).standard_normal((640, 20)) * 0.01
    M[:, 0] += 0.004  # a genuine, time-stable edge
    assert pbo.cscv(M, n_splits=16)["pbo"] < 0.05


def test_degradation_fields():
    A = np.array([4.5, 3.5, 1.5, 0.5, -1.5, -2.5, -2.5, -3.5])
    B = np.array([-3.5, -4.5, -0.5, -1.5, 2.5, 1.5, 3.5, 2.5])
    M = np.column_stack([A, B])
    deg = pbo.performance_degradation(M, n_splits=4)
    assert set(deg) == {"slope", "intercept", "r_squared"}
    assert deg["slope"] < 0
    # value pin (not a vacuous [0,1] bound): an ss_res/ss_tot swap bug yields
    # a negative r_squared and would be caught here.
    assert deg["r_squared"] == pytest.approx(0.3055, abs=1e-3)


def test_pbo_shapes_and_counts():
    M = np.random.default_rng(1).standard_normal((640, 20)) * 0.01
    r = pbo.cscv(M, n_splits=8)
    assert r["n_splits"] == 8
    assert r["n_combinations"] == math.comb(8, 4) == 70
    assert r["logits"].shape == (70,)
    assert r["is_sharpe"].shape == (70,) == r["oos_sharpe"].shape


def test_column_sharpes_matches_metrics_up_to_annualization():
    rng = np.random.default_rng(5)
    df = pd.DataFrame(rng.standard_normal((300, 3)) * 0.01, columns=list("ABC"))
    fast = pbo._column_sharpes(df.to_numpy())
    ref = np.array([metrics.sharpe(df[c], periods=1) for c in df.columns])
    assert np.allclose(fast, ref, atol=1e-12)


def test_pbo_guards():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="even"):
        pbo.cscv(rng.standard_normal((100, 5)), n_splits=15)
    with pytest.raises(ValueError, match=">= 2 configs"):
        pbo.cscv(rng.standard_normal((100, 1)), n_splits=4)
    with pytest.raises(ValueError, match="non-finite"):
        bad = rng.standard_normal((100, 5))
        bad[10, 2] = np.nan
        pbo.cscv(bad, n_splits=4)
    with pytest.raises(ValueError, match="T >= n_splits"):
        pbo.cscv(rng.standard_normal((3, 5)), n_splits=4)
