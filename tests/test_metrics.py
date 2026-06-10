import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import metrics


def test_sharpe_known_value():
    rng = np.random.default_rng(0)
    # mean = 0.001/day, std = 0.01/day -> SR = 0.1 * sqrt(252) ~ 1.587
    r = pd.Series(rng.normal(0.001, 0.01, 100_000))
    assert abs(metrics.sharpe(r) - 0.1 * np.sqrt(252)) < 0.1


def test_psr_high_for_strong_signal_low_for_noise():
    rng = np.random.default_rng(1)
    strong = pd.Series(rng.normal(0.002, 0.01, 2000))
    noise = pd.Series(rng.normal(0.0, 0.01, 2000))
    assert metrics.probabilistic_sharpe_ratio(strong) > 0.99
    assert metrics.probabilistic_sharpe_ratio(noise) < 0.95


def test_dsr_decreases_with_more_trials():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.0006, 0.01, 1500))
    dsr_1 = metrics.deflated_sharpe_ratio(r, n_trials=1)
    dsr_100 = metrics.deflated_sharpe_ratio(r, n_trials=100)
    dsr_10000 = metrics.deflated_sharpe_ratio(r, n_trials=10_000)
    assert dsr_1 > dsr_100 > dsr_10000


def test_max_drawdown_sign_and_bound():
    r = pd.Series([0.1, -0.5, 0.2])
    dd = metrics.max_drawdown(r)
    assert -1.0 <= dd <= 0.0
    assert abs(dd - (-0.5)) < 1e-12
