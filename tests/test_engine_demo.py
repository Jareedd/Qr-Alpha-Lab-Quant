"""Regression pin for the engine capstone demo (scripts/engine_demo.py).

The demo IS a runnable proof of the engine's honest property; this test keeps
that proof from silently rotting. Synthetic, deterministic (fixed seed), no
market data, no trial-count impact.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import engine_demo


def test_demo_all_honest_properties_pass():
    d = engine_demo.run_demo()
    assert engine_demo.assert_properties(d) == []


def test_demo_sizes_null_to_zero_and_commits_to_edge():
    d = engine_demo.run_demo()["honest_property"]
    # the headline: real edge gets capital, the paired null gets ~none
    assert d["planted_avg_gross_exposure"] > 0.2
    assert d["null_avg_gross_exposure"] < 0.05
    assert d["planted_net_mean_period"] > 0


def test_demo_neutralization_zeroes_net_beta():
    d = engine_demo.run_demo()["neutralization"]
    assert d["net_beta_exposure_raw"] > 1e-3          # un-neutralized book carries beta
    assert d["net_beta_exposure_neutralized"] < 1e-9  # projection zeroes it
