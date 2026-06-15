"""Execution/risk engine — end-to-end integration tests.

The headline property: the engine commits capital to a real edge and sizes a
no-edge book to ~ZERO — the honest counterweight to over-confidence. Built on
the synthetic quality world (planted vs null) so the "edge" is ground truth.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab.engine import PortfolioEngine
from quantlab.synthetic import make_quality_panel


def _engine():
    # monthly cadence (periods=12); 36-month trailing window for the confidence
    # estimate; loose caps so the test exercises sizing, not the limits.
    return PortfolioEngine(periods=12, lookback=36, target_vol=0.10,
                           max_weight=0.10, max_gross=2.0)


def test_engine_sizes_edge_up_and_noise_to_zero():
    eng = _engine()
    planted = make_quality_panel(120, 180, mode="planted_quality", seed=7)
    null = make_quality_panel(120, 180, mode="null_quality", seed=7)
    bt_p = eng.backtest({"q": planted.attrs["gp_a"]}, planted)
    bt_n = eng.backtest({"q": null.attrs["gp_a"]}, null)
    # commits to the real edge, stays ~flat on noise
    assert bt_p["avg_gross_exposure"] > 0.2
    assert bt_n["avg_gross_exposure"] < 0.05
    assert bt_p["net"].mean() > 0                          # the edge is profitable net


def test_engine_respects_position_and_gross_limits():
    eng = _engine()
    panel = make_quality_panel(120, 180, mode="planted_quality", seed=11)
    w = eng.backtest({"q": panel.attrs["gp_a"]}, panel)["weights"]
    assert w.abs().max().max() <= 0.10 + 1e-9              # per-name cap
    assert w.abs().sum(axis=1).max() <= 2.0 + 1e-9         # gross cap


def test_latest_orders_are_integer_and_capped():
    eng = _engine()
    panel = make_quality_panel(120, 180, mode="planted_quality", seed=7)
    plan = eng.latest_orders({"q": panel.attrs["gp_a"]}, panel, equity=1_000_000)
    assert plan["target_shares"].dtype.kind == "i"
    assert plan["n_orders"] >= 0
    assert plan["gross_notional"] <= 2.0 * 1_000_000 + 1.0  # within gross cap
