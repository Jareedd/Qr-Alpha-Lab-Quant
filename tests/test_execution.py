"""Engine — cost-aware execution-planning known-answer tests."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import execution


def test_target_shares_integer_and_capped():
    w = pd.Series({"A": 0.20, "B": -0.10})
    prices = pd.Series({"A": 100.0, "B": 50.0})
    sh = execution.target_shares(w, equity=1_000_000, prices=prices, max_weight=0.05)
    # capped to 5%: A -> $50k / $100 = 500 sh; B -> -$50k / $50 = -1000 sh
    assert sh["A"] == 500 and sh["B"] == -1000
    assert sh.dtype.kind == "i"                       # integer shares


def test_orders_are_deltas_from_current():
    cur = pd.Series({"A": 100, "B": -200})
    tgt = pd.Series({"A": 500, "B": -200, "C": 50})
    orders = execution.orders_from_targets(cur, tgt)
    assert orders["A"] == 400 and orders["C"] == 50
    assert "B" not in orders.index                    # no change -> no order


def test_turnover_cost_linear():
    prev = pd.Series({"A": 0.0, "B": 0.0})
    tgt = pd.Series({"A": 0.5, "B": -0.5})            # turnover 1.0
    assert execution.turnover_cost(prev, tgt, cost_bps=10.0) == pytest.approx(10 / 1e4)


def test_execution_plan_shape():
    w = pd.Series({"A": 0.04, "B": -0.04})
    prices = pd.Series({"A": 20.0, "B": 40.0})
    plan = execution.execution_plan(w, equity=500_000, prices=prices,
                                    prev_weights=pd.Series({"A": 0.0, "B": 0.0}),
                                    max_weight=0.05, cost_bps=10.0)
    assert plan["n_orders"] == 2
    assert plan["gross_notional"] == pytest.approx(0.08 * 500_000, rel=0.02)
    assert plan["est_rebalance_cost_frac"] == pytest.approx(0.08 * 10 / 1e4, rel=1e-6)
