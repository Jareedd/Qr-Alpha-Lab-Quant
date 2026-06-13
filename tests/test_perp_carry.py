"""H2 carry harness (quantlab.perp_carry) — pure logic, no network.

The real-data run is gated on these plus the synthetic machinery gate;
here we pin the construction (funding-inclusive returns, PIT universe,
dollar-neutral book, the shuffled-funding control, forward-label timing)
on hand-built frames with known answers.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import metrics, perp_carry
from quantlab.synthetic import make_perp_panel


def _frame(vals, cols, start="2022-01-03"):
    idx = pd.bdate_range(start, periods=len(vals))
    return pd.DataFrame(vals, index=idx, columns=cols)


def test_total_return_subtracts_funding():
    price = _frame([[100.0], [110.0]], ["A"])
    funding = _frame([[0.0], [0.01]], ["A"])  # longs pay 1% on day 2
    tot = perp_carry.total_returns(price, funding)
    # price return day2 = +10%, minus 1% funding = +9%
    assert tot["A"].iloc[1] == pytest.approx(0.09)


def test_pit_universe_top_n_and_min_names():
    vol = _frame(
        [[10, 5, 1, 8, 3, 9]] * 40,
        ["A", "B", "C", "D", "E", "F"],
    )
    uni = perp_carry.pit_universe(vol, top_n=3, lookback=5, min_names=3)
    last = uni.iloc[-1]
    assert set(last[last].index) == {"A", "F", "D"}  # top 3 by volume
    # top_n larger than the field selects everyone (the synthetic-gate path)
    uni_all = perp_carry.pit_universe(vol, top_n=999, lookback=5, min_names=3)
    assert uni_all.iloc[-1].all()


def test_carry_weights_are_dollar_neutral_and_correctly_signed():
    # 8 names, funding ascending A..H; quartile (n_side=2): short G,H
    # (highest funding => they pay => we collect), long A,B (lowest).
    cols = list("ABCDEFGH")
    funding = _frame([[i / 1000 for i in range(8)]] * 30, cols)
    uni = pd.DataFrame(True, index=funding.index, columns=cols)
    sig = perp_carry.carry_signal(funding, lookback=7)
    w = perp_carry.carry_weights(sig, uni, quantile=0.25, rebalance=7)
    row = w.iloc[-1]
    assert abs(row.sum()) < 1e-12               # dollar neutral
    assert abs(row.abs().sum() - 1.0) < 1e-9    # gross 1
    assert row["H"] < 0 and row["G"] < 0        # pay funding -> short
    assert row["A"] > 0 and row["B"] > 0        # receive -> long


def test_shuffle_funding_preserves_marginals_destroys_cross_section():
    cols = list("ABCDE")
    funding = _frame([[1.0, 2.0, 3.0, 4.0, 5.0]] * 10, cols)
    shuf = perp_carry.shuffle_funding(funding, seed=1)
    # each date's MULTISET of values is preserved (a permutation)...
    for d in funding.index:
        assert sorted(shuf.loc[d]) == sorted(funding.loc[d])
    # ...but the per-symbol assignment changed somewhere
    assert not shuf.equals(funding)


def test_forward_total_return_timing_is_forward_only():
    price = _frame([[100.0], [101.0], [102.0], [103.0], [104.0]], ["A"])
    funding = _frame([[0.0]] * 5, ["A"])
    fwd = perp_carry.forward_total_return(price, funding, horizon=2)
    tot = perp_carry.total_returns(price, funding)
    # fwd[t] must equal tot[t+1] + tot[t+2]
    expected = tot["A"].iloc[1] + tot["A"].iloc[2]
    assert fwd["A"].iloc[0] == pytest.approx(expected)


def test_machinery_gate_planted_beats_priced():
    # The synthetic falsification gate the real run runs first: planted
    # carry recovered, priced (null) rejected, paired per seed.
    diffs = []
    for seed in (7, 11):
        out = {}
        for mode in ("planted_carry", "priced_carry"):
            p = make_perp_panel(40, 1500, mode=mode, seed=seed)
            vol = pd.DataFrame(1.0, index=p.index, columns=p.columns)
            res = perp_carry.carry_backtest(
                {"price": p, "dollar_volume": vol, "funding": p.attrs["funding"]},
                cost_bps_per_side=0.0, top_n=999,
            )
            out[mode] = metrics.sharpe(res["net"])
        diffs.append(out["planted_carry"] - out["priced_carry"])
    assert min(diffs) > 0.6
