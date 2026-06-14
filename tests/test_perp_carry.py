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


def test_carry_weights_flattens_dropped_names():
    # Regression for a real pre-trial bug: a name in the book one rebalance
    # but NOT selected the next must go to ZERO, not keep its stale weight.
    # Funding flips between two rebalance windows so the quartiles fully
    # swap; after the second rebalance the first window's names must be flat.
    cols = list("ABCDEFGH")
    early = [8, 7, 6, 5, 4, 3, 2, 1]   # A,B highest -> short; G,H lowest -> long
    late = [1, 2, 3, 4, 5, 6, 7, 8]    # reversed: now A,B long flips to short etc.
    rows = [early] * 7 + [late] * 7
    funding = _frame([[v / 1000 for v in r] for r in rows], cols)
    uni = pd.DataFrame(True, index=funding.index, columns=cols)
    sig = perp_carry.carry_signal(funding, lookback=3)
    w = perp_carry.carry_weights(sig, uni, quantile=0.25, rebalance=7)

    # On the second rebalance row the book must be a FRESH dollar-neutral
    # vector -- not a superposition of two windows' books.
    second = w.iloc[7]
    assert abs(second.sum()) < 1e-12
    assert abs(second.abs().sum() - 1.0) < 1e-9
    # every row stays dollar-neutral and gross never exceeds 1 (the bug
    # would have pushed gross above 1 as stale positions accumulated)
    assert w.sum(axis=1).abs().max() < 1e-9
    assert w.abs().sum(axis=1).max() < 1.0 + 1e-9


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


def test_rank_band_universe_selects_the_tail_not_the_majors():
    # H9: descending volume => rank A=1 (most liquid) ... F=6. The band
    # [3,5] must select exactly the tail ranks C,D,E and exclude the majors.
    cols = list("ABCDEF")
    vol = _frame([[10, 9, 8, 7, 6, 5]] * 20, cols)
    uni = perp_carry.rank_band_universe(vol, rank_lo=3, rank_hi=5,
                                        lookback=5, min_names=2)
    last = uni.iloc[-1]
    assert set(last[last].index) == {"C", "D", "E"}   # ranks 3,4,5
    assert not last["A"] and not last["B"]            # majors excluded
    assert not last["F"]                              # below the band
    # too few names to reach past rank_lo + min_names => empty universe
    thin = perp_carry.rank_band_universe(vol, rank_lo=3, rank_hi=5,
                                         lookback=5, min_names=5)
    assert not thin.iloc[-1].any()


def test_carry_backtest_uses_injected_universe():
    # H9 injects a tail mask; carry_backtest must honour it and never put
    # weight on names outside it (H2's top_n path is unaffected).
    cols = [f"S{i}" for i in range(12)]
    funding = _frame([[i / 1000 for i in range(12)]] * 30, cols)
    price = _frame([[100.0] * 12] * 30, cols)
    vol = _frame([[1.0] * 12] * 30, cols)
    panels = {"price": price, "dollar_volume": vol, "funding": funding}
    inside = cols[2:10]                       # 8 names => n_side=2 at q=0.25
    outside = cols[:2] + cols[10:]
    uni = pd.DataFrame(False, index=funding.index, columns=cols)
    uni[inside] = True
    res = perp_carry.carry_backtest(panels, universe=uni, cost_bps_per_side=0.0,
                                    quantile=0.25, rebalance=7)
    held = res["weights"].fillna(0.0)
    assert (held[outside].abs().to_numpy() < 1e-12).all()   # majors never held
    assert held.sum(axis=1).abs().max() < 1e-9              # stays dollar-neutral


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
