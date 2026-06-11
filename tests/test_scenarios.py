"""Delisting-return scenario tool (synthetic.inject_delisting_returns).

The bound only means something if the injection is surgical: exactly one
synthetic print per dead name, on the right day, at the right price, and
nothing else in the panel moves.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab.synthetic import inject_delisting_returns, make_panel


def _panel_with_deaths():
    prices = make_panel(n_assets=10, n_days=200, mode="noise", seed=5)
    cols = list(prices.columns)
    prices.loc[prices.index[100]:, cols[0]] = np.nan   # died mid-window
    prices.loc[prices.index[197]:, cols[1]] = np.nan   # gap inside the buffer
    prices[cols[2]] = np.nan                            # never priced
    return prices, cols


def test_injection_is_one_synthetic_print_at_the_right_price():
    prices, cols = _panel_with_deaths()
    out = inject_delisting_returns(prices, -0.30, end_buffer_days=5)

    dead = cols[0]
    last_real = prices[dead].last_valid_index()
    death_day = out.index[out.index.get_loc(last_real) + 1]
    assert np.isclose(out.loc[death_day, dead], prices.loc[last_real, dead] * 0.70)
    # exactly one cell added for the dead name, nothing after it
    assert out[dead].last_valid_index() == death_day
    assert out[dead].notna().sum() == prices[dead].notna().sum() + 1
    # and the resulting return is exactly the scenario's delisting return
    assert np.isclose(out[dead].pct_change(fill_method=None).loc[death_day], -0.30)

    assert out.attrs["delist_injected"] == 1
    assert out.attrs["delist_return"] == -0.30


def test_living_recent_and_unpriced_names_are_untouched():
    prices, cols = _panel_with_deaths()
    out = inject_delisting_returns(prices, -0.30, end_buffer_days=5)

    # still-trading names: byte-identical
    for c in cols[3:]:
        assert out[c].equals(prices[c])
    # a hole within the end buffer is "maybe still trading", not a death
    assert out[cols[1]].equals(prices[cols[1]])
    # a never-priced name has no last print to extend
    assert out[cols[2]].isna().all()
    # the original panel is never mutated (scenario worlds are copies)
    assert prices[cols[0]].notna().sum() == out[cols[0]].notna().sum() - 1


def test_zero_return_injection_extends_but_does_not_move_prices():
    prices, cols = _panel_with_deaths()
    out = inject_delisting_returns(prices, 0.0, end_buffer_days=5)
    dead = cols[0]
    last_real = prices[dead].last_valid_index()
    death_day = out.index[out.index.get_loc(last_real) + 1]
    assert np.isclose(out.loc[death_day, dead], prices.loc[last_real, dead])
