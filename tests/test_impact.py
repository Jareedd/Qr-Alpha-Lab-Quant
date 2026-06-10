"""Square-root impact model: scaling laws, monotonicity, point-in-time ADV.

The impact model's whole value is its *shape* (costs grow like sqrt(AUM) per
unit traded, so total impact grows like AUM^1.5 in dollars / AUM^0.5 as a
return). If the implementation loses that shape, capacity conclusions are
garbage -- so the sqrt scaling itself is asserted, not assumed.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import backtest, impact
from quantlab.synthetic import make_panel


def _setup(n_assets=30, n_days=900, seed=8):
    prices = make_panel(n_assets=n_assets, n_days=n_days, mode="noise", seed=seed)
    rng = np.random.default_rng(seed)
    volumes = pd.DataFrame(
        rng.uniform(1e5, 5e6, prices.shape), index=prices.index, columns=prices.columns
    )
    preds = pd.DataFrame(
        rng.standard_normal(prices.shape), index=prices.index, columns=prices.columns
    ).stack()
    preds.index.names = ["date", "ticker"]
    weights = backtest.predictions_to_weights(preds)
    return prices, volumes, weights


def test_impact_costs_increase_with_aum_at_sqrt_rate():
    prices, volumes, weights = _setup()
    adv = impact.dollar_adv(prices, volumes)
    c1 = impact.impact_costs(weights, prices, adv, aum=1e7, spread_bps=0.0)
    c4 = impact.impact_costs(weights, prices, adv, aum=4e7, spread_bps=0.0)
    # Pure sqrt law with zero spread: 4x AUM -> exactly 2x return drag.
    ratio = c4.sum() / c1.sum()
    assert abs(ratio - 2.0) < 0.01


def test_zero_spread_zero_aum_means_zero_cost():
    prices, volumes, weights = _setup()
    adv = impact.dollar_adv(prices, volumes)
    c = impact.impact_costs(weights, prices, adv, aum=0.0, spread_bps=0.0)
    assert float(c.abs().sum()) == 0.0


def test_spread_component_matches_linear_cost_model():
    # With k=0 (no impact), the model must reduce to the linear backtest cost.
    prices, volumes, weights = _setup()
    adv = impact.dollar_adv(prices, volumes)
    c = impact.impact_costs(weights, prices, adv, aum=1e8, k=0.0, spread_bps=25.0)
    res = backtest.run_backtest(weights, prices, cost_bps=25.0)
    linear_costs = res["gross"] - res["net"]
    pd.testing.assert_series_equal(
        c.loc[linear_costs.index], linear_costs, check_names=False, atol=1e-15
    )


def test_adv_is_point_in_time():
    prices, volumes, weights = _setup()
    adv_full = impact.dollar_adv(prices, volumes)
    poisoned = volumes.copy()
    poisoned.iloc[600:] *= 100  # future liquidity explosion
    adv_poisoned = impact.dollar_adv(prices, poisoned)
    pd.testing.assert_frame_equal(adv_full.iloc[:601], adv_poisoned.iloc[:601])


def test_capacity_curve_monotone_and_dies():
    prices, volumes, weights = _setup()
    adv = impact.dollar_adv(prices, volumes)
    res = backtest.run_backtest(weights, prices, cost_bps=0.0)
    curve = impact.capacity_curve(
        weights, prices, adv, res["gross"], aums=(1e6, 1e8, 1e11)
    )
    srs = curve["sharpe_net"].to_list()
    assert srs[0] > srs[1] > srs[2]          # more AUM never helps
    assert curve["ann_cost_drag"].is_monotonic_increasing
    # A noise strategy at absurd AUM must be dead; capacity_at_zero finds it.
    cap = impact.capacity_at_zero(curve)
    assert cap is not None and cap <= 1e11
