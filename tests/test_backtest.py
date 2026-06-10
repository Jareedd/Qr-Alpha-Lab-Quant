import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import backtest
from quantlab.synthetic import make_panel


def _toy_preds(prices):
    rng = np.random.default_rng(0)
    wide = pd.DataFrame(
        rng.standard_normal(prices.shape), index=prices.index, columns=prices.columns
    )
    return wide.stack()


def test_costs_strictly_reduce_returns():
    prices = make_panel(n_assets=30, n_days=800, mode="noise", seed=3)
    preds = _toy_preds(prices)
    preds.index.names = ["date", "ticker"]
    w = backtest.predictions_to_weights(preds)
    free = backtest.run_backtest(w, prices, cost_bps=0.0)
    costly = backtest.run_backtest(w, prices, cost_bps=25.0)
    assert costly["net"].sum() < free["net"].sum()
    assert np.allclose(free["gross"], costly["gross"])


def test_weights_dollar_neutral_and_bounded():
    prices = make_panel(n_assets=40, n_days=600, mode="noise", seed=4)
    preds = _toy_preds(prices)
    preds.index.names = ["date", "ticker"]
    w = backtest.predictions_to_weights(preds)
    active = w[w.abs().sum(axis=1) > 0]
    assert (active.sum(axis=1).abs() < 1e-9).all()      # dollar neutral
    assert (active.abs().sum(axis=1) - 1.0 < 1e-9).all()  # gross exposure <= 1


def test_no_lookahead_same_day_returns_not_exploitable():
    # The classic backtest bug: weights formed from day t's return earning day
    # t's return. A correct backtester applies weights formed at t to returns
    # from t+1 onward. In a memoryless noise panel, day t's return has no
    # power for t+1, so using TODAY's realized return as the signal must earn
    # ~nothing. (A buggy, unshifted backtester would show daily SR >> 1 here.)
    prices = make_panel(n_assets=20, n_days=400, mode="noise", seed=5)
    rets = prices.pct_change()
    preds = rets.stack()  # today's own return as today's "prediction"
    preds.index.names = ["date", "ticker"]
    w = backtest.predictions_to_weights(preds, rebalance_every=1)
    res = backtest.run_backtest(w, prices, cost_bps=0.0)
    daily_sr = abs(res["net"].mean() / (res["net"].std() + 1e-12))
    assert daily_sr < 0.15

    # Sanity check of the check: genuine one-day foresight SHOULD be hugely
    # profitable in a correct backtester (prediction made at t, earns t+1).
    foresight = rets.shift(-1).stack()
    foresight.index.names = ["date", "ticker"]
    w2 = backtest.predictions_to_weights(foresight, rebalance_every=1)
    res2 = backtest.run_backtest(w2, prices, cost_bps=0.0)
    sr2 = res2["net"].mean() / (res2["net"].std() + 1e-12)
    assert sr2 > 1.0
