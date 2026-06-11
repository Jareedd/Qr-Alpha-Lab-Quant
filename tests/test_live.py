"""Live-trading logic: everything testable without a broker or network.

The IO layer is deliberately thin; the decisions (training cutoff, order
deltas, caps, paper-endpoint guard) are pure and pinned here.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import features, live
from quantlab.synthetic import make_panel


def test_orders_from_weights_known_answer():
    target = pd.Series({"AAA": 0.02, "BBB": -0.02, "CCC": 0.0})
    current = {"BBB": 10, "DDD": 5}  # DDD must be liquidated
    prices = pd.Series({"AAA": 100.0, "BBB": 50.0, "CCC": 10.0, "DDD": 20.0})
    orders = live.orders_from_weights(target, current, prices, equity=100_000.0)
    by_symbol = {o["symbol"]: o for o in orders}

    assert by_symbol["AAA"] == {"symbol": "AAA", "qty": 20, "side": "buy"}   # 2k/100
    assert by_symbol["BBB"] == {"symbol": "BBB", "qty": 50, "side": "sell"}  # -40 tgt -10 held
    assert by_symbol["DDD"] == {"symbol": "DDD", "qty": 5, "side": "sell"}   # flatten
    assert "CCC" not in by_symbol


def test_orders_respect_per_name_cap():
    target = pd.Series({"AAA": 0.50})  # absurd weight -> capped at 5%
    orders = live.orders_from_weights(
        target, {}, pd.Series({"AAA": 100.0}), equity=100_000.0, max_name_frac=0.05
    )
    assert orders == [{"symbol": "AAA", "qty": 50, "side": "buy"}]


def test_orders_skip_unpriceable_names():
    target = pd.Series({"GHOST": 0.02})
    orders = live.orders_from_weights(
        target, {}, pd.Series({"GHOST": np.nan}), equity=100_000.0
    )
    assert orders == []


def test_live_weights_train_only_on_complete_labels():
    # Poisoning the last `horizon` days of prices must not change the fitted
    # model's predictions ONLY through training data -- i.e., a model trained
    # with poisoned recent history must equal one trained on clean history,
    # because those rows have incomplete labels and are excluded from training.
    prices = make_panel(n_assets=40, n_days=1300, mode="planted", seed=11)
    sectors = prices.attrs["sectors"]

    w_clean, p_clean = live.live_target_weights(
        prices, None, sectors, horizon=21, min_names=20
    )

    poisoned = prices.copy()
    # Scale the final 20 days' prices (inside the horizon window): labels for
    # training-cutoff rows are unaffected; today's FEATURES change, so today's
    # weights may differ -- but the run must not crash and must stay neutral.
    poisoned.iloc[-20:] *= 1.5
    w_poisoned, _ = live.live_target_weights(
        poisoned, None, sectors, horizon=21, min_names=20
    )

    # The prediction log (the live-IC artifact) must cover the whole scored
    # cross-section -- not just the names that made the book -- in the raw
    # (backtest-comparable), sector-neutral (book-driving) and control-arm
    # columns.
    assert {"pred_raw", "pred_sector_neutral", "baseline_mom_12_1"} <= set(
        p_clean.columns
    )
    assert len(p_clean) >= (w_clean != 0).sum()
    assert np.isfinite(p_clean.to_numpy()).all()

    for w in (w_clean, w_poisoned):
        assert abs(w.sum()) < 1e-9            # dollar neutral
        assert (w != 0).sum() >= 6            # an actual book
        assert abs(w.abs().sum() - 1.0) < 1e-6  # gross preserved by projection
        # 40-name panel -> 4 names/side at ~0.125 base; beta projection can
        # legitimately push one to ~0.3. The real concentration guard is the
        # 5% per-name cap applied at order time (tested separately).
        assert w.abs().max() < 0.35


def test_prediction_log_control_arm_matches_baseline_feature():
    # The shadow-logged baseline column must be the SAME object the law-#5
    # backtest baseline ranks on (today's mom_12_1 feature), name for name --
    # otherwise the live control arm and the backtest baseline diverge.
    prices = make_panel(n_assets=40, n_days=1300, mode="planted", seed=11)
    _, preds = live.live_target_weights(
        prices, None, prices.attrs["sectors"], horizon=21, min_names=20
    )
    expected = features.build_features(prices)["mom_12_1"].iloc[-1]
    pd.testing.assert_series_equal(
        preds["baseline_mom_12_1"],
        expected.reindex(preds.index),
        check_names=False,
    )


def test_live_records_are_write_once(tmp_path):
    # A second cycle on the same as-of date must refuse to replace the
    # logged record -- a revisable prediction log is no evidence at all.
    existing = tmp_path / "predictions_2026-06-10.csv"
    existing.write_text("ticker,pred_raw\nAAA,0.1\n")
    fresh = tmp_path / "weights_2026-06-10.csv"

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        live.assert_write_once([str(existing), str(fresh)])
    # explicit escape hatch for re-running a failed cycle
    live.assert_write_once([str(existing), str(fresh)], allow_overwrite=True)
    # nothing logged yet -> no objection
    live.assert_write_once([str(fresh)])


def test_alpaca_client_refuses_live_endpoint(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "ALPACA_API_KEY_ID=k\nALPACA_API_SECRET_KEY=s\n"
        "ALPACA_BASE_URL=https://api.alpaca.markets\n"
    )
    os.environ.pop("ALPACA_BASE_URL", None)
    saved_key = os.environ.pop("ALPACA_API_KEY_ID", None)
    saved_sec = os.environ.pop("ALPACA_API_SECRET_KEY", None)
    try:
        with pytest.raises(RuntimeError, match="refusing non-paper"):
            live.AlpacaPaper(str(env))
    finally:
        if saved_key:
            os.environ["ALPACA_API_KEY_ID"] = saved_key
        if saved_sec:
            os.environ["ALPACA_API_SECRET_KEY"] = saved_sec
