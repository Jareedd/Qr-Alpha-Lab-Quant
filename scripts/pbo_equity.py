"""PBO over the equity price-feature config family (CSCV) — the family-wise
complement to per-trial DSR. ZERO trials (a reproduction diagnostic over
already-logged configs, like the capacity sweep; N unchanged).

It assembles aligned daily net-PnL for the COMPARABLE equity configs — same
point-in-time universe, same 21-day horizon — and runs CSCV:

    #2 ridge / raw label / no neutralize
    #3 ridge / raw label / sector+beta neutral
    #5 ridge / residual label / sector+beta neutral
    #6 GBR   / residual label / sector+beta neutral
    #7 MLP   / residual label / sector+beta neutral

Trials #1 (biased static universe) and #4 (63-day horizon) are EXCLUDED on
purpose — different universe / frequency cannot share a CSCV return matrix, and
forcing them in would be the exact misuse this metric should refuse.

Run:  .venv/Scripts/python.exe scripts/pbo_equity.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from quantlab import (backtest, baselines, features, models, pbo, risk,
                      universe, validation)
from quantlab.data import load_prices

HORIZON = 21
CONFIGS = [
    ("t2_ridge_raw",      "ridge", "raw",      "none"),
    ("t3_ridge_neutral",  "ridge", "raw",      "both"),
    ("t5_ridge_residual", "ridge", "residual", "both"),
    ("t6_gbr_residual",   "gbr",   "residual", "both"),
    ("t7_mlp_residual",   "mlp",   "residual", "both"),
]


def _net_series(panel, model_name, neutralize, prices, mask, sectors, betas) -> pd.Series:
    splitter = validation.WalkForwardSplitter(embargo_days=HORIZON)
    preds = models.walk_forward_predict(panel, splitter, model_name=model_name)
    if neutralize in ("sector", "both") and sectors:
        preds = risk.neutralize_predictions_by_sector(preds, sectors)
    weights = backtest.predictions_to_weights(preds, rebalance_every=HORIZON)
    if neutralize in ("beta", "both"):
        weights = risk.beta_neutralize_weights(weights, betas)
    return backtest.run_backtest(weights, prices, cost_bps=10.0)["net"]


def main() -> int:
    print("[pbo-equity] REPRODUCTION diagnostic (not a trial; N unchanged)")
    current, changes = universe.fetch_sp500_tables()
    intervals = universe.build_membership_intervals(current, changes, start="2010-01-01")
    members = universe.all_members_in_window(intervals)
    prices = load_prices(members, start="2009-01-01", min_coverage=0.0)
    mask = universe.membership_mask(prices.index, prices.columns, intervals)
    sectors = universe.sector_map(current, list(prices.columns))
    print(f"[universe] {prices.shape[1]} names x {prices.shape[0]} days (PIT)")

    feats = features.build_features(prices, member_mask=mask)
    in_index = mask.stack()
    in_index.index.names = ["date", "ticker"]
    panels = {}
    for resid in (False, True):
        labels = features.build_labels(prices, horizon=HORIZON, residualize=resid, member_mask=mask)
        p = features.stack_panel(feats, labels)
        panels["residual" if resid else "raw"] = p[in_index.reindex(p.index, fill_value=False)]

    mkt = baselines.equal_weight_returns(prices, member_mask=mask)
    betas = risk.rolling_beta(prices.pct_change(fill_method=None), mkt)

    nets = {}
    for name, model_name, label, neutralize in CONFIGS:
        print(f"[run] {name} ({model_name}/{label}/{neutralize}) ...", flush=True)
        nets[name] = _net_series(panels[label], model_name, neutralize,
                                 prices, mask, sectors, betas)
    matrix = pd.DataFrame(nets).dropna(how="any")
    print(f"[matrix] {matrix.shape[0]} aligned days x {matrix.shape[1]} configs")

    result = pbo.cscv_pbo(matrix, n_splits=16)
    sharpes = {name: float(s.mean() / s.std() * (252 ** 0.5)) if s.std() > 0 else 0.0
               for name, s in nets.items()}
    result["config_net_sharpe"] = sharpes
    result["_meta"] = {"trial_count_impact": 0, "kind": "reproduction-diagnostic",
                       "configs": [c[0] for c in CONFIGS],
                       "excluded": "trial #1 (biased universe), #4 (63d horizon)"}

    os.makedirs("results", exist_ok=True)
    with open("results/pbo_equity.json", "w") as fh:
        json.dump(result, fh, indent=2)

    print("-" * 64)
    print(f"  configs (net annualized Sharpe, OOS):")
    for name, sr in sharpes.items():
        print(f"    {name:<20} {sr:+.2f}")
    print(f"  PBO = {result['pbo']:.3f}  over {result['n_combinations']:,} symmetric "
          f"splits of {result['n_obs']:,} days")
    print(f"  IS->OOS Sharpe degradation slope = {result['perf_degradation_slope']:.2f}; "
          f"P(OOS loss for the IS-best) = {result['prob_oos_loss']:.2f}")
    print(f"  Wrote results/pbo_equity.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
