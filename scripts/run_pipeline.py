"""End-to-end research pipeline run.

Examples:
    python scripts/run_pipeline.py --data planted          # sanity: must find signal
    python scripts/run_pipeline.py --data noise            # sanity: must find nothing
    python scripts/run_pipeline.py --data yfinance         # real data (needs internet)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantlab import backtest, baselines, features, metrics, models, validation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["yfinance", "planted", "noise"], default="planted")
    ap.add_argument("--model", choices=["ridge", "gbr"], default="ridge")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument(
        "--n-trials",
        type=int,
        default=1,
        help="How many strategy variants you have tried IN TOTAL (be honest). "
        "Used by the Deflated Sharpe Ratio.",
    )
    ap.add_argument("--out", default="results")
    ap.add_argument(
        "--fail-if-dsr-below",
        type=float,
        default=None,
        help="Exit non-zero if DSR < this (CI gate: planted data must recover).",
    )
    ap.add_argument(
        "--fail-if-dsr-above",
        type=float,
        default=None,
        help="Exit non-zero if DSR > this (CI gate: noise data finding alpha = leakage).",
    )
    args = ap.parse_args()

    if args.data == "yfinance":
        from quantlab.data import load_prices

        prices = load_prices()
    else:
        from quantlab.synthetic import make_panel

        prices = make_panel(mode=args.data)
    print(f"[data] {prices.shape[1]} assets x {prices.shape[0]} days ({args.data})")

    feats = features.build_features(prices)
    labels = features.build_labels(prices, horizon=args.horizon)
    panel = features.stack_panel(feats, labels)
    print(f"[features] panel: {len(panel):,} rows, {len(feats)} features")

    splitter = validation.WalkForwardSplitter(embargo_days=args.horizon)
    preds = models.walk_forward_predict(panel, splitter, model_name=args.model)
    ic = models.information_coefficient(preds, panel)
    print(f"[model] {args.model}: mean rank IC = {ic.mean():.4f} (t={ic.mean()/ic.sem():.2f})")

    weights = backtest.predictions_to_weights(preds)
    result = backtest.run_backtest(weights, prices, cost_bps=args.cost_bps)
    stats = metrics.summary(
        result["net"], result["gross"], result["annual_turnover"], n_trials=args.n_trials
    )
    stats["mean_rank_ic"] = float(ic.mean())

    # Baselines (law #5): same OOS dates, same backtester, same costs.
    mom_w = baselines.momentum_baseline_weights(feats, preds.index)
    mom_res = backtest.run_backtest(mom_w, prices, cost_bps=args.cost_bps)
    ew = baselines.equal_weight_returns(prices, start=result["net"].index[0])
    stats["baseline_mom_sharpe_net"] = metrics.sharpe(mom_res["net"])
    stats["baseline_ew_sharpe"] = metrics.sharpe(ew)
    stats["beats_mom_baseline"] = bool(stats["sharpe_net"] > stats["baseline_mom_sharpe_net"])

    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.data}_{args.model}"
    with open(os.path.join(args.out, f"metrics_{tag}.json"), "w") as f:
        json.dump(stats, f, indent=2)

    fig, ax = plt.subplots(figsize=(10, 5))
    (1 + result["net"]).cumprod().plot(ax=ax, label=f"net ({args.cost_bps}bps)")
    (1 + result["gross"]).cumprod().plot(ax=ax, label="gross", alpha=0.6)
    ax.set_title(f"Long-short equity curve -- {tag} | net SR={stats['sharpe_net']:.2f} "
                 f"DSR={stats['dsr']:.2f}")
    ax.legend()
    ax.set_ylabel("growth of $1")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"equity_{tag}.png"), dpi=120)

    print("\n=== Out-of-sample results (all predictions are walk-forward) ===")
    for k, v in stats.items():
        print(f"  {k:>18}: {v:.4f}" if isinstance(v, float) else f"  {k:>18}: {v}")
    verdict = (
        "PASS: signal recovered" if stats["dsr"] > 0.95
        else "No statistically defensible signal (as expected for noise / weak edges)"
    )
    print(f"\n  DSR verdict: {verdict}")
    print(
        f"  Baselines (same OOS window, net of costs): "
        f"12-1 momentum SR={stats['baseline_mom_sharpe_net']:.2f}, "
        f"equal-weight SR={stats['baseline_ew_sharpe']:.2f} -> model "
        f"{'beats' if stats['beats_mom_baseline'] else 'DOES NOT beat'} momentum baseline"
    )

    if args.fail_if_dsr_below is not None and stats["dsr"] < args.fail_if_dsr_below:
        sys.exit(
            f"FALSIFICATION GATE FAILED: DSR {stats['dsr']:.4f} < "
            f"{args.fail_if_dsr_below} -- pipeline failed to recover a planted signal."
        )
    if args.fail_if_dsr_above is not None and stats["dsr"] > args.fail_if_dsr_above:
        sys.exit(
            f"FALSIFICATION GATE FAILED: DSR {stats['dsr']:.4f} > "
            f"{args.fail_if_dsr_above} -- pipeline 'found alpha' in noise: hunt the leak."
        )


if __name__ == "__main__":
    main()
