"""Daily live paper-trading entrypoint (Phase 6).

    python scripts/live_trade.py                # predict + submit paper orders
    python scripts/live_trade.py --dry-run      # predict + log, no orders

Run after US market close (e.g. 21:30 UTC weekdays). Predictions are logged
to results/live/ BEFORE any order is sent -- the prediction log is the
artifact that later yields live IC vs backtest IC, the project's ultimate
out-of-sample test.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab.live import run_daily


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ridge", choices=["ridge", "gbr", "mlp"])
    ap.add_argument("--dry-run", action="store_true", help="log predictions, send no orders")
    ap.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="re-run a failed cycle: replace the (normally write-once) "
        "per-date prediction/weights logs",
    )
    ap.add_argument("--out", default="results/live")
    args = ap.parse_args()

    summary = run_daily(
        model_name=args.model,
        out_dir=args.out,
        submit=not args.dry_run,
        allow_overwrite=args.allow_overwrite,
    )
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if summary.get("n_failed", 0) > 0:
        print("  (some orders failed -- typically hard-to-borrow shorts; logged in summary)")


if __name__ == "__main__":
    main()
