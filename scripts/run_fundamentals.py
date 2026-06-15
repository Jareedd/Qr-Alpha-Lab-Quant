"""H1 / trial #12 entry point — registration-gated. PREP STATE.

    python scripts/run_fundamentals.py --hypothesis H1 --source free_sec

Order of operations (the registration's):
1. Registration gate: H1 must be PROPOSED (law #3).
2. Machinery gate: synthetic planted_quality recovered, null_quality rejected
   (paired). Proves the harness can tell a quality premium from its absence.
3. DATA GATE: a graded trial requires a SURVIVORSHIP-SAFE source. The free SEC
   source is current-ticker-only (~73% coverage; audit 2026-06-14) — running H1
   on it would re-commit trial #1's original sin, so the script REFUSES and
   spends no trial. The day WRDS lands, implement CompustatSource.field_series
   (+ delisting-inclusive prices) and re-run with --source compustat: same
   harness, one command, clean trial #12.

This is the prep promised while sponsorship emails are out: everything except
the data is built and tested.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import fundamentals, metrics
from quantlab.fundamentals_data import CompustatSource, FreeSECSource
from quantlab.registry import require_runnable_registration


def _run_trial(source, n_trials: int) -> None:
    """The graded run — reachable only with a survivorship-safe source. Assembles
    PIT GP/A + accruals/A from `source`, builds the quality signal, backtests at
    the registered slow rebalance, and reports DSR at N. Prices (delisting-
    inclusive, PIT) come from the same safe source."""
    import pandas as pd
    universe = source.universe()                      # survivorship-safe membership
    asof = pd.bdate_range(source.start, source.end, freq="BME")
    panels = fundamentals.pit_feature_panels(source, universe, asof)
    signal = fundamentals.quality_signal(panels["gp_a"], panels["accruals_a"])
    prices = source.prices(universe, asof)
    res = fundamentals.quality_backtest(signal, prices, cost_bps_per_side=10.0)
    dsr = metrics.deflated_sharpe_ratio(res["net"], n_trials=n_trials)
    print(f"[H1] net SR {metrics.sharpe(res['net'], periods=12):.3f}  DSR {dsr:.3f}  "
          f"@ N={n_trials}  turnover {res['annual_turnover']:.2f}/yr")
    print("  (log the row whatever it says — N becomes 12 here.)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypothesis", default="H1")
    ap.add_argument("--n-trials", type=int, default=12)
    ap.add_argument("--source", choices=["free_sec", "compustat"], default="free_sec")
    args = ap.parse_args()

    try:
        require_runnable_registration(args.hypothesis)
    except RuntimeError as exc:
        sys.exit(f"REGISTRATION GATE: {exc}")
    print(f"[registration] {args.hypothesis} verified PROPOSED.")

    print("[gate] synthetic quality world: planted must beat null (paired)...")
    gate = fundamentals.machinery_gate()
    for s, p, n in zip((7, 11, 23), gate["planted_sr"], gate["null_sr"]):
        print(f"  seed {s}: planted SR {p:+.2f} | null SR {n:+.2f}")
    if not gate["passed"]:
        sys.exit(f"MACHINERY GATE FAILED: differential {min(gate['diffs']):.2f} "
                 "<= 0.5 — harness cannot tell quality from its absence; abort.")
    print(f"[gate] PASS (min paired differential {min(gate['diffs']):.2f})")

    source = FreeSECSource() if args.source == "free_sec" else CompustatSource()
    if not source.survivorship_safe:
        sys.exit(
            "\nDATA GATE: the free SEC source is SURVIVORSHIP-BLOCKED -- its "
            "ticker->CIK map is current-only (~73% coverage; dead/renamed names "
            "dropped, audit 2026-06-14). A graded H1 trial on it would re-commit "
            "trial #1's survivorship sin, so this run is REFUSED and spends no "
            "trial (N unchanged).\n  -> Connect CompustatSource (WRDS: filing-"
            "date-PIT fundamentals + delisting-inclusive prices) and re-run with "
            "--source compustat. The harness above is proven; this is one command "
            "from a clean trial #12.")

    _run_trial(source, args.n_trials)                 # reachable once WRDS is wired


if __name__ == "__main__":
    main()
