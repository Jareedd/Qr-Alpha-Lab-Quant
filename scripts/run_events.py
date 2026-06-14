"""H8 / trial #9 — discretionary-deletion post-effective drift vs matched
control. Registration-gated; synthetic machinery gate before the real run.

    python scripts/run_events.py --hypothesis H8 --n-trials 9
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
import numpy as np
import pandas as pd

from quantlab import events, metrics
from quantlab.registry import require_runnable_registration
from quantlab.synthetic import inject_post_event_drift, make_panel

HORIZON = 60


def _machinery_gate() -> None:
    """Synthetic planted-event gate (law 4): recover a planted post-event
    drift, reject (one-sided) when none is planted."""
    print("[gate] synthetic planted-event world: recover drift, reject null...")
    panel = make_panel(n_assets=60, n_days=1500, mode="noise", seed=5)
    rng = np.random.default_rng(3)
    ds = panel.index[300:-80]
    ev = [(ds[rng.integers(0, len(ds))], panel.columns[rng.integers(0, 60)])
          for _ in range(50)]
    df = pd.DataFrame([{"effective_date": d, "ticker": t} for d, t in ev])
    planted_panel = inject_post_event_drift(panel, ev, drift=0.15, horizon=HORIZON)
    planted = events.event_study(df, planted_panel, planted_panel * 1.0,
                                 horizon=HORIZON)
    null = events.event_study(df, panel, panel * 1.0, horizon=HORIZON)
    sp = metrics.sharpe(planted["daily_portfolio"])
    s0 = metrics.sharpe(null["daily_portfolio"])
    print(f"  planted Sharpe {sp:+.2f} | null Sharpe {s0:+.2f} | "
          f"event-excess planted {planted['event_total_excess'].mean():+.3f} "
          f"null {null['event_total_excess'].mean():+.3f}")
    if not (sp > 0.5 and planted["event_total_excess"].mean() > 0.05
            and s0 < 0.3):
        sys.exit("MACHINERY GATE FAILED: harness did not recover planted "
                 "event drift / reject null. Real run not trustworthy.")
    print("[gate] PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypothesis", required=True)
    ap.add_argument("--n-trials", type=int, required=True)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    try:
        require_runnable_registration(args.hypothesis)
    except RuntimeError as exc:
        sys.exit(f"REGISTRATION GATE: {exc}")
    print(f"[registration] {args.hypothesis} verified PROPOSED -- SPENDS a "
          f"trial at N={args.n_trials}.")

    _machinery_gate()

    from quantlab import universe as univ
    from quantlab.data import load_prices, load_volumes

    current, changes = univ.fetch_sp500_tables()
    intervals = univ.build_membership_intervals(current, changes, start="2010-01-01")
    members = univ.all_members_in_window(intervals)
    prices = load_prices(members, start="2009-01-01", min_coverage=0.0)
    volumes = load_volumes(members, start="2009-01-01")
    dollar_vol = (prices * volumes.reindex_like(prices)).dropna(how="all")
    mask = univ.membership_mask(prices.index, prices.columns, intervals)

    dels = events.discretionary_deletions(changes, start="2010-01-01")
    print(f"[events] {len(dels)} discretionary deletions; "
          f"{dels['ticker'].isin(prices.columns).sum()} priceable")

    res = events.event_study(dels, prices, dollar_vol, member_mask=mask,
                             horizon=HORIZON)
    port = res["daily_portfolio"]
    n = res["n_events"]
    t_nw = metrics.newey_west_tstat(port, lags=HORIZON)
    dsr = metrics.deflated_sharpe_ratio(port, n_trials=args.n_trials)
    ev_mean = float(res["event_total_excess"].mean())
    ev_t = (ev_mean / (res["event_total_excess"].std() / np.sqrt(n))
            if n > 1 else float("nan"))

    # subperiod (decay check): 2010-2014 vs later
    early = port.loc[:"2014-12-31"]
    late = port.loc["2015-01-01":]
    stats = {
        "n_events_usable": n,
        "daily_net_sharpe": metrics.sharpe(port),
        "daily_t_nw": float(t_nw),
        "dsr": float(dsr),
        "n_trials_assumed": args.n_trials,
        "event_mean_excess_60d": ev_mean,
        "event_tstat_simple": float(ev_t),
        "control_raw_rebound_60d": float(res["control_raw_return"].mean()),
        "sharpe_2010_2014": metrics.sharpe(early),
        "sharpe_2015_plus": metrics.sharpe(late),
        "n_days": int(len(port)),
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "metrics_h8_events.json"), "w") as f:
        json.dump(stats, f, indent=2)
    if len(port):
        fig, ax = plt.subplots(figsize=(10, 4))
        (1 + port).cumprod().plot(ax=ax)
        ax.set_title(f"H8 deletion vs matched control — net SR "
                     f"{stats['daily_net_sharpe']:.2f}, DSR {dsr:.2f} @ N={args.n_trials}")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "equity_h8_events.png"), dpi=120)

    print("\n=== H8 deletion rebound vs matched control — trial #9 ===")
    for k, v in stats.items():
        print(f"  {k:>26}: {v:.4f}" if isinstance(v, float) else f"  {k:>26}: {v}")
    met = (t_nw >= 2 and dsr >= 0.95 and metrics.sharpe(port) > 0
           and min(metrics.sharpe(early), metrics.sharpe(late)) > 0)
    print(f"\n  REGISTERED CRITERIA: {'MET' if met else 'NOT MET'} "
          "(daily t_NW>=+2 AND DSR>=0.95 AND net SR>0 AND not 2010-14-only)")


if __name__ == "__main__":
    main()
