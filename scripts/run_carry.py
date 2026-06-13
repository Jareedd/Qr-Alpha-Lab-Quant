"""H2 / trial #8 — the single registered carry run. Registration-gated.

    python scripts/run_carry.py --hypothesis H2 --n-trials 8

Order of operations is the registration's, not negotiable:
1. Registration gate: H2 must be PROPOSED (law #3, mechanized).
2. Machinery gate, in THIS environment, immediately before the real run:
   the synthetic planted_carry world must be recovered and priced_carry
   rejected (paired differential). If the harness can't tell planted from
   priced carry today, no real number is trustworthy -- abort.
3. The real run on the assembled PIT panels: funding-inclusive returns,
   net of fees, with the registered paired control (shuffled funding),
   the funding-income/price-drag decomposition, and the top-3-exclusion
   robustness check.
4. Report against the registered success AND kill criteria. The script
   does NOT edit research_log.md or bump N -- spending the trial is a
   deliberate human act taken after reading the result.
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

from quantlab import metrics, perp_carry
from quantlab import perp_data as pdat
from quantlab.registry import require_runnable_registration
from quantlab.synthetic import make_perp_panel


def _machinery_gate() -> None:
    """Synthetic falsification gate for the carry harness (law 4)."""
    print("[gate] synthetic carry world: planted must beat priced (paired)...")
    diffs = []
    for seed in (7, 11, 23):
        planted = make_perp_panel(40, 1500, mode="planted_carry", seed=seed)
        priced = make_perp_panel(40, 1500, mode="priced_carry", seed=seed)
        for world in (planted, priced):
            world.attrs  # noqa: B018 -- panels carry funding in attrs
        sr_p = _synth_sr(planted)
        sr_n = _synth_sr(priced)
        diffs.append(sr_p - sr_n)
        print(f"  seed {seed}: planted SR {sr_p:+.2f} | priced SR {sr_n:+.2f}")
    if min(diffs) <= 0.6:
        sys.exit(f"MACHINERY GATE FAILED: planted-priced differential "
                 f"{min(diffs):.2f} <= 0.6 -- harness cannot tell carry from "
                 "its absence; the real run is not trustworthy.")
    print(f"[gate] PASS (min paired differential {min(diffs):.2f})")


def _synth_sr(panel: pd.DataFrame) -> float:
    funding = panel.attrs["funding"]
    vol = pd.DataFrame(1.0, index=panel.index, columns=panel.columns)
    res = perp_carry.carry_backtest(
        {"price": panel, "dollar_volume": vol, "funding": funding},
        cost_bps_per_side=0.0, top_n=999,
    )
    return metrics.sharpe(res["net"])


def _ic_tstat(signal, fwd, universe, horizon):
    sig = signal.where(universe)
    ics = []
    for d in sig.index:
        a, b = sig.loc[d].dropna(), fwd.loc[d].dropna()
        common = a.index.intersection(b.index)
        if len(common) >= 6:
            ics.append(a[common].rank().corr(b[common].rank()))
    ic = pd.Series(ics).dropna()
    # negative sign expected: HIGH funding -> SHORT -> negative IC of the
    # raw funding signal vs forward return is the PROFITABLE direction.
    return float(ic.mean()), float(metrics.newey_west_tstat(ic, lags=horizon))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypothesis", required=True)
    ap.add_argument("--n-trials", type=int, required=True)
    ap.add_argument("--start", default="2019-09-01")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    try:
        require_runnable_registration(args.hypothesis)
    except RuntimeError as exc:
        sys.exit(f"REGISTRATION GATE: {exc}")
    print(f"[registration] {args.hypothesis} verified PROPOSED -- this run "
          f"SPENDS a trial at N={args.n_trials}. Log it whatever it says.")

    _machinery_gate()

    # Load the assembled panels (built by scripts/build_perp_panels.py).
    end = pd.Timestamp.today().strftime("%Y-%m-01")
    base = os.path.join(pdat.CACHE, f"panels_{args.start}_{end}")
    try:
        panels = {n: pd.read_parquet(f"{base}__{n}.parquet")
                  for n in ("price", "dollar_volume", "funding")}
    except OSError:
        sys.exit(f"panels not found at {base}__*.parquet -- run "
                 "scripts/build_perp_panels.py first (the data download).")
    price = panels["price"]
    print(f"[data] panels: {price.shape[1]} symbols x {price.shape[0]} days")

    res = perp_carry.carry_backtest(panels)
    net, gross = res["net"], res["gross"]
    fwd = perp_carry.forward_total_return(panels["price"], panels["funding"], 7)
    ic_mean, ic_t = _ic_tstat(res["signal"], fwd, res["universe"], 7)

    stats = metrics.summary(net, gross, res["annual_turnover"], n_trials=args.n_trials)
    stats["mean_rank_ic"] = ic_mean
    stats["ic_tstat_newey_west"] = ic_t
    stats["funding_pnl_cumulative"] = float(res["funding_pnl"].sum())
    stats["price_pnl_cumulative"] = float(res["price_pnl"].sum())

    # Registered paired control: shuffled funding must earn ~nothing.
    ctrl_panels = dict(panels)
    ctrl_panels["funding"] = perp_carry.shuffle_funding(panels["funding"], seed=0)
    ctrl = perp_carry.carry_backtest(ctrl_panels)
    stats["control_shuffled_funding_sharpe"] = metrics.sharpe(ctrl["net"])

    # Robustness: drop the 3 names with the largest |average weight|.
    w = res["weights"]
    top3 = w.abs().mean().nlargest(3).index.tolist()
    p2 = {k: v.drop(columns=top3, errors="ignore") for k, v in panels.items()}
    res2 = perp_carry.carry_backtest(p2)
    stats["sharpe_net_ex_top3"] = metrics.sharpe(res2["net"])
    stats["excluded_top3"] = top3

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "metrics_h2_carry.json"), "w") as f:
        json.dump(stats, f, indent=2)
    fig, ax = plt.subplots(figsize=(10, 5))
    (1 + net).cumprod().plot(ax=ax, label=f"net carry (SR {stats['sharpe_net']:.2f})")
    (1 + ctrl["net"]).cumprod().plot(ax=ax, alpha=0.6,
                                     label=f"shuffled-funding control "
                                           f"(SR {stats['control_shuffled_funding_sharpe']:.2f})")
    ax.legend(); ax.set_title(f"H2 perp carry — DSR {stats['dsr']:.2f} @ N={args.n_trials}")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "equity_h2_carry.png"), dpi=120)

    print("\n=== H2 carry — trial #8 (registered config, net of 7bps/side) ===")
    for k, v in stats.items():
        print(f"  {k:>28}: {v:.4f}" if isinstance(v, float) else f"  {k:>28}: {v}")
    crit = (stats["ic_tstat_newey_west"] <= -2 and stats["sharpe_net"] > 0
            and stats["dsr"] >= 0.95 and stats["sharpe_net_ex_top3"] > 0
            and abs(stats["control_shuffled_funding_sharpe"]) < 0.3)
    print(f"\n  REGISTERED SUCCESS CRITERIA: "
          f"{'MET' if crit else 'NOT MET'} "
          "(t_NW<=-2 AND net SR>0 AND DSR>=0.95 AND survives ex-top3 AND "
          "shuffled control ~flat)")


if __name__ == "__main__":
    main()
