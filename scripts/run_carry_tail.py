"""H9 / trial #10 — long-tail perp funding carry. Registration-gated.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/run_carry_tail.py --hypothesis H9 --n-trials 10

Same discipline as run_carry.py (H2), one universe apart: the book trades the
liquid TAIL (trailing-ADV ranks 31-150) BENEATH the majors, where funding is
far wider but fills worse. Order of operations is the registration's:
1. Registration gate: H9 must be PROPOSED (law #3).
2. Machinery gate (synthetic planted_carry recovered / priced_carry rejected,
   paired) in THIS environment, immediately before the real run.
3. The real run on the assembled PIT panels with the tail universe:
   funding-inclusive returns, 20 bps/side, shuffled-funding control,
   funding/price decomposition, ex-top-3 robustness, 2020-21 vs 2022+ split.
4. Report against H9's FROZEN success+kill criteria. Does NOT edit
   research_log.md or bump N -- logging the trial is a deliberate act after
   reading the result.
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
import pandas as pd

from quantlab import metrics, perp_carry
from quantlab import perp_data as pdat
from quantlab.registry import require_runnable_registration
from quantlab.synthetic import make_perp_panel

# H9 frozen config
RANK_LO, RANK_HI, MIN_NAMES = 31, 150, 20
COST_BPS_PER_SIDE = 20.0
HORIZON = 7
SUBPERIOD_SPLIT = "2022-01-01"


def _machinery_gate() -> None:
    print("[gate] synthetic carry world: planted must beat priced (paired)...")
    diffs = []
    for seed in (7, 11, 23):
        planted = make_perp_panel(40, 1500, mode="planted_carry", seed=seed)
        priced = make_perp_panel(40, 1500, mode="priced_carry", seed=seed)
        diffs.append(_synth_sr(planted) - _synth_sr(priced))
        print(f"  seed {seed}: paired differential {diffs[-1]:+.2f}")
    if min(diffs) <= 0.6:
        sys.exit(f"MACHINERY GATE FAILED: differential {min(diffs):.2f} <= 0.6 "
                 "-- harness cannot tell carry from its absence; abort.")
    print(f"[gate] PASS (min paired differential {min(diffs):.2f})")


def _synth_sr(panel: pd.DataFrame) -> float:
    funding = panel.attrs["funding"]
    vol = pd.DataFrame(1.0, index=panel.index, columns=panel.columns)
    res = perp_carry.carry_backtest(
        {"price": panel, "dollar_volume": vol, "funding": funding},
        cost_bps_per_side=0.0, top_n=999)
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
    return float(ic.mean()), float(metrics.newey_west_tstat(ic, lags=horizon))


def _run_tail(panels: dict) -> dict:
    """Build the tail universe on THESE panels and run the carry book."""
    uni = perp_carry.rank_band_universe(
        panels["dollar_volume"], RANK_LO, RANK_HI, min_names=MIN_NAMES)
    return perp_carry.carry_backtest(
        panels, cost_bps_per_side=COST_BPS_PER_SIDE, universe=uni)


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

    end = pd.Timestamp.today().strftime("%Y-%m-01")
    base = os.path.join(pdat.CACHE, f"panels_{args.start}_{end}")
    try:
        panels = {n: pd.read_parquet(f"{base}__{n}.parquet")
                  for n in ("price", "dollar_volume", "funding")}
    except OSError:
        sys.exit(f"panels not found at {base}__*.parquet -- run "
                 "scripts/build_perp_panels.py first.")
    print(f"[data] panels: {panels['price'].shape[1]} symbols x "
          f"{panels['price'].shape[0]} days")

    res = _run_tail(panels)
    net, gross = res["net"], res["gross"]
    n_tail = int(res["universe"].sum(axis=1).replace(0, pd.NA).dropna().median())
    print(f"[universe] tail ranks {RANK_LO}-{RANK_HI}: median {n_tail} names/day, "
          f"{int((res['universe'].sum(axis=1) >= MIN_NAMES).sum())} active days")

    fwd = perp_carry.forward_total_return(panels["price"], panels["funding"], HORIZON)
    ic_mean, ic_t = _ic_tstat(res["signal"], fwd, res["universe"], HORIZON)

    stats = metrics.summary(net, gross, res["annual_turnover"], n_trials=args.n_trials)
    stats["mean_rank_ic"] = ic_mean
    stats["ic_tstat_newey_west"] = ic_t
    stats["funding_pnl_cumulative"] = float(res["funding_pnl"].sum())
    stats["price_pnl_cumulative"] = float(res["price_pnl"].sum())
    stats["median_tail_names"] = n_tail

    # Registered paired control: shuffled funding must earn ~nothing.
    ctrl_panels = dict(panels)
    ctrl_panels["funding"] = perp_carry.shuffle_funding(panels["funding"], seed=0)
    stats["control_shuffled_funding_sharpe"] = metrics.sharpe(_run_tail(ctrl_panels)["net"])

    # Robustness: drop the 3 names with the largest |average weight|.
    top3 = res["weights"].abs().mean().nlargest(3).index.tolist()
    p2 = {k: v.drop(columns=top3, errors="ignore") for k, v in panels.items()}
    stats["sharpe_net_ex_top3"] = metrics.sharpe(_run_tail(p2)["net"])
    stats["excluded_top3"] = top3

    # Decay check: 2020-21 vs 2022+ subperiods.
    early = net[net.index < SUBPERIOD_SPLIT]
    late = net[net.index >= SUBPERIOD_SPLIT]
    stats["sharpe_2020_2021"] = metrics.sharpe(early)
    stats["sharpe_2022_plus"] = metrics.sharpe(late)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "metrics_h9_carry_tail.json"), "w") as f:
        json.dump(stats, f, indent=2)
    fig, ax = plt.subplots(figsize=(10, 5))
    (1 + net).cumprod().plot(ax=ax, label=f"net tail carry (SR {stats['sharpe_net']:.2f})")
    ax.legend(); ax.set_title(f"H9 long-tail perp carry — DSR {stats['dsr']:.2f} @ N={args.n_trials}")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "equity_h9_carry_tail.png"), dpi=120)

    print(f"\n=== H9 long-tail carry — trial #{args.n_trials} (net of {COST_BPS_PER_SIDE:.0f}bps/side) ===")
    for k, v in stats.items():
        print(f"  {k:>30}: {v:.4f}" if isinstance(v, float) else f"  {k:>30}: {v}")
    crit = (stats["ic_tstat_newey_west"] <= -2 and stats["sharpe_net"] > 0
            and stats["dsr"] >= 0.95 and stats["sharpe_net_ex_top3"] > 0
            and abs(stats["control_shuffled_funding_sharpe"]) < 0.3)
    print(f"\n  REGISTERED SUCCESS CRITERIA: {'MET -> GRADUATES' if crit else 'NOT MET'} "
          "(t_NW<=-2 AND net SR>0 AND DSR>=0.95 AND ex-top3>0 AND shuffled ~flat)")


if __name__ == "__main__":
    main()
