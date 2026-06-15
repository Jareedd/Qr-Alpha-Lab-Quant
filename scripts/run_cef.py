"""H6 / trial #11 — CEF discount-z mean-reversion. Registration-gated.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/run_cef.py --hypothesis H6 --n-trials 11

Order is the registration's: (1) H6 must be PROPOSED; (2) synthetic machinery
gate (planted reversion recovered / random-walk null rejected, paired) in-env;
(3) the real run on the assembled 5Y-weekly panel with the pre-declared controls;
(4) report vs the FROZEN criteria. Does not auto-log or bump N.
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

from quantlab import cef, cef_data, metrics
from quantlab.registry import require_runnable_registration
from quantlab.synthetic import make_cef_panel

PPY = 52          # weekly
HORIZON = 4       # 4-week hold
LB, MINP = 52, 26
QUANTILE, REBAL, COST = 0.2, 4, 25.0


def machinery_gate() -> None:
    print("[gate] synthetic CEF world: planted reversion must beat null (paired)...")
    diffs = []
    for seed in (1, 2, 3):
        out = {}
        for mode in ("planted_reversion", "null"):
            p = make_cef_panel(60, 250, mode=mode, seed=seed)
            r = p.pct_change(fill_method=None)
            res = cef.reversion_backtest(p.attrs["disc"], r, cost_bps_per_side=0.0,
                                         lookback=LB, min_periods=MINP, quantile=QUANTILE, rebalance=REBAL)
            out[mode] = metrics.sharpe(res["net"], periods=PPY)
        diffs.append(out["planted_reversion"] - out["null"])
        print(f"  seed {seed}: planted {out['planted_reversion']:+.2f} | null {out['null']:+.2f}")
    if min(diffs) <= 0.6:
        sys.exit(f"MACHINERY GATE FAILED ({min(diffs):.2f} <= 0.6) — abort.")
    print(f"[gate] PASS (min paired differential {min(diffs):.2f})")


def fwd_return(ret: pd.DataFrame, h: int) -> pd.DataFrame:
    return ret.shift(-1).rolling(h).sum().shift(-(h - 1))


def run(disc, ret, n_trials, label="") -> dict:
    res = cef.reversion_backtest(disc, ret, cost_bps_per_side=COST,
                                 lookback=LB, min_periods=MINP, quantile=QUANTILE,
                                 rebalance=REBAL, periods_per_year=PPY)
    net = res["net"]
    ic = cef.reversion_ic(disc, fwd_return(ret, HORIZON), lookback=LB, min_periods=MINP)
    s = {
        "sharpe_net": metrics.sharpe(net, periods=PPY),
        "sharpe_gross": metrics.sharpe(res["gross"], periods=PPY),
        "dsr": metrics.deflated_sharpe_ratio(net, n_trials=n_trials),
        "mean_rank_ic": float(ic.mean()),
        "ic_tstat_newey_west": float(metrics.newey_west_tstat(ic, lags=HORIZON)),
        "ann_turnover": res["annual_turnover"],
        "skew": float(net.skew()),
        "max_drawdown": metrics.max_drawdown(net),
        "n_obs": int(len(net)),
    }
    return {"stats": s, "net": net, "weights": res["weights"]}


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
    print(f"[registration] {args.hypothesis} PROPOSED — spends a trial at N={args.n_trials}.")
    machinery_gate()

    base = os.path.join(cef_data.CACHE, "panel")
    px = pd.read_parquet(f"{base}_px.parquet")
    disc = pd.read_parquet(f"{base}_disc.parquet")
    meta = pd.read_parquet(f"{base}_meta.parquet")
    ret = px.pct_change(fill_method=None)
    print(f"[data] {px.shape[1]} funds x {px.shape[0]} weekly obs "
          f"({px.index.min().date()} -> {px.index.max().date()})")

    main_res = run(disc, ret, args.n_trials)
    s = main_res["stats"]

    # --- baseline: equal-weight CEF universe (net of nothing — it's a benchmark)
    ew = ret.mean(axis=1).dropna()
    s["baseline_ew_cef_sharpe"] = metrics.sharpe(ew, periods=PPY)

    # --- control 1: label-shuffle (permute forward returns across funds per week)
    rng = np.random.default_rng(0)
    shuf = ret.copy()
    arr = shuf.to_numpy()
    for i in range(arr.shape[0]):
        fin = np.where(np.isfinite(arr[i]))[0]
        if len(fin) > 1:
            arr[i, fin] = arr[i, fin][rng.permutation(len(fin))]
    shuf = pd.DataFrame(arr, index=ret.index, columns=ret.columns)
    s["control_shuffle_sharpe"] = run(disc, shuf, args.n_trials)["stats"]["sharpe_net"]

    # --- control 2: daily-NAV-only subuniverse (fresh NAVPublished within 7d of last)
    if "NAVPublished" in meta.columns:
        npub = pd.to_datetime(meta["NAVPublished"], errors="coerce")
        fresh = npub[npub >= (npub.max() - pd.Timedelta(days=7))].index
        keep = [c for c in disc.columns if c in set(fresh)]
        s["daily_nav_subuniverse_sharpe"] = run(disc[keep], ret[keep], args.n_trials)["stats"]["sharpe_net"]
        s["n_daily_nav_funds"] = len(keep)

    # --- control 3: ex-largest-mcap-decile
    if "MarketCapUSDm" in meta.columns:
        mc = meta["MarketCapUSDm"].dropna()
        big = set(mc.nlargest(max(1, len(mc) // 10)).index)
        keep = [c for c in disc.columns if c not in big]
        s["ex_top_mcap_decile_sharpe"] = run(disc[keep], ret[keep], args.n_trials)["stats"]["sharpe_net"]

    os.makedirs(args.out, exist_ok=True)
    json.dump(s, open(os.path.join(args.out, "metrics_h6_cef.json"), "w"), indent=2)
    net = main_res["net"]
    fig, ax = plt.subplots(figsize=(10, 5))
    (1 + net).cumprod().plot(ax=ax, label=f"H6 net (SR {s['sharpe_net']:.2f})")
    ax.legend(); ax.set_title(f"H6 CEF discount reversion — DSR {s['dsr']:.2f} @ N={args.n_trials}")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "equity_h6_cef.png"), dpi=120)

    print(f"\n=== H6 CEF discount reversion — trial #{args.n_trials} (net {COST:.0f}bps/side, weekly) ===")
    for k, v in s.items():
        print(f"  {k:>30}: {v:.4f}" if isinstance(v, float) else f"  {k:>30}: {v}")
    crit = (s["ic_tstat_newey_west"] >= 2 and s["sharpe_net"] > 0 and s["dsr"] >= 0.95
            and s["sharpe_net"] > s["baseline_ew_cef_sharpe"] and abs(s["control_shuffle_sharpe"]) < 0.5)
    print(f"\n  REGISTERED CRITERIA: {'MET -> GRADUATES' if crit else 'NOT MET'} "
          "(t_NW>=2 AND net SR>0 AND DSR>=0.95 AND beats EW-CEF AND shuffle flat)")


if __name__ == "__main__":
    main()
