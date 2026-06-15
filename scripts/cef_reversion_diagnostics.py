"""H6 trial #11 adversarial diagnostics — run AFTER the registered result, to
decide whether a too-good number (net SR ~1.1, DSR ~0.999) is a real edge or a
leak. Mirrors the trial-#8 carry diagnostics (results/h2_carry_diagnostics.json).

The decisive test is the ENTRY-LAG SWEEP: form the signal at week w but enter
1..k weeks later. A real multi-week discount reversion decays GRACEFULLY; a
look-ahead or 1-week microstructure/bid-ask bounce COLLAPSES immediately. Plus:
subperiod split (is it all 2020 panic-reversion?) and shuffle-control stability
across seeds (the registered control came in at 0.277, close to its 0.3 line).

Loads the cached panels — no re-download, does not touch the registered run.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from quantlab import cef_data, cef_reversion, metrics

WPY = 52
OUT = os.path.join("results", "h6_reversion_diagnostics.json")


def main() -> None:
    panels = cef_data.build_weekly_panels([])  # cached; [] is fine, won't refetch
    price, discount = panels["price"], panels["discount"]
    total_ret = price.pct_change(fill_method=None)
    signal = cef_reversion.discount_z(discount)
    weights = cef_reversion.reversion_weights(signal)
    cost = weights.diff().abs().sum(axis=1).fillna(0.0) * (25.0 / 1e4)

    # 1. ENTRY-LAG SWEEP (the decisive leak test).
    lag_sr = {}
    for lag in range(0, 6):
        held = weights.shift(1 + lag)
        net = ((held * total_ret).sum(axis=1, min_count=1) - cost).dropna()
        lag_sr[f"lag_{lag}w"] = round(metrics.sharpe(net, periods=WPY), 3)

    # 2. SUBPERIOD SPLIT (2020-concentration check).
    held0 = weights.shift(1)
    net0 = ((held0 * total_ret).sum(axis=1, min_count=1) - cost).dropna()
    sub = {}
    for label, lo, hi in [("pre_2020", "2000", "2020-01-01"),
                          ("2020", "2020-01-01", "2021-01-01"),
                          ("post_2020", "2021-01-01", "2100")]:
        seg = net0[(net0.index >= lo) & (net0.index < hi)]
        sub[label] = {"sharpe": round(metrics.sharpe(seg, periods=WPY), 3),
                      "n_weeks": int(len(seg))}
    by_year = {str(y): round(metrics.sharpe(net0[net0.index.year == y], periods=WPY), 3)
               for y in sorted(set(net0.index.year))}

    # 3. SHUFFLE-CONTROL STABILITY across seeds (registered came in at 0.277).
    shuf_sr = []
    for seed in range(6):
        sh = cef_reversion.shuffle_returns(total_ret, seed=seed)
        c = ((held0 * sh).sum(axis=1, min_count=1)).dropna()
        shuf_sr.append(round(metrics.sharpe(c, periods=WPY), 3))

    diag = {
        "entry_lag_sweep": lag_sr,
        "entry_lag_interpretation": (
            "graceful decay across lags => real multi-week reversion; a collapse "
            "from lag_0 to lag_1 => look-ahead or 1-week microstructure bounce."),
        "subperiod": sub,
        "by_year_sharpe": by_year,
        "shuffle_control_sharpe_by_seed": shuf_sr,
        "shuffle_control_max_abs": round(max(abs(s) for s in shuf_sr), 3),
    }
    os.makedirs("results", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    print("=== H6 trial #11 diagnostics ===")
    print("  entry-lag SR:", lag_sr)
    print("  subperiod SR:", {k: v["sharpe"] for k, v in sub.items()})
    print("  by-year SR:", by_year)
    print("  shuffle control SR by seed:", shuf_sr,
          f"(max |SR| {diag['shuffle_control_max_abs']})")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
