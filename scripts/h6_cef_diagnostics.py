"""H6 trial #11 adversarial diagnostics (the prime-directive 'investigate the
suspiciously-good result' pass). The face-value run scored net SR 1.70 / DSR
0.987 / t_NW 6.68 with a -2% maxDD -- too clean. These checks decompose it:

  * ENTRY-LAG sweep: a bid-ask-bounce / 1-week-reversal artifact collapses when
    entry is delayed; a durable edge decays gracefully.
  * COST sensitivity: CEF small-fund spreads routinely exceed the 25 bps headline.
  * SIGNAL-shuffle: a structure-masquerading-as-signal artifact would be POSITIVE.

Verdict (see research_log trial #11): the signal is real but the monetizable
edge net of the first-week bounce AND realistic spreads is ~0 -> H6 does NOT
graduate. Reproduces the inline diagnostics; writes results/h6_cef_diagnostics.json.
"""
from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from quantlab import cef, cef_data, metrics


def main() -> None:
    base = os.path.join(cef_data.CACHE, "panel")
    px = pd.read_parquet(f"{base}_px.parquet")
    disc = pd.read_parquet(f"{base}_disc.parquet")
    ret = px.pct_change(fill_method=None)
    w = cef.reversion_weights(disc, lookback=52, min_periods=26, quantile=0.2, rebalance=4)
    to = w.diff().abs().sum(axis=1).fillna(0.0)
    out = {"entry_lag_net_sr": {}, "cost_net_sr_at_lag1": {}}

    for lag in (1, 2, 3, 4):
        g = (w.shift(lag) * ret).sum(axis=1, min_count=1)
        net = (g - to * 25 / 1e4).dropna()
        out["entry_lag_net_sr"][f"{lag}w"] = round(metrics.sharpe(net, periods=52), 3)

    g1 = (w.shift(1) * ret).sum(axis=1, min_count=1)
    for c in (25, 50, 75, 100, 150):
        net = (g1 - to * c / 1e4).dropna()
        out["cost_net_sr_at_lag1"][f"{c}bps"] = round(metrics.sharpe(net, periods=52), 3)

    # signal-shuffle (proper artifact check)
    rng = np.random.default_rng(0)
    z = cef.discount_zscore(disc, 52, 26)
    a = z.to_numpy().copy()
    for i in range(a.shape[0]):
        fin = np.where(np.isfinite(a[i]))[0]
        if len(fin) > 1:
            a[i, fin] = a[i, fin][rng.permutation(len(fin))]
    zs = pd.DataFrame(a, index=z.index, columns=z.columns)
    tgt = pd.DataFrame(np.nan, index=disc.index, columns=disc.columns)
    sig = -zs
    for i in range(0, len(disc.index), 4):
        row = sig.iloc[i].dropna(); n = int(len(row) * 0.2); tgt.iloc[i] = 0.0
        if n < 2:
            continue
        tgt.iloc[i, tgt.columns.get_indexer(row.nlargest(n).index)] = 0.5 / n
        tgt.iloc[i, tgt.columns.get_indexer(row.nsmallest(n).index)] = -0.5 / n
    ws = tgt.ffill(limit=3).fillna(0.0)
    gs = (ws.shift(1) * ret).sum(axis=1, min_count=1)
    tos = ws.diff().abs().sum(axis=1).fillna(0.0)
    out["signal_shuffle_net_sr"] = round(metrics.sharpe((gs - tos * 25 / 1e4).dropna(), periods=52), 3)
    out["verdict"] = ("real signal but microstructure-inflated (lag-2 net 0.43) and "
                      "cost-fragile (net<=0 by 75bps); does NOT graduate")

    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/h6_cef_diagnostics.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
