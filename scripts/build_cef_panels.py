"""H6 Stage-1: assemble the CEF panel (price/NAV/discount) + metadata.

Fetches the live CEF universe (DailyPricing) and each fund's full daily history
(pricinghistory/MAX), caches per fund (gitignored), and writes aligned panels +
a metadata table to data_cache/cef/. Polite sequential fetch. Zero-trial.

Run:  PYTHONPATH=src .venv/Scripts/python.exe scripts/build_cef_panels.py
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quantlab import cef_data


def main() -> None:
    uni = cef_data.universe(use_cache=True)
    tickers = sorted(uni["Ticker"].dropna().unique().tolist())
    print(f"[universe] {len(tickers)} live CEFs")

    px, nav, disc = {}, {}, {}
    ok, empty, err = 0, 0, 0
    for i, t in enumerate(tickers):
        try:
            h = cef_data.price_history(t, "5Y", use_cache=True)
            if len(h):
                px[t], nav[t], disc[t] = h["px"], h["nav"], h["disc"]
                ok += 1
            else:
                empty += 1
        except Exception:
            err += 1
        if not os.path.exists(os.path.join(cef_data.CACHE, f"ph_{t}_5Y.json")):
            time.sleep(0.25)  # polite only on real fetches (cache hits don't sleep)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(tickers)}  ok={ok} empty={empty} err={err}")

    if not px:
        print(f"[panel] EMPTY — ok={ok} empty={empty} err={err}; aborting")
        return
    px = pd.DataFrame(px).sort_index()
    nav = pd.DataFrame(nav).reindex_like(px)
    disc = pd.DataFrame(disc).reindex_like(px)
    print(f"[panel] {px.shape[1]} funds x {px.shape[0]} days "
          f"({px.index.min().date()} -> {px.index.max().date()})  ok={ok} empty={empty} err={err}")

    base = os.path.join(cef_data.CACHE, "panel")
    px.to_parquet(f"{base}_px.parquet")
    nav.to_parquet(f"{base}_nav.parquet")
    disc.to_parquet(f"{base}_disc.parquet")
    # metadata aligned to the panel columns
    meta_cols = ["Ticker", "CategoryName", "MarketCapUSDm", "AvgDailyVolume",
                 "DistributionRatePrice", "ExpenseRatio", "NAVPublished",
                 "DistributionFrequency", "IsLeveraged"]
    meta = uni[[c for c in meta_cols if c in uni.columns]].set_index("Ticker")
    meta = meta[meta.index.isin(px.columns)]
    meta.to_parquet(f"{base}_meta.parquet")
    print(f"[saved] {base}_{{px,nav,disc,meta}}.parquet")


if __name__ == "__main__":
    main()
