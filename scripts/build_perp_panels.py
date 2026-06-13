"""Download + cache the full USDT-perp panels for H2 (trial #8).

Heavy and resumable: per-symbol parquet caches persist, so an interrupted
run re-runs cheaply. Assembled panels are written to
data_cache/perp/panels_{start}_{end}.parquet (one file per field).

    python scripts/build_perp_panels.py            # full universe, full history
    python scripts/build_perp_panels.py --max 40   # smoke slice (engineering only)
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import perp_data as pd_

START = "2019-09-01"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=None, help="cap symbols (smoke test)")
    ap.add_argument("--start", default=START)
    args = ap.parse_args()

    syms = pd_.list_usdt_perp_symbols()
    if args.max:
        syms = syms[: args.max]
    print(f"[build] {len(syms)} symbols from {args.start}", flush=True)

    panels = pd_.build_panels(syms, start=args.start, progress=True)
    end = panels["price"].attrs["data_end"]
    out = os.path.join(pd_.CACHE, f"panels_{args.start}_{end}")
    for name, frame in panels.items():
        frame.to_parquet(f"{out}__{name}.parquet")
    price = panels["price"]
    print(f"[build] done: price {price.shape}, "
          f"{len(price.attrs['missing_symbols'])} symbols had no klines; "
          f"panels -> {out}__*.parquet", flush=True)


if __name__ == "__main__":
    main()
