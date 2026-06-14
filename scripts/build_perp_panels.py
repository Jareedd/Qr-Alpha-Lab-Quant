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


def _keep_system_awake() -> None:
    """Ask Windows not to idle-sleep while this download runs (it stalled
    twice for hours when the laptop slept mid-request). Self-reverting:
    the request is cleared automatically when the process exits. Does NOT
    override lid-close sleep -- that's a power-plan setting we won't touch.
    No-op off Windows."""
    try:
        import ctypes

        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
        print("[build] system-sleep suppressed for the duration of this run",
              flush=True)
    except Exception as exc:  # noqa: BLE001 -- best-effort, never fatal
        print(f"[build] could not suppress sleep ({exc}); continuing", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=None, help="cap symbols (smoke test)")
    ap.add_argument("--start", default=START)
    args = ap.parse_args()

    _keep_system_awake()
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
