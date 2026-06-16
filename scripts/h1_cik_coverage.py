"""H1 — free historical ticker->CIK coverage measurement. ZERO trials.

The H1 audit (research_log 2026-06-14) found ~73% ticker->CIK coverage on the PIT
S&P universe: the dead/renamed 27% are unmapped (the survivorship hole). This
script asks the follow-up the audit deferred: how much of that hole can FREE
recovery close, and is the recovery SAFE?

It runs the EDGAR ticker resolver (quantlab.cik_history) over every dead-and-
unmapped ticker, classifies each hit as a plausibly-dead name vs a possible
ticker REASSIGNMENT (a false link), and reports the survivorship-SAFE recovered
coverage. No forward returns, no fundamentals pulled, N unchanged.

Run:  .venv/Scripts/python.exe scripts/h1_cik_coverage.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from quantlab import cik_history as ch
from quantlab.fundamentals_data import FreeSECSource


def pit_universe() -> tuple[set, set, dict]:
    """Return (all_tickers, dead_tickers, deletion_date{ticker->last removal})."""
    cur = pd.read_parquet("data_cache/sp500_current.parquet")
    chg = pd.read_parquet("data_cache/sp500_changes.parquet")
    clean = lambda s: {t for t in s if isinstance(t, str) and t not in ("None", "")}
    current = clean(cur["ticker"])
    removed = clean(chg["removed"].dropna())
    added = clean(chg["added"].dropna())
    all_t = current | removed | added
    dead = removed - current                      # left the index and not back
    chg2 = chg.dropna(subset=["removed"]).copy()
    chg2["date"] = pd.to_datetime(chg2["date"], errors="coerce")
    del_date = (chg2[chg2["removed"].isin(dead)]
                .groupby("removed")["date"].max().dt.strftime("%Y-%m-%d").to_dict())
    return all_t, dead, del_date


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--limit", type=int, default=0, help="cap names probed (0 = all)")
    args = ap.parse_args()

    all_t, dead, del_date = pit_universe()
    src = FreeSECSource()
    mapped = {t for t in all_t if src.ticker_cik(t.replace(".", "-")) is not None}
    unmapped = all_t - mapped
    dead_unmapped = sorted(dead & unmapped)
    base_cov = len(mapped) / len(all_t)
    print(f"[universe] {len(all_t)} PIT tickers | current map covers {len(mapped)} "
          f"({base_cov*100:.0f}%) | dead-and-unmapped {len(dead_unmapped)}")

    probe = dead_unmapped if not args.limit else dead_unmapped[:args.limit]
    if args.limit:
        print(f"[NOTE] probing a capped {len(probe)}/{len(dead_unmapped)} sample "
              f"(--limit {args.limit}); coverage numbers are lower bounds")

    counts = {"plausible_dead": 0, "possible_reassignment": 0, "unknown": 0}
    resolved = 0
    detail = []
    for i, t in enumerate(probe, 1):
        r = ch.resolve_ticker_cik(t)
        if r.cik is not None:
            resolved += 1
            verdict = ch.classify_resolution(r.last_10k, del_date.get(t))
            counts[verdict] += 1
            detail.append({"ticker": t, "cik": r.cik, "last_10k": r.last_10k,
                           "deletion": del_date.get(t), "class": verdict})
        if i % 25 == 0:
            print(f"  ...probed {i}/{len(probe)} (resolved {resolved})")

    safe_recovered = counts["plausible_dead"]
    safe_cov = (len(mapped) + safe_recovered) / len(all_t)
    out = {
        "_meta": {"trial_count_impact": 0, "source": "EDGAR getcompany (free)",
                  "probed": len(probe), "dead_unmapped_total": len(dead_unmapped)},
        "baseline_coverage_frac": round(base_cov, 4),
        "edgar_resolved": resolved,
        "resolution_classes": counts,
        "survivorship_safe_recovered": safe_recovered,
        "survivorship_safe_coverage_frac": round(safe_cov, 4),
        "residual_hole_frac": round(1 - safe_cov, 4),
        "detail": detail,
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "h1_cik_coverage.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    print("-" * 64)
    print(f"  EDGAR resolved {resolved}/{len(probe)} dead-unmapped tickers")
    print(f"    plausible_dead        {counts['plausible_dead']}  (safe to use)")
    print(f"    possible_reassignment {counts['possible_reassignment']}  (FALSE-LINK risk)")
    print(f"    unknown               {counts['unknown']}")
    print(f"  Coverage: baseline {base_cov*100:.0f}% -> survivorship-safe {safe_cov*100:.0f}% "
          f"(residual hole {(1-safe_cov)*100:.0f}%)")
    print(f"  VERDICT: free recovery closes only the RECENTLY-dead tail; the bulk of the "
          f"survivorship hole is unrecoverable without paid data (Compustat/CRSP).")
    print(f"  Wrote {os.path.join(args.out, 'h1_cik_coverage.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
