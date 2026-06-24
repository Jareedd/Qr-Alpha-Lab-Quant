"""H1 name->CIK crosswalk audit — does resolving dead names BY COMPANY NAME
(SEC cik-lookup-data.txt) recover the survivorship-safe fundamentals coverage
that ticker-keyed maps cannot? ZERO trials, no forward returns.

Baseline (research_log 2026-06-24): the SEC current ticker->CIK map covers only
39% of the 309 dead S&P names (and some are reassignments). Method 1 (bulk
submissions tickers) recovers ~0% (the arrays are emptied at delisting). This
audit measures the NAME route: Wikipedia "Removed Security" name -> normalized
match against cik-lookup-data.txt -> operating-company gate (has 10-K filings).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import cik_crosswalk as cx
from quantlab import universe as univ
from quantlab.fundamentals_data import FreeSECSource


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-gate", dest="gate", action="store_false", default=True,
                    help="skip the operating-company 10-K gate (faster, but no "
                    "reassignment/namesake filtering)")
    ap.add_argument("--out", default="results/h1_name_crosswalk_audit.json")
    args = ap.parse_args()

    current, changes = univ.fetch_sp500_tables()
    intervals = univ.build_membership_intervals(current, changes, start=args.start)
    members = univ.all_members_in_window(intervals)
    current_set = set(current["ticker"])
    dead = sorted(set(members) - current_set)
    if args.limit:
        dead = dead[: args.limit]

    names = cx.fetch_sp500_security_names()      # free dead-name names (Wikipedia)
    sec = FreeSECSource()                        # current-only map = the baseline
    resolver = cx.NameCikResolver()
    resolver.index()                             # warm cik-lookup-data.txt (cached)
    print(f"[universe] {len(dead)} dead names; cik-lookup index "
          f"{len(resolver.index()):,} normalized names")

    rows = []
    for i, t in enumerate(dead):
        nm = names.get(t)
        base_cik = sec.ticker_cik(t)             # ticker route (baseline)
        name_cik = None
        if nm:
            name_cik = (resolver.operating_cik(nm) if args.gate
                        else (cx.match_name(nm, resolver.index()) or [None])[0])
        rows.append({"ticker": t, "name": nm, "baseline_ticker_cik": base_cik,
                     "name_cik": name_cik})
        if (i + 1) % 25 == 0:
            print(f"  resolved {i + 1}/{len(dead)}")

    n = len(rows) or 1
    base_ok = sum(r["baseline_ticker_cik"] is not None for r in rows)
    name_ok = sum(r["name_cik"] is not None for r in rows)
    either = sum((r["baseline_ticker_cik"] is not None) or (r["name_cik"] is not None)
                 for r in rows)
    has_name = sum(r["name"] is not None for r in rows)
    report = {
        "n_dead": len(rows),
        "n_with_wikipedia_name": has_name,
        "baseline_ticker_cik_pct": round(base_ok / n, 4),
        "name_crosswalk_cik_pct": round(name_ok / n, 4),
        "either_route_pct": round(either / n, 4),
        "gate_applied": args.gate,
        "note": ("name_cik = Wikipedia removed-name -> normalize -> "
                 "cik-lookup-data.txt -> latest-10-K operating filer. Resolves by "
                 "NAME, so ticker reassignment (e.g. MON->SPAC) cannot leak; "
                 "residual risk is a dead/live NAMESAKE collision (dead_by date "
                 "gate is the next refinement). Coverage here = a CIK with 10-K "
                 "filings exists; pulling its SEC fundamentals is the graded run."),
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[name crosswalk] dead names (n={len(rows)}): "
          f"baseline ticker->CIK {report['baseline_ticker_cik_pct']:.0%} | "
          f"NAME->CIK {report['name_crosswalk_cik_pct']:.0%} | "
          f"either {report['either_route_pct']:.0%}")
    # sanity vs known ground truth
    truth = {"CELG": "0000816284", "MON": "0001110783", "XLNX": "0000743988",
             "CERN": "0000804753", "RHT": "0001087423"}
    by_t = {r["ticker"]: r for r in rows}
    for t, cik in truth.items():
        r = by_t.get(t)
        if r:
            ok = "OK" if r["name_cik"] == cik else f"got {r['name_cik']}"
            print(f"    {t} ({r['name']}): name_cik {ok}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
