"""SEC<->Tiingo merge feasibility audit (H1 survivorship gate). ZERO trials.

Answers with a NUMBER the load-bearing claim of the "SEC fundamentals + Tiingo
prices" architecture: of the point-in-time S&P 500's DEAD names (removed and not
currently in the index), what fraction get BOTH
  (a) survivorship-safe PRICES from Tiingo, AND
  (b) a SEC ticker->CIK crosswalk for FUNDAMENTALS?

Tiingo is expected to solve the price leg (it carries delisted EOD). The prior
audit (research_log 2026-06-14) measured the SEC dead-ticker crosswalk at ~75%
OVERALL; this isolates the DEAD-name subset, where the survivorship wall actually
is. The end-to-end "both" number is the gate: if it clears the free-SEC baseline
with low reassignment risk, H1 is finally runnable free at 15yr depth; if it
lands at the same wall, Compustat/CRSP remains the real unlock.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import universe as univ
from quantlab.fundamentals_data import FreeSECSource
from quantlab.tiingo_data import TiingoSource


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--limit", type=int, default=0, help="cap dead names probed (0=all)")
    ap.add_argument("--out", default="results/tiingo_sec_merge_audit.json")
    args = ap.parse_args()

    current, changes = univ.fetch_sp500_tables()
    intervals = univ.build_membership_intervals(current, changes, start=args.start)
    members = univ.all_members_in_window(intervals)
    current_set = set(current["ticker"])
    dead = sorted(set(members) - current_set)
    if args.limit:
        dead = dead[: args.limit]
    print(f"[universe] {len(members)} PIT members ever; {len(current_set)} current; "
          f"{len(dead)} dead names to probe")

    sec = FreeSECSource()       # CURRENT-only ticker->CIK map (the documented hole)
    tg = TiingoSource()         # survivorship-safe prices

    # Tiingo coverage via the BULK supported-tickers list (one download), not one
    # metadata call per name -- the list carries start/end for every ticker incl.
    # delisted ones, so this is a pure lookup (and avoids rate limits entirely).
    sup = tg.supported_tickers()
    tcov: dict[str, tuple] = {}
    for _, r in sup.iterrows():
        prev = tcov.get(r["ticker"])
        end = r["enddate"]
        # keep the row with the latest end date if a ticker appears on >1 exchange
        if prev is None or (pd.notna(end) and (prev[1] is None or end > prev[1])):
            tcov[r["ticker"]] = (r["startdate"], end)
    print(f"[tiingo] supported-tickers list: {len(tcov):,} US equities (incl. delisted)")

    rows = []
    for t in dead:
        cik = sec.ticker_cik(t)
        ts, te = tcov.get(t, (None, None))
        rows.append({
            "ticker": t,
            "sec_cik": cik,
            "tiingo_start": str(ts.date()) if ts is not None and pd.notna(ts) else None,
            "tiingo_end": str(te.date()) if te is not None and pd.notna(te) else None,
        })

    n = len(rows) or 1
    tiingo_ok = sum(r["tiingo_start"] is not None for r in rows)
    sec_ok = sum(r["sec_cik"] is not None for r in rows)
    both = sum((r["tiingo_start"] is not None) and (r["sec_cik"] is not None)
               for r in rows)
    report = {
        "n_members_ever": len(members),
        "n_current": len(current_set),
        "n_dead_probed": len(rows),
        "dead_tiingo_price_pct": round(tiingo_ok / n, 4),
        "dead_sec_cik_pct": round(sec_ok / n, 4),
        "dead_both_pct": round(both / n, 4),
        "note": ("Tiingo = survivorship-safe PRICE leg. sec_cik via CURRENT-only "
                 "company_tickers.json (the documented crosswalk hole). A dead "
                 "ticker that DOES map MAY be a REASSIGNMENT to a living company "
                 "(prior finding: >1/2 of recovered links) -> a cik_history "
                 "reassignment-safety cross-check is the next refinement before any "
                 "graded trial. This audit measures the BASELINE merge coverage."),
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[merge audit] DEAD names (n={len(rows)}): "
          f"Tiingo prices {report['dead_tiingo_price_pct']:.0%} | "
          f"SEC crosswalk {report['dead_sec_cik_pct']:.0%} | "
          f"BOTH {report['dead_both_pct']:.0%}")
    print("  Tiingo solves the price leg; the SEC ticker->CIK crosswalk is the "
          "bottleneck (and some 'mapped' dead names are reassignments).")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
