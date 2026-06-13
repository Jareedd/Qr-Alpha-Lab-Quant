"""H8 pre-trial power gate: count discretionary S&P 500 deletions.

    python scripts/h8_event_census.py

ZERO trials, ZERO price data: this script touches only the membership
change table (dates, tickers, Wikipedia's reason text). H8's registration
spec kills the hypothesis at zero cost if discretionary deletions number
fewer than ~100 in 2010->present -- this census produces that number
BEFORE any registration is signed or any price is downloaded.

Output: results/h8_event_census.json + a console table, including the
unknown-bucket rows verbatim (a census that hides what it could not
classify is not a census).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from quantlab import universe as univ


def main() -> None:
    current, changes = univ.fetch_sp500_tables()
    removals = changes.dropna(subset=["removed"]).copy()
    removals = removals[removals["date"] >= "2010-01-01"]
    removals["bucket"] = removals["reason"].map(univ.classify_removal_reason)

    counts = removals["bucket"].value_counts().to_dict()
    by_year = (
        removals[removals["bucket"] == "discretionary"]
        .groupby(removals["date"].dt.year)
        .size()
    )

    print(f"S&P 500 removals since 2010: {len(removals)}")
    for bucket, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {bucket:>16}: {n}")
    print("\ndiscretionary deletions per year:")
    for year, n in by_year.items():
        print(f"  {year}: {n}")

    n_disc = counts.get("discretionary", 0)
    verdict = (
        "POWER GATE PASSES (>= 100 events)" if n_disc >= 100
        else f"POWER GATE FAILS ({n_disc} < 100) -- H8 dies at zero cost"
    )
    print(f"\n{verdict}")

    unknowns = removals[removals["bucket"] == "unknown"]
    if len(unknowns):
        print(f"\n{len(unknowns)} unclassified reasons (shown, not dropped):")
        for _, row in unknowns.head(15).iterrows():
            print(f"  {row['date'].date()} {row['removed']}: {row['reason']!r}")

    os.makedirs("results", exist_ok=True)
    payload = {
        "asof": str(pd.Timestamp.today().date()),
        "window": "2010->present",
        "n_removals": int(len(removals)),
        "buckets": {k: int(v) for k, v in counts.items()},
        "discretionary_by_year": {int(y): int(n) for y, n in by_year.items()},
        "power_gate_threshold": 100,
        "power_gate_passes": bool(n_disc >= 100),
        "note": "keyword census only; H8 registration mandates a 20-event "
                "manual reconciliation against press releases before any run",
    }
    with open("results/h8_event_census.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("\n[census] written to results/h8_event_census.json")


if __name__ == "__main__":
    main()
