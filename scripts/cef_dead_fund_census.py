"""H6 Stage-1 DEAD-FUND census (ZERO-TRIAL) — the make-or-break gate.

Tests the H6 thesis's load-bearing claim that CEF deaths happen at NAV
(liquidation / open-ending / merger / term maturity), so omitting dead funds
biases a discount-long backtest CONSERVATIVELY. If the deaths are NAV events,
the project's #1 killer (missing dead names) is, uniquely, a tailwind here.

Pipeline (quantlab.cef_deaths): EFTS Form 25 (delistings) per year -> fund-name
pre-screen -> submissions JSON -> N-2 (closed-end/BDC) filter -> fully-delisted
check (exclude preferred/note redemptions of still-listed funds) -> terminal-
outcome classifier. Unknowns are SHOWN, never guessed (H8 discipline).

Decision gate (graduation_candidates §3): if dead funds are absent from free
sources AND the conservative-direction argument fails -> KILL H6 before a trial.

Output: results/cef_dead_fund_census.json. N unchanged. No price data, no
signal, no forward return.

Run: python scripts/cef_dead_fund_census.py [start_year] [end_year]
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import cef_deaths as cd

OUT = os.path.join("results", "cef_dead_fund_census.json")


def main(start_year: int = 2019, end_year: int = 2026) -> None:
    # 1. enumerate delistings, dedup by CIK (a fund delists once; keep earliest
    #    in-window Form 25 date), name-pre-screen before any submissions fetch.
    earliest: dict[str, dict] = {}
    n_form25 = 0
    for yr in range(start_year, end_year + 1):
        hits = cd.form_delistings(yr)
        n_form25 += len(hits)
        for h in hits:
            if not cd.is_fund_name(h["name"]):
                continue
            cur = earliest.get(h["cik"])
            if cur is None or (h["date"] or "") < (cur["date"] or "9999"):
                earliest[h["cik"]] = h
        print(f"  [{yr}] Form 25 hits={len(hits)}  fund-name candidates so far={len(earliest)}",
              flush=True)

    # 2. classify each candidate from its submissions JSON.
    deaths, n_cef = [], 0
    for i, (cik, h) in enumerate(earliest.items()):
        try:
            sub = cd.submissions(cik)
        except Exception as e:  # noqa: BLE001
            print(f"    skip CIK {cik}: {type(e).__name__}", flush=True)
            continue
        cls = cd.classify_death(sub["name"] or h["name"], set(sub["forms"]),
                                sub.get("tickers", []))
        if not cls["is_cef_or_bdc"]:
            continue
        n_cef += 1
        if cls["fully_delisted"]:                     # an actual fund death
            deaths.append({
                "cik": cik, "name": sub["name"], "delist_date": h["date"],
                "sic": sub.get("sic"), "is_bdc": cls["is_bdc"],
                "outcome": cls["outcome"], "nav_event": cls["nav_event"],
            })
        if (i + 1) % 50 == 0:
            print(f"    classified {i+1}/{len(earliest)} candidates", flush=True)

    # 3. aggregate + the conservative-direction verdict.
    outcomes = Counter(d["outcome"] for d in deaths)
    nav_true = sum(1 for d in deaths if d["nav_event"] is True)
    nav_unknown = sum(1 for d in deaths if d["nav_event"] is None)
    nav_false = sum(1 for d in deaths if d["nav_event"] is False)
    n_deaths = len(deaths)
    n_bdc = sum(1 for d in deaths if d["is_bdc"])
    frac_nav = round(nav_true / n_deaths, 3) if n_deaths else None

    census = {
        "window": f"{start_year}-{end_year}",
        "source": "SEC EDGAR EFTS form 25-NSE (delistings) + submissions JSON",
        "method_note": ("Deaths = N-2 filers (closed-end/BDC) that FULLY delisted "
                        "(no current ticker; preferred/note redemptions of still-"
                        "listed funds excluded). Outcome from filing-type "
                        "signature + name; unknowns shown, not guessed. "
                        "liquidation-vs-merger split needs filing text (deferred); "
                        "the NAV-event vs distress BINARY is what the gate needs."),
        "counts": {
            "form25_delistings_scanned": n_form25,
            "fund_name_candidates": len(earliest),
            "closed_end_or_bdc_filers": n_cef,
            "fund_deaths_fully_delisted": n_deaths,
            "of_which_bdc": n_bdc,
            "of_which_cef": n_deaths - n_bdc,
        },
        "outcome_distribution": dict(outcomes.most_common()),
        "nav_event": {
            "nav_event_true": nav_true,
            "unknown_shown_not_guessed": nav_unknown,
            "distress_non_nav": nav_false,
            "frac_nav_event_of_deaths": frac_nav,
        },
        "deaths": sorted(deaths, key=lambda d: d["delist_date"] or ""),
        "conservative_direction_verdict": _verdict(n_deaths, frac_nav, nav_unknown, nav_false),
    }

    os.makedirs("results", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(census, f, indent=2)

    print("\n=== H6 DEAD-FUND CENSUS ===")
    print(f"  window {census['window']}: {n_form25} delistings -> "
          f"{n_cef} CEF/BDC filers -> {n_deaths} fund deaths "
          f"({n_deaths - n_bdc} CEF, {n_bdc} BDC)")
    print(f"  outcomes: {census['outcome_distribution']}")
    print(f"  NAV-event: {nav_true} yes / {nav_unknown} unknown / {nav_false} distress "
          f"(frac NAV = {frac_nav})")
    print(f"\n  VERDICT: {census['conservative_direction_verdict']}")
    print(f"  wrote {OUT}")


def _verdict(n, frac_nav, unknown, distress) -> str:
    if n == 0:
        return ("NO dead CEFs enumerated — re-check the source/window before any "
                "conclusion. Gate NOT cleared.")
    if distress == 0 and frac_nav is not None and frac_nav >= 0.8:
        return (f"CONSERVATIVE DIRECTION HOLDS: {frac_nav:.0%} of {n} dead CEFs are "
                f"NAV events (liquidation/merger/open-end/term), ZERO distress "
                f"delistings found, {unknown} unknown (bounded residual). Omitting "
                f"dead funds biases a discount-long backtest AGAINST itself — the "
                f"H6 survivorship gate CLEARS on direction. Residual: confirm the "
                f"{unknown} unknowns + any BDCs by filing text before Stage-2.")
    return (f"MIXED/INCONCLUSIVE: {frac_nav} NAV-event fraction, {distress} distress, "
            f"{unknown} unknown of {n}. Conservative-direction claim NOT established; "
            f"read filing text for the non-NAV/unknown cases before any Stage-2 call.")


if __name__ == "__main__":
    a = sys.argv
    sy = int(a[1]) if len(a) > 1 else 2019
    ey = int(a[2]) if len(a) > 2 else 2026
    main(sy, ey)
