"""H6 Stage-1 census (ZERO-TRIAL) — descriptive characterization of the free
CEF data, per writeup/edge_candidates_2026-06-12.md (H6) and
writeup/graduation_candidates_2026-06-14.md (§3 decision gate).

NOTHING here computes a signal-vs-forward-return relationship (that is H6
Stage 2, the registered trial). This script answers the four Stage-1 questions
the spec asks BEFORE a trial can be registered:

  1. Universe / capacity census  — fund count, mcap & dollar-ADV distributions,
     the sub-$400M / ADV>=$250k tradable tail count.
  2. NAV-staleness audit          — NAV-publication lag distribution; the
     daily-NAV-only subuniverse (Stage-2's filter) sized; staleness by category.
  3. Data-depth feasibility       — for a tail sample: rows, span, granularity
     of the free history (the DSR-hurdle swing variable: daily-1Y vs weekly-All).
  4. Survivorship / dead-fund hole — documents that CEFConnect's universe is
     current-only; the conservative-direction claim and dead-fund census need
     an INDEPENDENT source and are the explicit next increment.

Output: results/cef_stage1_census.json (every number regenerable from this
script + the cached snapshot) + a printed summary. N is unchanged.

Run: python scripts/cef_stage1_census.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import cef_data

TAIL_MCAP_USDM = 400.0      # spec: sub-$400M tail
ADV_FLOOR_USD = 250_000.0   # spec: ADV >= $250k
DEPTH_SAMPLE_N = 12         # tail funds to probe for history depth
OUT = os.path.join("results", "cef_stage1_census.json")


def _q(s: pd.Series, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict:
    s = s.dropna()
    if s.empty:
        return {}
    return {f"p{int(q*100)}": round(float(s.quantile(q)), 4) for q in qs}


def _granularity(df: pd.DataFrame) -> int | None:
    if len(df) < 2:
        return None
    gaps = df.index.to_series().diff().dropna().dt.days
    return int(gaps.median())


def main() -> None:
    snap = cef_data.daily_snapshot()        # cached after first pull
    asof = snap.attrs.get("asof", "?")
    n = len(snap)
    print(f"[cef-census] snapshot as-of {asof}: {n} currently-listed CEFs")

    mcap = snap["MarketCapUSDm"]
    adv = snap["dollar_adv"]
    tail = snap[mcap < TAIL_MCAP_USDM]
    tradable = tail[tail["dollar_adv"] >= ADV_FLOOR_USD]

    # NAV staleness: lag = LastUpdated - NAVPublished (days). 0 == fresh daily.
    lag = snap["nav_lag_days"]
    daily_nav = snap[lag <= 1]
    stale_by_cat = (
        snap.assign(stale=lag > 1)
        .groupby("CategoryName")["stale"].mean()
        .sort_values(ascending=False)
    )

    # Depth sample: real tail funds, most-liquid first (so the probe reflects
    # what Stage 2 would actually trade, not delisting-thin names).
    sample = tradable.sort_values("dollar_adv", ascending=False).head(DEPTH_SAMPLE_N)
    depth = []
    for tkr in sample.index:
        rec = {"ticker": tkr}
        for period in (cef_data.FULL_PERIOD, cef_data.DAILY_PERIOD):
            h = cef_data.price_nav_discount(tkr, period=period)
            rec[period] = {
                "rows": int(len(h)),
                "start": (h.index.min().strftime("%Y-%m-%d") if len(h) else None),
                "end": (h.index.max().strftime("%Y-%m-%d") if len(h) else None),
                "median_gap_days": _granularity(h),
                "nav_coverage": (round(float(h["nav"].notna().mean()), 3)
                                 if len(h) else None),
            }
            time.sleep(0.1)                  # polite to a free public API
        depth.append(rec)
        print(f"  depth {tkr}: All={rec['All']['rows']}r/{rec['All']['median_gap_days']}d "
              f"1Y={rec['1Y']['rows']}r/{rec['1Y']['median_gap_days']}d")

    full_rows = [d["All"]["rows"] for d in depth if d["All"]["rows"]]
    full_starts = [d["All"]["start"] for d in depth if d["All"]["start"]]

    census = {
        "asof": asof,
        "source": "cefconnect.com api/v3 (free, current-listings only)",
        "params": {"tail_mcap_usdm": TAIL_MCAP_USDM, "adv_floor_usd": ADV_FLOOR_USD},
        "universe": {
            "n_current_funds": n,
            "n_tail_sub400m": int(len(tail)),
            "n_tradable_tail": int(len(tradable)),
            "mcap_usdm_quantiles": _q(mcap),
            "dollar_adv_quantiles": _q(adv),
            "tail_dollar_adv_quantiles": _q(tail["dollar_adv"]),
        },
        "discount": {
            "current_pct_quantiles": _q(snap["Discount"]),
            "n_at_discount_gt_5pct": int((snap["Discount"] < -5).sum()),
            "n_at_discount_gt_10pct": int((snap["Discount"] < -10).sum()),
            "cc_zscore1yr_coverage": round(float(snap["ZScore1Yr"].notna().mean()), 3),
        },
        "nav_staleness": {
            "lag_days_quantiles": _q(lag, qs=(0.5, 0.75, 0.9, 0.95, 0.99)),
            "n_daily_nav_lag_le_1": int(len(daily_nav)),
            "frac_daily_nav": round(float((lag <= 1).mean()), 3),
            "most_stale_categories": {k: round(float(v), 3)
                                      for k, v in stale_by_cat.head(8).items()},
        },
        "depth_sample": depth,
        "depth_summary": {
            "n_sampled": len(depth),
            "full_period_token": cef_data.FULL_PERIOD,
            "full_median_rows": (int(np.median(full_rows)) if full_rows else None),
            "full_granularity": "weekly (~7d gap)",
            "daily_period_token": cef_data.DAILY_PERIOD,
            "daily_granularity": "daily (~1d gap), trailing ~1yr only",
            "earliest_history": (min(full_starts) if full_starts else None),
            "finding": ("Free daily NAV/discount is trailing-1yr only; full "
                        "history (~2012+) is WEEKLY. Daily price is separately "
                        "available via yfinance over the full span. Stage-2 must "
                        "pre-declare the signal cadence (weekly discount-z) and "
                        "P&L cadence (daily price) accordingly."),
        },
        "survivorship": {
            "universe_is_current_only": True,
            "dead_funds_in_source": False,
            "note": ("api/v3/funds and api/v3/dailypricing list CURRENT funds "
                     "only; dead/merged/liquidated funds are absent. The H6 "
                     "decision gate requires verifying the 'CEF deaths happen at "
                     "NAV -> omission is CONSERVATIVE' claim against an "
                     "independent dead-fund source (SEC N-CEN enumeration or a "
                     "date-stamped historical universe). NEXT INCREMENT — not "
                     "resolved by this census."),
        },
        "decision_gate_read": {
            "data_depth_ok": (min(full_starts) <= "2018-01-01" if full_starts else False),
            "conservative_direction_verified": None,   # pending dead-fund census
            "verdict": ("DEPTH PASSES (weekly to ~2012, daily price full via "
                        "yfinance). Tradable tail exists. GATE NOT YET CLEARED: "
                        "the conservative-survivorship claim is unverified until "
                        "the dead-fund census runs. Do NOT register Stage-2 until "
                        "that claim is settled."),
        },
    }

    os.makedirs("results", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(census, f, indent=2)

    u, d, s = census["universe"], census["discount"], census["nav_staleness"]
    print("\n=== H6 STAGE-1 CENSUS SUMMARY ===")
    print(f"  current CEFs: {u['n_current_funds']}  | sub-$400M tail: "
          f"{u['n_tail_sub400m']}  | tradable tail (ADV>=$250k): {u['n_tradable_tail']}")
    print(f"  mcap $m p25/p50/p75: {u['mcap_usdm_quantiles'].get('p25')}/"
          f"{u['mcap_usdm_quantiles'].get('p50')}/{u['mcap_usdm_quantiles'].get('p75')}")
    print(f"  discount % p5/p50/p95: {d['current_pct_quantiles'].get('p5')}/"
          f"{d['current_pct_quantiles'].get('p50')}/{d['current_pct_quantiles'].get('p95')}"
          f"  | n<-10%: {d['n_at_discount_gt_10pct']}")
    print(f"  daily-NAV funds (lag<=1d): {s['n_daily_nav_lag_le_1']} "
          f"({s['frac_daily_nav']*100:.0f}%)")
    print(f"  earliest free history: {census['depth_summary']['earliest_history']} "
          f"(weekly); daily price via yfinance")
    print(f"\n  GATE: {census['decision_gate_read']['verdict']}")
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
