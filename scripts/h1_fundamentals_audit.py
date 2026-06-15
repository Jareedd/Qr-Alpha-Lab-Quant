"""H1 (quality fundamentals) — PIT DATA AUDIT. Zero trials.

H1's registration is explicitly blocked until "the data question is answered
honestly; that audit is its own session." This is that session. It asks, of
free SEC XBRL data on the S&P universe, the only questions that decide whether
H1 can run point-in-time-safely:

  1. COVERAGE: can we compute GP/Assets (gross profitability, Novy-Marx) and
     total accruals/Assets (Sloan, cash-flow definition NI - CFO) for enough of
     the universe?
  2. FILING-DATE LAG: every fundamental MUST be lagged by FILING date, not
     period end (using a number before it was public is look-ahead). Does XBRL
     give the `filed` date, and what is the lag distribution (the >=3-month
     floor the registration assumes)?

It computes NO signal-vs-forward-return relationship -- it only measures data
availability and timing. Mirrors H6's Stage-1 census / H8's power gate.

Run:  PYTHONPATH=src .venv/Scripts/python.exe scripts/h1_fundamentals_audit.py
"""
from __future__ import annotations

import json
import time
import urllib.request

import pandas as pd

UA = {"User-Agent": "qr-alpha-lab academic research Jared@how.co"}
FRAMES = "https://data.sec.gov/api/xbrl/frames/us-gaap/{concept}/USD/{period}.json"
FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
TICKERS = "https://www.sec.gov/files/company_tickers.json"


def _get(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
            return json.load(r)
    except Exception:
        return None


def universe_ciks() -> tuple[dict, set]:
    """ticker->CIK for the PIT S&P universe (current + ever-removed names)."""
    cur = pd.read_parquet("data_cache/sp500_current.parquet")
    chg = pd.read_parquet("data_cache/sp500_changes.parquet")
    tickers = set(cur["ticker"]) | set(chg["added"].dropna()) | set(chg["removed"].dropna())
    tickers = {t for t in tickers if isinstance(t, str) and t not in ("None", "")}
    t2c = {}
    raw = _get(TICKERS) or {}
    for row in raw.values():
        t2c[row["ticker"].replace(".", "-")] = int(row["cik_str"])
    mapped = {t: t2c[t] for t in tickers if t in t2c}
    return mapped, tickers


def frame_ciks(concept: str, period: str) -> dict:
    """{cik: val} for one concept/period frame (the cross-section), or {}."""
    d = _get(FRAMES.format(concept=concept, period=period))
    if not d or "data" not in d:
        return {}
    return {int(e["cik"]): e["val"] for e in d["data"]}


def main() -> None:
    mapped, all_tickers = universe_ciks()
    uni_ciks = set(mapped.values())
    print(f"[universe] {len(all_tickers)} PIT tickers; "
          f"{len(mapped)} mapped to a CIK ({len(mapped)/max(len(all_tickers),1)*100:.0f}%)")

    # --- Part A: coverage via frames (union of CY2023 & CY2024 to absorb
    # off-calendar fiscal years, which sit in different frames) ---
    concepts_dur = ["GrossProfitLoss", "Revenues",
                    "RevenueFromContractWithCustomerExcludingAssessedTax",
                    "CostOfGoodsAndServicesSold", "CostOfRevenue",
                    "NetIncomeLoss", "NetCashProvidedByUsedInOperatingActivities"]
    cover = {c: set() for c in concepts_dur}
    assets = set()
    for yr in ("CY2023", "CY2024"):
        for c in concepts_dur:
            ciks = set(frame_ciks(c, yr)) & uni_ciks
            cover[c] |= ciks
            time.sleep(0.12)
        for q in ("Q4I", "Q2I"):  # instantaneous balance-sheet
            assets |= set(frame_ciks("Assets", f"{yr}{q}")) & uni_ciks
            time.sleep(0.12)

    n = len(uni_ciks)
    def pct(s):
        return f"{len(s)}/{n} ({len(s)/n*100:.0f}%)"
    print("\n=== COVERAGE on the mapped universe (union CY2023/24) ===")
    print(f"  Assets (denominator):        {pct(assets)}")
    for c in concepts_dur:
        print(f"  {c:<48} {pct(cover[c])}")

    gp_direct = cover["GrossProfitLoss"]
    rev = cover["Revenues"] | cover["RevenueFromContractWithCustomerExcludingAssessedTax"]
    cogs = cover["CostOfGoodsAndServicesSold"] | cover["CostOfRevenue"]
    gp_computable = (gp_direct | (rev & cogs)) & assets
    accruals_computable = cover["NetIncomeLoss"] & cover["NetCashProvidedByUsedInOperatingActivities"] & assets
    both = gp_computable & accruals_computable
    print("\n=== DERIVED FEATURE COMPUTABILITY ===")
    print(f"  GP/Assets computable (GP direct OR Rev&CoGS): {pct(gp_computable)}")
    print(f"  Accruals/Assets computable (NI & CFO):        {pct(accruals_computable)}")
    print(f"  BOTH features computable:                     {pct(both)}")

    # --- Part B: filing-LAG distribution via companyfacts on a sample ---
    print("\n=== FILING-DATE LAG (filed - period_end), Assets, sample of 30 ===")
    lags = []
    sample = list(uni_ciks)[:30]
    for cik in sample:
        d = _get(FACTS.format(cik=cik))
        time.sleep(0.12)
        if not d:
            continue
        try:
            units = d["facts"]["us-gaap"]["Assets"]["units"]["USD"]
        except KeyError:
            continue
        for f in units:
            if f.get("form") in ("10-K", "10-Q") and f.get("filed") and f.get("end"):
                lag = (pd.Timestamp(f["filed"]) - pd.Timestamp(f["end"])).days
                if 0 < lag < 400:
                    lags.append(lag)
    if lags:
        s = pd.Series(lags)
        print(f"  n={len(s)} fact-filings; lag days: "
              f"min {s.min()}, p10 {s.quantile(.1):.0f}, median {s.median():.0f}, "
              f"p90 {s.quantile(.9):.0f}, max {s.max()}")
        print(f"  share with lag >= 60d: {(s >= 60).mean()*100:.0f}%  "
              f"(the registration assumes a >=3-month/~90d floor)")
    else:
        print("  no filing-lag data retrieved")

    print("\n=== AUDIT VERDICT ===")
    ok_cov = len(both) / n
    print(f"  feature coverage (both, of mapped universe): {ok_cov*100:.0f}%")
    print("  filing dates available in XBRL: "
          f"{'YES' if lags else 'NOT CONFIRMED'} -> PIT lag by `filed` is "
          f"{'feasible' if lags else 'unverified'}")
    print("  NOTE: this audits CURRENT mapped names; dead/renamed tickers and "
          "off-calendar filers lower real coverage. Coverage on the FULL PIT "
          "set (incl. unmapped/dead) is the binding number for the trial.")


if __name__ == "__main__":
    main()
