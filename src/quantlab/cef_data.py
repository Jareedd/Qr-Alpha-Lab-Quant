"""Closed-end fund (H6) data loader — CEFConnect public API.

Endpoints discovered/verified 2026-06-14 (local compute, no API spend):
  * UNIVERSE:    GET /api/v3/DailyPricing
                 -> list of ~362 US CEFs with Ticker, Discount,
                    DistributionRatePrice/NAV, DistributionAmtUSD, CategoryName.
  * PER-FUND:    GET /api/v3/pricinghistory/{ticker}/{range}   (range MAX ~ 8yr)
                 -> {"Data":[{NAVData, Data(=price), DiscountData, DataDate}, ...]}.
  * DISTRIBUTIONS: endpoint TBD (needed for Stage-2 total return; DailyPricing
    carries the current distribution RATE/amount, but not the ex-date history).

Caches to data_cache/cef/ (gitignored — CEFConnect data is licensed and stays
local, never committed; same rule as the perp dumps). Parsing is pinned by a
known-answer test on a synthetic fixture (tests/test_cef_data.py) so it does not
depend on the licensed cache.

Stage discipline: this is Stage-1 plumbing (universe + price/NAV/discount). It
computes NO signal-vs-forward-return; that is Stage-2 (the registered trial).
"""
from __future__ import annotations

import json
import os
import urllib.request

import pandas as pd

BASE = "https://www.cefconnect.com/api/v3"
CACHE = os.path.join("data_cache", "cef")
_UA = {"User-Agent": "Mozilla/5.0 qr-alpha-lab academic research", "Accept": "application/json"}


def _get(url: str, timeout: int = 30) -> dict | list:
    with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=timeout) as r:
        return json.loads(r.read())


def parse_price_history(payload: dict) -> pd.DataFrame:
    """Parse a pricinghistory payload into a clean daily frame.

    Columns: ``px`` (market price), ``nav``, ``disc`` (premium/discount %),
    indexed by date (ascending, de-duplicated). Pure function — no network — so
    it is unit-tested on a synthetic fixture."""
    rows = payload.get("Data", []) if isinstance(payload, dict) else []
    if not rows:
        return pd.DataFrame(columns=["px", "nav", "disc"])
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "px": pd.to_numeric(df["Data"], errors="coerce"),
        "nav": pd.to_numeric(df["NAVData"], errors="coerce"),
        "disc": pd.to_numeric(df["DiscountData"], errors="coerce"),
    })
    out.index = pd.to_datetime(df["DataDate"])
    return out[~out.index.duplicated(keep="last")].sort_index()


def universe(use_cache: bool = True) -> pd.DataFrame:
    """The CEF universe snapshot (DailyPricing): one row per fund with discount,
    distribution rate/amount, and Morningstar category. Cached daily."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "universe_dailypricing.json")
    if use_cache and os.path.exists(path):
        payload = json.load(open(path, encoding="utf-8"))
    else:
        payload = _get(f"{BASE}/DailyPricing")
        json.dump(payload, open(path, "w", encoding="utf-8"))
    return pd.DataFrame(payload)


def price_history(ticker: str, rng: str = "MAX", use_cache: bool = True) -> pd.DataFrame:
    """Daily price/NAV/discount history for one fund (cached per ticker+range)."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"ph_{ticker}_{rng}.json")
    if use_cache and os.path.exists(path):
        payload = json.load(open(path, encoding="utf-8"))
    else:
        payload = _get(f"{BASE}/pricinghistory/{ticker}/{rng}")
        json.dump(payload, open(path, "w", encoding="utf-8"))
    return parse_price_history(payload)
