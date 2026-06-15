"""CEFConnect data layer for H6 (closed-end-fund discount reversion) — free
public API only, mirroring perp_data.py's conventions (stdlib urllib, parquet
cache, pure parsers split out so they are pinned by offline known-answer tests).

Source: https://www.cefconnect.com (Nuveen's public CEF data site). Surface
verified reachable 2026-06-15:

- ``api/v3/funds``            -> [{Ticker, Name}, ...] for CURRENTLY-LISTED funds
                                only (363). Dead/merged/liquidated funds are NOT
                                retained here -- the survivorship question the H6
                                Stage-1 dead-fund census must answer from an
                                independent source (this layer documents the
                                hole; it cannot fill it).
- ``api/v3/dailypricing``    -> rich CURRENT snapshot per fund (mcap, total
                                assets, ADV, NAV-published date, distribution
                                frequency, expense ratio, leverage, category,
                                and CEFConnect's own discount z-scores). This is
                                a COLLECTION-FORWARD dataset: the snapshot at t
                                cannot be reconstructed at t+1, so each pull is
                                cached write-once by its own as-of date.
- ``api/v3/pricinghistory/{ticker}/{period}`` -> per-fund price (``Data``), NAV
                                (``NAVData``) and discount (``DiscountData``,
                                already (P-NAV)/NAV in %, sign-matching
                                quantlab.cef.discount). GRANULARITY DEPENDS ON
                                PERIOD: ``1Y`` is DAILY (~246 rows); ``All`` is
                                WEEKLY back to ~2012 (~700 rows). There is no
                                free daily-NAV path beyond the trailing year --
                                a Stage-1 finding that constrains the Stage-2
                                design, not this module.

Nothing here computes a signal or a forward-return relationship. That is H6
Stage 2 (the registered trial). This is data assembly + descriptive primitives.
"""
from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request

import pandas as pd

# Match perp_data.py: no socket op blocks forever (a sleeping machine froze a
# download run for hours otherwise).
socket.setdefaulttimeout(120)

_UA = "Mozilla/5.0 (qr-alpha-lab research)"
BASE = "https://www.cefconnect.com/api/v3"
CACHE = os.path.join("data_cache", "cef")

# pricinghistory period tokens, verified 2026-06-15. Only these return data;
# "Max"/"10Y"/numeric tokens return an empty PriceHistory (recognised ticker,
# no rows) -- a silent trap if you assume "Max" means "all history".
DAILY_PERIOD = "1Y"     # ~246 daily rows (median gap 1d)
FULL_PERIOD = "All"     # ~700 weekly rows (median gap 7d), back to ~2012


def _get_json(url: str, timeout: int = 60):
    req = urllib.request.Request(
        url, headers={"User-Agent": _UA, "Accept": "application/json,*/*"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# --------------------------------------------------------------------------- #
# Pure parsers (no network) -- pinned by tests/test_cef_data.py.
# --------------------------------------------------------------------------- #

# Snapshot fields we type as numeric; everything else is kept as-is (strings,
# bools, ids). Names match the live api/v3/dailypricing schema exactly.
_SNAPSHOT_NUMERIC = (
    "Discount", "Price", "NAV", "MarketCapUSDm", "TotalAssetsUSDm",
    "AvgDailyVolume", "ExpenseRatio", "LeverageRatioPercentage",
    "DistributionRateNAV", "DistributionRatePrice", "DistributionAmtUSD",
    "ZScore1Yr", "ZScore3M", "ZScore6M", "Discount52WkAvg", "Price52WkAvg",
)
_SNAPSHOT_DATES = ("NAVPublished", "LastUpdated", "InceptionDate", "ZScoreDate")


def parse_snapshot(records: list[dict]) -> pd.DataFrame:
    """api/v3/dailypricing list -> DataFrame indexed by Ticker.

    Numeric fields coerced to float (CEFConnect sends them as numbers, but a
    missing field arrives as null -> NaN, never a string); date fields parsed.
    Adds ``dollar_adv`` = AvgDailyVolume * Price (the spec's ADV floor is in
    dollars) and ``nav_lag_days`` = LastUpdated - NAVPublished (the staleness
    primitive). Raises on an empty payload rather than returning a silent
    empty frame."""
    if not records:
        raise ValueError("empty snapshot payload")
    df = pd.DataFrame(records)
    if "Ticker" not in df.columns:
        raise ValueError(f"snapshot missing Ticker; got {list(df.columns)[:6]}")
    for c in _SNAPSHOT_NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in _SNAPSHOT_DATES:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df = df.set_index("Ticker").sort_index()
    if {"AvgDailyVolume", "Price"} <= set(df.columns):
        df["dollar_adv"] = df["AvgDailyVolume"] * df["Price"]
    if {"LastUpdated", "NAVPublished"} <= set(df.columns):
        df["nav_lag_days"] = (df["LastUpdated"] - df["NAVPublished"]).dt.days
    return df


def parse_pricinghistory(payload: dict) -> pd.DataFrame:
    """api/v3/pricinghistory JSON -> date-indexed frame [price, nav, discount_cc].

    ``discount_cc`` is CEFConnect's own (P-NAV)/NAV in PERCENT (a cross-check on
    quantlab.cef.discount, which we compute from price+nav at panel time).
    Returns an EMPTY frame (not an error) when PriceHistory is [] -- that is the
    legitimate answer for an unrecognised period token or a fund with no served
    history, and callers (the depth sample, the panel builder) must handle it."""
    ph = ((payload or {}).get("Data") or {}).get("PriceHistory") or []
    cols = ["price", "nav", "discount_cc"]
    if not ph:
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="date"))
    df = pd.DataFrame(ph)
    out = pd.DataFrame({
        "price": pd.to_numeric(df["Data"], errors="coerce"),
        "nav": pd.to_numeric(df["NAVData"], errors="coerce"),
        "discount_cc": pd.to_numeric(df["DiscountData"], errors="coerce"),
    })
    out.index = pd.to_datetime(df["DataDate"]).dt.normalize()
    out.index.name = "date"
    return out[~out.index.duplicated(keep="last")].sort_index()


# --------------------------------------------------------------------------- #
# Network fetchers (cached).
# --------------------------------------------------------------------------- #

def current_universe(cache_dir: str = CACHE, refresh: bool = False) -> pd.DataFrame:
    """Currently-listed CEFs [Ticker, Name] from api/v3/funds. Cached JSON.

    CURRENT LISTINGS ONLY -- this is the survivorship-biased universe; the
    dead-fund census is a separate, independent-source job."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "funds.json")
    if os.path.exists(path) and not refresh:
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
    else:
        records = _get_json(f"{BASE}/funds")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f)
    return pd.DataFrame(records).set_index("Ticker").sort_index()


def daily_snapshot(cache_dir: str = CACHE, refresh: bool = False) -> pd.DataFrame:
    """Rich current snapshot (api/v3/dailypricing), parsed and cached parquet by
    its OWN as-of date (max LastUpdated). Collection-forward: re-pulling on a
    later day yields a different file, never overwrites an existing as-of."""
    os.makedirs(cache_dir, exist_ok=True)
    # Probe-free cache reuse: if any snapshot parquet exists and not refreshing,
    # return the newest one without a network call.
    if not refresh:
        existing = sorted(f for f in os.listdir(cache_dir)
                          if f.startswith("snapshot_") and f.endswith(".parquet"))
        if existing:
            return pd.read_parquet(os.path.join(cache_dir, existing[-1]))
    df = parse_snapshot(_get_json(f"{BASE}/dailypricing"))
    asof = pd.to_datetime(df["LastUpdated"]).max()
    stamp = asof.strftime("%Y-%m-%d") if pd.notna(asof) else "unknown"
    df.attrs["asof"] = stamp
    path = os.path.join(cache_dir, f"snapshot_{stamp}.parquet")
    if not os.path.exists(path):                      # write-once per as-of
        df.to_parquet(path)
    return df


def price_nav_discount(
    ticker: str, period: str = FULL_PERIOD, cache_dir: str = CACHE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Per-fund [price, nav, discount_cc] frame from pricinghistory. Cached
    parquet per (ticker, period). ``period='1Y'`` daily, ``'All'`` weekly.
    Empty frame if the fund/period serves no history (cached as such)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"hist_{ticker}_{period}.parquet")
    if os.path.exists(path) and not refresh:
        return pd.read_parquet(path)
    try:
        payload = _get_json(f"{BASE}/pricinghistory/{ticker}/{period}")
    except urllib.error.HTTPError as e:
        if e.code == 404:                             # unknown ticker
            df = parse_pricinghistory({})
            df.to_parquet(path)
            return df
        raise
    df = parse_pricinghistory(payload)
    df.to_parquet(path)
    return df


def build_weekly_panels(
    tickers: list[str], start: str = "2010-01-01", cache_dir: str = CACHE,
    refresh: bool = False, min_weeks: int = 26,
) -> dict:
    """Assemble the H6 weekly backtest panels for ``tickers``:

    - ``price``: distribution-INCLUSIVE total-return price (yfinance adjusted
      close, resampled to W-FRI) — the holder's actual total return.
    - ``discount``: (price-NAV)/NAV from CEFConnect ``pricinghistory/All``
      (weekly), aligned onto the Friday index PIT-safely (ffill within 10 days,
      so a fund's discount never carries past its last real observation).

    Both are (week x fund). Cached to parquet; ``refresh`` rebuilds. The weekly
    cadence is the registration's (free NAV history is weekly beyond ~1yr).
    """
    os.makedirs(cache_dir, exist_ok=True)
    p_path = os.path.join(cache_dir, "panels_weekly_price.parquet")
    d_path = os.path.join(cache_dir, "panels_weekly_discount.parquet")
    if os.path.exists(p_path) and os.path.exists(d_path) and not refresh:
        return {"price": pd.read_parquet(p_path),
                "discount": pd.read_parquet(d_path)}

    # 1. CEFConnect weekly discount per fund (cached per ticker).
    disc = {}
    for i, t in enumerate(tickers):
        h = price_nav_discount(t, period=FULL_PERIOD)
        if len(h) >= min_weeks and h["nav"].notna().sum() >= min_weeks:
            s = (h["price"] - h["nav"]) / h["nav"]
            disc[t] = s[s.index >= pd.Timestamp(start)]
        if (i + 1) % 40 == 0:
            print(f"  [cef panels] discount {i+1}/{len(tickers)}", flush=True)
    discount = pd.DataFrame(disc).sort_index()

    # 2. yfinance distribution-inclusive total-return price -> weekly (W-FRI).
    import yfinance as yf
    raw = yf.download(list(tickers), start=start, auto_adjust=True,
                      progress=False, threads=True)
    adj = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    price_w = adj.resample("W-FRI").last()

    # 3. align discount onto the Friday index, PIT-safe (asof within 10 days);
    #    build all columns at once (avoids a fragmented-frame insert loop).
    disc_w = pd.DataFrame(
        {t: discount[t].dropna().reindex(
            price_w.index, method="ffill", tolerance=pd.Timedelta("10D"))
         for t in discount.columns},
        index=price_w.index,
    )

    common = sorted(set(price_w.columns) & set(disc_w.columns))
    price_w, disc_w = price_w[common], disc_w[common]
    price_w.attrs["n_funds"] = len(common)
    price_w.to_parquet(p_path)
    disc_w.to_parquet(d_path)
    print(f"  [cef panels] built {len(common)} funds x {len(price_w)} weeks")
    return {"price": price_w, "discount": disc_w}
