"""SEC XBRL fundamentals layer for H1 (quality tilts) — filing-date point-in-time.

Free and authoritative: SEC's XBRL company-facts API. The hard part of H1 is PIT
safety — a fundamental may only enter the universe by its FILING date (when the
market could first know it), never its period END. Every series here is indexed
by ``filed``.

Logical fields assembled (with tag fallbacks, since issuers tag inconsistently):
Assets, NetIncome, CFO, Revenue, CostOfGoodsSold → features GP/A and accruals/A
(in quantlab.fundamentals).

**KNOWN LIMITATION, measured in the H1 audit (research_log 2026-06-14):** SEC's
``company_tickers.json`` map is CURRENT-ONLY, so dead/renamed names are unmapped
(~73% coverage) — the survivorship hole, reprised. A survivorship-SAFE H1 run
therefore needs a historical ticker→CIK map or a paid source (Compustat/CRSP).
This module exposes a ``FundamentalsSource`` interface with a free-SEC
implementation (research/dev, survivorship-limited) and a ``CompustatSource``
adapter SLOT for when WRDS access lands. The H1 runner refuses a real trial on
the free source BY DESIGN — building the machine now so it is one command from a
clean trial #12 the day institutional data arrives.
"""
from __future__ import annotations

import gzip
import json
import os
import socket
import time
import urllib.error
import urllib.request

import pandas as pd

socket.setdefaulttimeout(120)
_UA = "qr-alpha-lab research Jared@how.co"          # SEC fair-access: real contact
CACHE = os.path.join("data_cache", "fundamentals")
_MIN_INTERVAL = 0.12
_last = [0.0]

# Logical field -> ordered us-gaap tag candidates (first that has data wins).
FIELD_TAGS = {
    "assets": ["Assets"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues", "SalesRevenueNet"],
    "cogs": ["CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfRevenue"],
    "gross_profit": ["GrossProfit"],
}
PERIODIC_FORMS = ("10-K", "10-Q", "10-K/A", "10-Q/A")
ANNUAL_FORMS = ("10-K", "10-K/A")


def _get(url: str, timeout: int = 60, retries: int = 4) -> bytes:
    import time as _t
    req = urllib.request.Request(
        url, headers={"User-Agent": _UA, "Accept-Encoding": "gzip, deflate"})
    for attempt in range(retries):
        wait = _MIN_INTERVAL - (_t.monotonic() - _last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
            _last[0] = _t.monotonic()
            return raw
        except urllib.error.HTTPError as e:
            _last[0] = _t.monotonic()
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
        except urllib.error.URLError:
            _last[0] = _t.monotonic()
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------- #
# Pure parsers (no network) — pinned by tests.
# --------------------------------------------------------------------------- #

def parse_company_concept_frame(
    payload: dict, forms: tuple[str, ...] = PERIODIC_FORMS,
) -> pd.DataFrame:
    """SEC ``companyconcept`` JSON → filing-date-indexed rows with ``value``,
    ``form``, and ``end`` columns.

    The free SEC path needs this metadata so H1 can request ANNUAL-ONLY flow
    numerators (10-K/10-KA) while leaving assets as every known stock value.
    When a filing date carries several period values, keep the row with the
    latest period ``end`` — the freshest figure disclosed in that filing.
    """
    units = (payload or {}).get("units", {}) or {}
    rows = units.get("USD") or []
    recs = []
    for r in rows:
        if r.get("form") in forms and r.get("filed") and r.get("val") is not None:
            recs.append(
                (r["filed"], r.get("end", ""), r["form"], float(r["val"]))
            )
    if not recs:
        return pd.DataFrame(columns=["value", "form", "end"])
    df = pd.DataFrame(
        recs, columns=["filed", "end", "form", "value"]
    ).sort_values(["filed", "end"])
    df = df.drop_duplicates("filed", keep="last")
    out = df.set_index(pd.to_datetime(df["filed"]))[["value", "form", "end"]]
    out.index.name = "filed"
    return out[~out.index.duplicated(keep="last")].sort_index()


def parse_company_concept(
    payload: dict, forms: tuple[str, ...] = PERIODIC_FORMS,
) -> pd.Series:
    """SEC ``companyconcept`` JSON → filing-date-indexed USD values.

    Kept as a Series wrapper for callers/tests that only need the values.
    """
    frame = parse_company_concept_frame(payload, forms=forms)
    if frame.empty:
        return pd.Series(dtype=float, name="value")
    out = frame["value"].copy()
    out.index.name = "filed"
    return out


def parse_ticker_cik_map(payload: dict) -> dict[str, str]:
    """SEC ``company_tickers.json`` → {TICKER: zero-padded-CIK}. CURRENT-ONLY —
    the survivorship limitation documented above."""
    out = {}
    for rec in (payload or {}).values():
        t, cik = rec.get("ticker"), rec.get("cik_str")
        if t and cik is not None:
            out[t.upper()] = str(int(cik)).zfill(10)
    return out


# --------------------------------------------------------------------------- #
# Source interface + implementations.
# --------------------------------------------------------------------------- #

class FundamentalsSource:
    """Interface the H1 harness consumes. ``field_series(ticker, field)`` returns
    a filing-date-indexed Series for a logical field in FIELD_TAGS."""

    survivorship_safe: bool = False

    def field_series(
        self, ticker: str, field: str, *, annual_only: bool = False
    ) -> pd.Series:  # pragma: no cover
        raise NotImplementedError


class FreeSECSource(FundamentalsSource):
    """Free SEC XBRL. Research/dev only — **survivorship-limited** (current-only
    ticker→CIK map; ~73% coverage per the audit). NOT valid for a graded trial."""

    survivorship_safe = False

    def __init__(self, cache_dir: str = CACHE):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._map: dict[str, str] | None = None

    def ticker_cik(self, ticker: str) -> str | None:
        if self._map is None:
            path = os.path.join(self.cache_dir, "company_tickers.json")
            if os.path.exists(path):
                payload = json.loads(open(path, encoding="utf-8").read())
            else:
                payload = json.loads(_get("https://www.sec.gov/files/company_tickers.json"))
                open(path, "w", encoding="utf-8").write(json.dumps(payload))
            self._map = parse_ticker_cik_map(payload)
        return self._map.get(ticker.upper())

    def _concept_frame(self, cik: str, tag: str) -> pd.DataFrame:
        path = os.path.join(self.cache_dir, f"cc_{cik}_{tag}.parquet")
        if os.path.exists(path):
            cached = pd.read_parquet(path)
            if {"value", "form", "end"}.issubset(cached.columns):
                cached.index.name = "filed"
                return cached[["value", "form", "end"]].sort_index()
        url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
        try:
            frame = parse_company_concept_frame(json.loads(_get(url)))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                frame = pd.DataFrame(columns=["value", "form", "end"])
            else:
                raise
        frame.to_parquet(path)
        return frame

    def field_series(
        self, ticker: str, field: str, *, annual_only: bool = False
    ) -> pd.Series:
        cik = self.ticker_cik(ticker)
        if cik is None:
            return pd.Series(dtype=float, name="value")     # unmapped (the hole)
        forms = ANNUAL_FORMS if annual_only else PERIODIC_FORMS
        for tag in FIELD_TAGS[field]:
            frame = self._concept_frame(cik, tag)
            if frame.empty:
                s = pd.Series(dtype=float, name="value")
            else:
                filtered = frame[frame["form"].isin(forms)]
                s = filtered["value"].copy()
                s.index.name = "filed"
            if not s.empty:
                return s
        return pd.Series(dtype=float, name="value")


class CompustatSource(FundamentalsSource):
    """Adapter SLOT for Compustat/CRSP via WRDS — survivorship-safe (dead names
    retained, point-in-time). Implement ``field_series`` against the WRDS pull
    when access lands; the harness then runs unchanged. Until then it refuses
    loudly so no one mistakes the free source for the real one."""

    survivorship_safe = True

    def field_series(
        self, ticker: str, field: str, *, annual_only: bool = False
    ) -> pd.Series:
        raise NotImplementedError(
            "CompustatSource is a slot — connect WRDS/Compustat (filing-date-PIT "
            "fundamentals + delisting-inclusive prices) here. See "
            "writeup/preregistered_hypotheses.md H1.")
