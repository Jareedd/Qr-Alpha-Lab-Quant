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
# Retained for back-compat (callers/tests that key off the bare us-gaap tag list).
# The resolution machinery now reads FIELD_CONCEPTS below, which carries the
# (namespace, tag, unit) each tag actually lives under on SEC's API.
FIELD_TAGS = {
    "assets": ["Assets"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues", "SalesRevenueNet"],
    "cogs": ["CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfRevenue"],
    "gross_profit": ["GrossProfit"],
    # Shares outstanding for value-weighting (market_cap = price * shares). The
    # us-gaap share tags; see FIELD_CONCEPTS for the namespace+unit they require
    # and for the dei primary that actually survives for dead names.
    "shares": ["CommonStockSharesOutstanding",
               "WeightedAverageNumberOfSharesOutstandingBasic",
               "CommonStockSharesIssued"],
}

# Logical field -> ordered (namespace, tag, unit) candidates (first with data
# wins). This is the source of truth the readers iterate. WHY a richer config:
#
#   * Most fundamentals are us-gaap monetary concepts reported under XBRL unit
#     "USD" — so the default mapping is just (us-gaap, tag, USD), byte-identical
#     to the historic behavior.
#   * SHARES OUTSTANDING is the exception twice over. (1) It is reported under the
#     XBRL unit ``shares``, not ``USD`` — reading units["USD"] returns EMPTY for
#     every share tag (root cause of the H1 "0 names" market-cap coverage bug).
#     (2) The reliable, dead-name-surviving concept is dei/
#     EntityCommonStockSharesOutstanding (returns data for live AAPL/MSFT AND for
#     delisted names like CELG); the us-gaap CommonStockShares* tags exist for
#     live names but 404 for many dead ones. So shares resolves dei FIRST, with
#     us-gaap shares-unit fallbacks.
#
# Default fields are derived from FIELD_TAGS to keep the two in lockstep; only
# ``shares`` is overridden with its namespace/unit-aware candidate list.
FIELD_CONCEPTS: dict[str, list[tuple[str, str, str]]] = {
    field: [("us-gaap", tag, "USD") for tag in tags]
    for field, tags in FIELD_TAGS.items()
}
FIELD_CONCEPTS["shares"] = [
    ("dei", "EntityCommonStockSharesOutstanding", "shares"),  # primary; survives dead names
    ("us-gaap", "CommonStockSharesOutstanding", "shares"),    # live-name fallback
    ("us-gaap", "CommonStockSharesIssued", "shares"),         # coarse (gross of treasury)
]

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
    payload: dict, forms: tuple[str, ...] = PERIODIC_FORMS, unit: str = "USD",
) -> pd.DataFrame:
    """SEC ``companyconcept`` JSON → filing-date-indexed rows with ``value``,
    ``form``, and ``end`` columns.

    The free SEC path needs this metadata so H1 can request ANNUAL-ONLY flow
    numerators (10-K/10-KA) while leaving assets as every known stock value.
    When a filing date carries several period values, keep the row with the
    latest period ``end`` — the freshest figure disclosed in that filing.

    ``unit`` selects which XBRL unit bucket to read. Monetary concepts live under
    ``"USD"`` (the default, preserving historic behavior); SHARE counts live under
    ``"shares"`` — reading the wrong bucket returns EMPTY, the H1 market-cap bug.
    """
    units = (payload or {}).get("units", {}) or {}
    rows = units.get(unit) or []
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
    payload: dict, forms: tuple[str, ...] = PERIODIC_FORMS, unit: str = "USD",
) -> pd.Series:
    """SEC ``companyconcept`` JSON → filing-date-indexed values for ``unit``.

    Kept as a Series wrapper for callers/tests that only need the values.
    """
    frame = parse_company_concept_frame(payload, forms=forms, unit=unit)
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


def _read_concept_frame(
    freesec: "FreeSECSource", cik: str, tag: str, namespace: str, unit: str,
) -> pd.DataFrame:
    """Call ``freesec._concept_frame`` for one (namespace, tag, unit) candidate.

    The legacy us-gaap/USD path is called POSITIONALLY (``_concept_frame(cik, tag)``)
    so it stays byte-identical to the historic single-namespace reader — and so
    test doubles that stub ``_concept_frame`` with the old ``(self, cik, tag)``
    signature keep working. Non-default namespaces (e.g. ``dei`` for shares) pass
    the namespace/unit through as keywords. This is the single shared concept
    reader behind both FreeSECSource and SurvivorshipSafeSECSource.
    """
    if namespace == "us-gaap" and unit == "USD":
        return freesec._concept_frame(cik, tag)
    return freesec._concept_frame(cik, tag, namespace=namespace, unit=unit)


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

    def _concept_frame(
        self, cik: str, tag: str, namespace: str = "us-gaap", unit: str = "USD",
    ) -> pd.DataFrame:
        """Read one SEC ``companyconcept`` (namespace/tag), parsed from XBRL unit
        ``unit``, as a filing-date-indexed value/form/end frame. Cached to parquet.

        ``namespace`` selects the API taxonomy (``us-gaap`` default; ``dei`` for
        shares-outstanding, which only the dei concept carries for dead names).
        The us-gaap/USD path keeps its historic cache filename byte-for-byte;
        other namespaces get a namespace-prefixed filename so dei and us-gaap
        concepts of the same tag never collide in the cache.
        """
        # us-gaap keeps the legacy filename (cache continuity); others namespaced.
        stem = f"cc_{cik}_{tag}" if namespace == "us-gaap" else f"cc_{namespace}_{cik}_{tag}"
        path = os.path.join(self.cache_dir, f"{stem}.parquet")
        if os.path.exists(path):
            cached = pd.read_parquet(path)
            if {"value", "form", "end"}.issubset(cached.columns):
                cached.index.name = "filed"
                return cached[["value", "form", "end"]].sort_index()
        url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/{namespace}/{tag}.json"
        try:
            frame = parse_company_concept_frame(json.loads(_get(url)), unit=unit)
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
        for namespace, tag, unit in FIELD_CONCEPTS[field]:
            frame = _read_concept_frame(self, cik, tag, namespace, unit)
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
