"""SEC Form 4 (insider transactions) data layer for H10 — filing-date PIT.

Free and authoritative: SEC's submissions API lists every Form 4 a company filed
(with its ``filingDate``), and each accession carries a RAW ``ownershipDocument``
XML with the structured transaction detail. The H10 hypothesis (Lakonishok-Lee
2001; Cohen-Malloy-Pomorski 2012) is that OPPORTUNISTIC insider CLUSTER-BUYING of
a firm's own stock earns positive forward returns; the signal layer lives in
``quantlab.insider``. This module is JUST the data plumbing: enumerate Form 4s,
fetch + parse the raw XML, and surface open-market PURCHASES indexed by FILING
date (when the market could first know — Form 4 is due within 2 business days, so
the filing date is near-PIT).

THE XML SUBTLETY (verified). The submissions ``primaryDocument`` for a Form 4 is
the XSLT-RENDERED HTML (path like ``xslF345X06/...``) which has NO structured
tags. The structured fields live in the RAW ``ownershipDocument`` XML — a separate
``.xml`` in the same accession folder. We resolve it via the accession directory
listing (``index.json``), picking the ``.xml`` whose root element is
``<ownershipDocument>`` (NOT the xsl rendering), then parse with the stdlib XML
parser. Raw fields used:
  <issuer><issuerCik><issuerTradingSymbol>
  <reportingOwner><reportingOwnerId><rptOwnerName>
                  <reportingOwnerRelationship>(isDirector/isOfficer/
                  isTenPercentOwner/officerTitle)
  <nonDerivativeTable><nonDerivativeTransaction>:
      <transactionDate><value>
      <transactionCoding><transactionCode>   (P = open-market purchase, S = sale)
      <transactionAmounts><transactionShares><value>
                          <transactionPricePerShare><value>
                          <transactionAcquiredDisposedCode><value>  (A/D)
H10 cares about code ``P`` with acquired code ``A`` (open-market purchases).

Mirrors ``fundamentals_data.py``: the same ``_UA`` fair-access header, the same
rate-limited ``_get`` with retry/backoff, parquet caching, and a PURE parser
(``parse_form4_xml``) pinned by a known-answer test so the network is never
needed to validate the parse logic.
"""
from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd

socket.setdefaulttimeout(120)
_UA = "qr-alpha-lab research Jared@how.co"          # SEC fair-access: real contact
CACHE = os.path.join("data_cache", "insider")
_MIN_INTERVAL = 0.12
_last = [0.0]

_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
# Accession directory listing; {cik} is the integer CIK (no zero-pad in the path),
# {acc_nodash} is the accession number with dashes stripped.
_INDEX_URL = ("https://www.sec.gov/cgi-bin/browse-edgar")  # unused; kept for ref
_ARCHIVE_DIR = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/"


def _get(url: str, timeout: int = 60, retries: int = 6) -> bytes:
    """Rate-limited GET with retry/backoff — SEC fair-access <10 req/s, exponential
    backoff on 429/5xx AND on transient network errors.

    Resilience note: a socket-level read timeout surfaces as a bare ``TimeoutError``
    raised inside ``getresponse()`` — it is NOT a subclass of ``urllib.error.URLError``,
    so it must be caught explicitly or a SINGLE slow SEC response aborts an entire
    multi-thousand-filing fetch (observed live: the 40-name coverage probe died on
    name #8 on one read timeout). It is retried with the same exponential backoff as
    429/5xx; ``retries`` is 6 (was 4) for the long Form-4 crawls. Mirrors the
    resumable/resilient policy the Tiingo layer already adopted."""
    import time as _t
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    for attempt in range(retries):
        wait = _MIN_INTERVAL - (_t.monotonic() - _last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
            _last[0] = _t.monotonic()
            return raw
        except urllib.error.HTTPError as e:
            _last[0] = _t.monotonic()
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, socket.timeout):
            # URLError covers DNS/conn-refused/reset; TimeoutError/socket.timeout
            # covers a read timeout mid-response (NOT a URLError subclass).
            _last[0] = _t.monotonic()
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------- #
# Pure parser (no network) — pinned by tests.
# --------------------------------------------------------------------------- #

def _strip_ns(tag: str) -> str:
    """Drop any XML namespace prefix (``{ns}local`` -> ``local``). Form 4 XML is
    usually un-namespaced, but be robust if a filer wraps it."""
    return tag.rsplit("}", 1)[-1]


def _find(elem: ET.Element | None, path: list[str]) -> ET.Element | None:
    """Namespace-insensitive descend through ``path`` of child local-names."""
    cur = elem
    for name in path:
        if cur is None:
            return None
        nxt = None
        for child in cur:
            if _strip_ns(child.tag) == name:
                nxt = child
                break
        cur = nxt
    return cur


def _text(elem: ET.Element | None, path: list[str]) -> str | None:
    """Text of the descendant at ``path`` (stripped), or None if absent/empty."""
    node = _find(elem, path)
    if node is None or node.text is None:
        return None
    t = node.text.strip()
    return t or None


def _flag(elem: ET.Element | None, path: list[str]) -> bool:
    """A Form 4 boolean relationship flag: SEC encodes true as ``1`` or ``true``."""
    t = _text(elem, path)
    return str(t).strip().lower() in {"1", "true"} if t is not None else False


def _to_float(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_form4_xml(xml_str: str) -> list[dict]:
    """Parse a raw Form 4 ``<ownershipDocument>`` XML string into one dict per
    NON-DERIVATIVE transaction.

    Each dict carries:
        issuer_cik, ticker, owner_name, is_officer, is_director, is_tenpct,
        code, shares, price, acq_disp, transaction_date

    Robust to missing optional fields (price absent on some gifts/awards, ticker
    occasionally blank, multiple reporting owners). Returns ``[]`` for malformed
    input or a document with no non-derivative table — never raises on shape.

    PURE: no network, no I/O — the known-answer-tested core of the data layer.
    Derivative transactions (options) are intentionally NOT parsed: H10 keys on
    open-market common-stock purchases (code P), which are non-derivative.
    """
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return []

    issuer = _find(root, ["issuer"])
    issuer_cik = _text(issuer, ["issuerCik"])
    ticker = _text(issuer, ["issuerTradingSymbol"])

    # A filing may list several reporting owners; collapse to the first owner's
    # identity/role for the transaction rows (Form 4s are almost always single-
    # owner; when not, the first reportingOwner is the filer of record).
    owner = _find(root, ["reportingOwner"])
    owner_name = _text(owner, ["reportingOwnerId", "rptOwnerName"])
    rel = _find(owner, ["reportingOwnerRelationship"]) if owner is not None else None
    is_officer = _flag(rel, ["isOfficer"]) if rel is not None else False
    is_director = _flag(rel, ["isDirector"]) if rel is not None else False
    is_tenpct = _flag(rel, ["isTenPercentOwner"]) if rel is not None else False

    table = _find(root, ["nonDerivativeTable"])
    if table is None:
        return []

    rows: list[dict] = []
    for txn in table:
        if _strip_ns(txn.tag) != "nonDerivativeTransaction":
            continue
        amounts = _find(txn, ["transactionAmounts"])
        rows.append({
            "issuer_cik": issuer_cik,
            "ticker": ticker,
            "owner_name": owner_name,
            "is_officer": is_officer,
            "is_director": is_director,
            "is_tenpct": is_tenpct,
            "code": _text(txn, ["transactionCoding", "transactionCode"]),
            "shares": _to_float(_text(amounts, ["transactionShares", "value"])),
            "price": _to_float(
                _text(amounts, ["transactionPricePerShare", "value"])),
            "acq_disp": _text(amounts, ["transactionAcquiredDisposedCode", "value"]),
            "transaction_date": _text(txn, ["transactionDate", "value"]),
        })
    return rows


def is_ownership_xml(text: str) -> bool:
    """True if ``text`` is the RAW ownershipDocument XML (root ``ownershipDocument``)
    rather than the XSLT HTML rendering or some other accession artifact."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return False
    return _strip_ns(root.tag) == "ownershipDocument"


def parse_form4_accessions(submissions: dict) -> pd.DataFrame:
    """Submissions JSON -> DataFrame of Form 4 filings: columns
    ``[accession, filed]`` (filed = filingDate), filing-date-sorted.

    Reads the ``filings.recent`` arrays (form / accessionNumber / filingDate),
    keeping only ``form == "4"``. PURE (no network) so it is test-pinnable.
    """
    recent = (submissions or {}).get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accs = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    recs = [
        (acc, fd)
        for form, acc, fd in zip(forms, accs, dates)
        if form == "4" and acc and fd
    ]
    if not recs:
        return pd.DataFrame(columns=["accession", "filed"])
    df = pd.DataFrame(recs, columns=["accession", "filed"])
    df["filed"] = pd.to_datetime(df["filed"])
    return df.sort_values("filed").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Live source (network, rate-limited, parquet-cached).
# --------------------------------------------------------------------------- #

class InsiderSource:
    """Free SEC Form 4 reader. Enumerate a company's Form 4 accessions, fetch the
    raw ownershipDocument XML, and surface open-market PURCHASES indexed by FILING
    date. Mirrors ``FreeSECSource`` (same UA + rate limit + parquet cache).

    *** COMPLETENESS WARNING (2026-06-25) ***  This crawl is INCOMPLETE for prolific
    filers. ``list_form4_accessions`` reads ONLY the submissions ``filings.recent``
    block, which SEC caps at ~1000 filings; older filings live in the
    ``filings.files`` OVERFLOW pages, which this reader NEVER fetches. For an active
    issuer (e.g. AON: ``recent`` reaches back only to 2014-05-13, with 1,813 older
    filings in one overflow page) this silently DROPS older Form 4s — an open-market
    buy before the recent cutoff is invisible. Verified live against the bulk data
    set. **Prefer ``quantlab.insider_bulk.BulkInsiderSource`` for any cross-sectional
    / multi-year insider analysis** — it reads SEC's bulk quarterly Form 345 data
    sets (no pagination gap, far fewer requests) and is cross-checked byte-identical
    to this reader on the recent window. This class is fine for a single recent name
    (smoke tests); a full crawl fix (reading the overflow pages) is a follow-up."""

    def __init__(self, cache_dir: str = CACHE):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._ticker_map: dict[str, str] | None = None

    # -- ticker -> CIK (current-only SEC map; same survivorship caveat as H1) -- #
    def ticker_cik(self, ticker: str) -> str | None:
        if self._ticker_map is None:
            path = os.path.join(self.cache_dir, "company_tickers.json")
            if os.path.exists(path):
                payload = json.loads(open(path, encoding="utf-8").read())
            else:
                payload = json.loads(
                    _get("https://www.sec.gov/files/company_tickers.json"))
                open(path, "w", encoding="utf-8").write(json.dumps(payload))
            self._ticker_map = {
                rec["ticker"].upper(): str(int(rec["cik_str"])).zfill(10)
                for rec in payload.values()
                if rec.get("ticker") and rec.get("cik_str") is not None
            }
        return self._ticker_map.get(ticker.upper())

    def _submissions(self, cik: str) -> dict:
        """Per-CIK submissions metadata (cached)."""
        path = os.path.join(self.cache_dir, f"sub_{cik}.json")
        if os.path.exists(path):
            return json.loads(open(path, encoding="utf-8").read())
        try:
            raw = _get(_SUBMISSIONS_URL.format(cik=cik), timeout=60)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {}
            raise
        open(path, "wb").write(raw)
        return json.loads(raw)

    def list_form4_accessions(self, cik: str) -> pd.DataFrame:
        """Form 4 accessions for ``cik`` with their filing dates (columns
        ``[accession, filed]``). PIT: ``filed`` is the submissions filingDate.

        INCOMPLETE for prolific filers (see the class COMPLETENESS WARNING): only
        ``filings.recent`` (~1000 most-recent filings) is read; older Form 4s in the
        ``filings.files`` overflow pages are NOT included. Use
        ``insider_bulk.BulkInsiderSource`` for complete multi-year coverage."""
        cik = str(int(cik)).zfill(10)
        return parse_form4_accessions(self._submissions(cik))

    def _resolve_raw_xml_url(self, cik: str, accession: str) -> str | None:
        """Find the RAW ownershipDocument ``.xml`` in an accession folder.

        The accession directory's ``index.json`` lists every document. We pick the
        ``.xml`` that is NOT under an ``xsl...`` rendering path and whose name does
        not start with the XSLT prefix — the form-4 document itself. Resolution is
        cached so a repeat call is free."""
        cik_int = str(int(cik))
        acc_nodash = accession.replace("-", "")
        index_url = _ARCHIVE_DIR.format(cik=cik_int, acc_nodash=acc_nodash) + "index.json"
        try:
            listing = json.loads(_get(index_url, timeout=60))
        except urllib.error.HTTPError:
            return None
        items = (listing.get("directory", {}) or {}).get("item", []) or []
        candidates = []
        for it in items:
            name = it.get("name", "")
            low = name.lower()
            if not low.endswith(".xml"):
                continue
            # Skip XSLT renderings (path/name carries the xsl marker) and the
            # filing-summary / R-document artifacts.
            if low.startswith("xsl") or "/xsl" in low or low in {
                    "filingsummary.xml", "primary_doc.xml.xsl"}:
                continue
            candidates.append(name)
        # Prefer a name that contains "form" or starts with the doc number; else
        # fall back to the first non-xsl .xml (the form-4 doc is the only
        # ownership .xml in a Form 4 accession in practice).
        for name in candidates:
            if "form" in name.lower() or name.lower().startswith("primary_doc"):
                return _ARCHIVE_DIR.format(cik=cik_int, acc_nodash=acc_nodash) + name
        if candidates:
            return _ARCHIVE_DIR.format(cik=cik_int, acc_nodash=acc_nodash) + candidates[0]
        return None

    def fetch_form4_raw(self, cik: str, accession: str) -> str | None:
        """Fetch (and cache) the RAW ownershipDocument XML text for one accession.

        Returns None if no ownership XML can be resolved. Cached as a ``.xml`` file
        keyed by accession so re-parsing never hits the network. Validates the root
        element is ``ownershipDocument`` before caching (guards against caching an
        XSLT rendering by mistake)."""
        cik = str(int(cik)).zfill(10)
        acc_nodash = accession.replace("-", "")
        path = os.path.join(self.cache_dir, f"form4_{cik}_{acc_nodash}.xml")
        if os.path.exists(path):
            return open(path, encoding="utf-8").read()
        url = self._resolve_raw_xml_url(cik, accession)
        if url is None:
            return None
        try:
            text = _get(url, timeout=60).decode("utf-8", "ignore")
        except urllib.error.HTTPError:
            return None
        if not is_ownership_xml(text):
            return None
        open(path, "w", encoding="utf-8").write(text)
        return text

    def _transactions(self, cik_or_ticker: str, code: str, acq: str) -> pd.DataFrame:
        """Shared body of ``purchases``/``sells``: open-market transactions of a
        single (code, acquired/disposed) kind for a company, as a FILING-DATE-
        indexed DataFrame.

        Columns: ``owner_name, role, shares, value, transaction_date, ticker,
        accession``. The index is ``filed_date`` (the submissions filingDate — the
        PIT instant the market could know). ``value`` = shares x price (NaN if
        price absent). ``role`` is a compact tag (officer/director/10%/insider).
        Filters ``parse_form4_xml`` rows to ``code == code`` AND
        ``acq_disp == acq`` (P/A for buys, S/D for sells — the two open-market
        legs). The PURE parser is untouched; this only selects the kind.

        Accepts a ticker (resolved via the current SEC map) or a raw CIK. Returns
        an empty frame (with the right columns) for an unmapped/empty name."""
        cik = (cik_or_ticker if str(cik_or_ticker).isdigit()
               else self.ticker_cik(cik_or_ticker))
        cols = ["owner_name", "role", "shares", "value", "transaction_date",
                "ticker", "accession"]
        if cik is None:
            return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="filed_date"))
        cik = str(int(cik)).zfill(10)
        accs = self.list_form4_accessions(cik)
        rows = []
        for acc, filed in zip(accs["accession"], accs["filed"]):
            xml = self.fetch_form4_raw(cik, acc)
            if xml is None:
                continue
            for r in parse_form4_xml(xml):
                if r["code"] != code or r["acq_disp"] != acq:
                    continue  # the requested open-market leg only
                shares = r["shares"]
                price = r["price"]
                rows.append({
                    "filed_date": filed,
                    "owner_name": r["owner_name"],
                    "role": _role_tag(r),
                    "shares": shares,
                    "value": (shares * price
                              if shares is not None and price is not None else None),
                    "transaction_date": (pd.to_datetime(r["transaction_date"])
                                         if r["transaction_date"] else pd.NaT),
                    "ticker": r["ticker"],
                    "accession": acc,
                })
        if not rows:
            return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="filed_date"))
        df = pd.DataFrame(rows).set_index("filed_date").sort_index()
        df.index.name = "filed_date"
        return df[cols]

    def purchases(self, cik_or_ticker: str) -> pd.DataFrame:
        """Open-market PURCHASES (code P, acquired A) for a company, as a
        FILING-DATE-indexed DataFrame.

        Columns: ``owner_name, role, shares, value, transaction_date, ticker,
        accession``. The index is ``filed_date`` (the submissions filingDate — the
        PIT instant the market could know). ``value`` = shares x price (NaN if
        price absent). ``role`` is a compact tag (officer/director/10%/insider).

        Accepts a ticker (resolved via the current SEC map) or a raw CIK. Returns
        an empty frame (with the right columns) for an unmapped/empty name.

        PIT: indexed by ``filed_date`` only — a buy enters at its filing instant,
        never its (earlier) transaction date (the market could not see it sooner)."""
        return self._transactions(cik_or_ticker, code="P", acq="A")

    def sells(self, cik_or_ticker: str) -> pd.DataFrame:
        """Open-market SELLS (code S, disposed D) for a company, as a FILING-DATE-
        indexed DataFrame with the SAME columns/shape as ``purchases`` — the
        net-of-sells leg the H10 frozen signal subtracts.

        Symmetric to ``purchases`` in every respect (filing-date index, value =
        shares x price, role tag); only the transaction kind differs (S/D, the
        open-market sale, vs P/A). Accepts a ticker or raw CIK; empty frame for an
        unmapped name.

        PIT: indexed by ``filed_date`` only — a sale enters at its filing instant
        (Form 4 due within 2 business days), never its earlier transaction date,
        so the net signal at t never peeks at a not-yet-filed disposal."""
        return self._transactions(cik_or_ticker, code="S", acq="D")


def _role_tag(r: dict) -> str:
    """Compact role label from the parsed relationship flags."""
    if r.get("is_officer"):
        return "officer"
    if r.get("is_director"):
        return "director"
    if r.get("is_tenpct"):
        return "tenpct"
    return "insider"
