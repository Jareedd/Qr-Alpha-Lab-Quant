"""Dead closed-end-fund census for H6 Stage-1 — the make-or-break gate.

CEFConnect's universe is current-listings-only (cef_data.py), so the dead funds
that decide H6's survivorship direction must come from an INDEPENDENT source.
This module enumerates them from SEC EDGAR (free, authoritative) and classifies
their terminal outcomes, to test the H6 thesis's load-bearing claim:

    "CEF deaths happen at NAV (liquidation / open-ending / merger), so OMITTING
     dead funds biases a discount-long backtest CONSERVATIVELY (against us)."

If that holds, H6 is the one idea on the board where the project's #1 killer —
missing dead names — is a tailwind, not a bias to fear. If it fails (deaths are
distress sales below NAV), the gate kills H6 before a trial is spent.

Method (verified 2026-06-15):
- ENUMERATE deaths via SEC EFTS full-text search, form **25-NSE** (Notification
  of Removal from Listing). Every exchange-listed CEF death — liquidation,
  open-ending, merger, term maturity — delists, so Form 25 captures them all.
  (EFTS does NOT index N-2 / N-8F as form facets — both 500 — so those signals
  are read from each filer's submissions JSON instead.) Query PER YEAR; each
  year is < the EFTS 10k `from` cap.
- FILTER to closed-end funds/BDCs: the filer's submissions JSON shows it filed
  **N-2** (the closed-end/BDC registration form; ETF/open-end trusts file N-1A
  and are excluded). A name pre-screen cuts the ~1k/yr non-fund delistings
  before the submissions fetch (recall-first; precision comes from the N-2 test).
- EXCLUDE security-level redemptions: a fund that delists a PREFERRED/NOTE while
  its common stock keeps trading is NOT dead (it still has a current ticker).
  Only fully-delisted entities (no current ticker) count as deaths.
- CLASSIFY terminal outcome from filing-type signature + name (keyword
  classifier, H8 `classify_removal_reason` discipline: unknowns are shown, never
  guessed). Distress (the only NON-NAV outcome) is checked separately.

Pure parsers/classifiers are split out and pinned by tests/test_cef_deaths.py;
the network orchestration lives in the fetchers + scripts/cef_dead_fund_census.py.
Nothing here computes a signal or forward return — this is a descriptive census.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request

socket.setdefaulttimeout(120)

# SEC fair-access policy: a real User-Agent with contact, and <=10 req/s.
_UA = "qr-alpha-lab research Jared@how.co"
EFTS = "https://efts.sec.gov/LATEST/search-index"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
CACHE = os.path.join("data_cache", "cef_deaths")
_MIN_INTERVAL = 0.12          # ~8 req/s, comfortably under the limit
_last_call = [0.0]


def _get(url: str, timeout: int = 60, retries: int = 4) -> bytes:
    # crude global rate limiter (single-threaded census; threads not needed)
    import time as _t
    req = urllib.request.Request(
        url, headers={"User-Agent": _UA, "Accept-Encoding": "gzip, deflate"}
    )
    for attempt in range(retries):
        wait = _MIN_INTERVAL - (_t.monotonic() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
            _last_call[0] = _t.monotonic()
            return raw
        except urllib.error.HTTPError as e:
            # EFTS 500s sporadically; 429/503 are rate/availability. Back off
            # and retry; re-raise client errors (404 etc.) and the final 5xx.
            _last_call[0] = _t.monotonic()
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
        except urllib.error.URLError:
            _last_call[0] = _t.monotonic()
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("unreachable")  # loop returns or raises


# --------------------------------------------------------------------------- #
# Pure parsers / classifiers (no network) -- pinned by tests.
# --------------------------------------------------------------------------- #

_CIK_TAG = re.compile(r"\s*\(CIK\s*\d+\)\s*$")
_TICKER_TAG = re.compile(r"\s*\([A-Z0-9.\-, ]+\)\s*$")

# A delisting is plausibly a fund if its name carries fund vocabulary. Broad on
# purpose (recall-first): the N-2 submissions test supplies precision, so a few
# extra submissions fetches are cheaper than missing a real CEF.
_FUND_NAME = re.compile(
    r"\b(fund|trust|income|municipal|muni|term|capital|strateg\w*|"
    r"opportunit\w*|dividend|yield|portfolio|closed[- ]?end|bdc|infrastructure|"
    r"energy|credit|bond|equit\w*|growth|value|senior|floating|tax|advantage|"
    r"premium|global|intermediate|target|select|holdings|investors?)\b",
    re.IGNORECASE,
)

# Target-term CEFs liquidate at a pre-set date BY DESIGN, returning ~NAV.
_TERM = re.compile(r"\b(target[- ]?term|term (fund|trust)|20\d\d (target|term))\b",
                   re.IGNORECASE)
# Distress markers (the only NON-NAV outcome) -- scanned in filing text, not names.
_DISTRESS = re.compile(
    r"(bankrupt|chapter 11|chapter 7|deficienc|delinquen|noncompliance|"
    r"non-compliance|minimum (bid|market)|going concern|fraud|receiver)",
    re.IGNORECASE,
)


def clean_name(display_name: str) -> str:
    """'PIMCO ... Fund  (PDI)  (CIK 0001234567)' -> 'PIMCO ... Fund'."""
    s = _CIK_TAG.sub("", display_name)
    s = _TICKER_TAG.sub("", s)
    return s.strip()


def parse_efts_hits(payload: dict) -> list[dict]:
    """EFTS search JSON -> [{cik, name, date}] for the DELISTING entity (the
    first display_name; the second is the exchange). Empty list if no hits."""
    out = []
    for h in (payload.get("hits", {}) or {}).get("hits", []) or []:
        src = h.get("_source", {}) or {}
        names = src.get("display_names") or []
        ciks = src.get("ciks") or []
        if names and ciks:
            out.append({
                "cik": str(int(ciks[0])),
                "name": clean_name(names[0]),
                "date": src.get("file_date"),
            })
    return out


def is_fund_name(name: str) -> bool:
    return bool(_FUND_NAME.search(name or ""))


def classify_death(name: str, recent_forms: set[str],
                   current_tickers: list[str]) -> dict:
    """Classify a delisted entity from its filing-type signature + name.

    Returns a dict with: is_cef_or_bdc, is_bdc, fully_delisted, outcome, and
    nav_event (True = pays ~NAV: liquidation/merger/open-end/term; None =
    unknown, shown not guessed; False is set only by an explicit distress scan).
    """
    forms = {f.upper() for f in recent_forms}
    filed_n2 = any(f.startswith("N-2") for f in forms)
    # BDCs file N-2 AND 10-K (they are operating/reporting cos); CEFs file
    # N-2 + N-CSR/N-CEN and no 10-K. The split matters: BDC deaths can be
    # distress, CEF deaths are typically NAV events.
    is_bdc = filed_n2 and ("10-K" in forms or "10-K405" in forms)
    is_cef_or_bdc = filed_n2
    fully_delisted = len(current_tickers or []) == 0
    has_n8f = any(f.startswith("N-8F") for f in forms)
    still_fund_filings = any(f.startswith(("N-CEN", "N-CSR")) for f in forms)

    if not is_cef_or_bdc:
        outcome, nav = "not_a_closed_end_fund", None
    elif not fully_delisted:
        # common stock still listed -> a preferred/note was redeemed, not death
        outcome, nav = "security_redemption_fund_alive", None
    elif _TERM.search(name or ""):
        outcome, nav = "term_liquidation", True          # pays NAV by design
    elif has_n8f:
        outcome, nav = "liquidation_or_merger", True     # deregistered -> NAV
    elif still_fund_filings:
        outcome, nav = "open_ending_or_conversion", True  # redeemable at NAV
    else:
        outcome, nav = "delisted_unknown", None          # shown, not guessed

    return {
        "is_cef_or_bdc": is_cef_or_bdc, "is_bdc": is_bdc,
        "fully_delisted": fully_delisted, "outcome": outcome, "nav_event": nav,
    }


def text_has_distress(text: str) -> bool:
    return bool(_DISTRESS.search(text or ""))


# --------------------------------------------------------------------------- #
# Network fetchers (cached).
# --------------------------------------------------------------------------- #

def form_delistings(year: int, form: str = "25-NSE",
                    cache_dir: str = CACHE) -> list[dict]:
    """All `form` filings in `year` from EFTS (paginated, q empty). Cached JSON.
    One calendar year is well under the EFTS 10k `from` cap for Form 25."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{form}_{year}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    out, frm = [], 0
    while True:
        url = (f"{EFTS}?q=&forms={form}&startdt={year}-01-01"
               f"&enddt={year}-12-31&from={frm}")
        payload = json.loads(_get(url))
        hits = parse_efts_hits(payload)
        out.extend(hits)
        total = (payload.get("hits", {}) or {}).get("total", {}).get("value", 0)
        frm += len(payload.get("hits", {}).get("hits", []) or [])
        if frm >= total or not hits:
            break
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out


def submissions(cik: str, cache_dir: str = CACHE) -> dict:
    """Filer submissions JSON (form history, current tickers/exchanges, SIC).
    Cached per CIK. The recent-forms list is what the classifier reads."""
    os.makedirs(cache_dir, exist_ok=True)
    cikpad = str(int(cik)).zfill(10)
    path = os.path.join(cache_dir, f"sub_{cikpad}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    sub = json.loads(_get(SUBMISSIONS.format(cik=cikpad)))
    # keep only what the census needs (the full file is large)
    rec = sub.get("filings", {}).get("recent", {})
    slim = {
        "name": sub.get("name"),
        "sic": sub.get("sicDescription"),
        "tickers": sub.get("tickers", []),
        "exchanges": sub.get("exchanges", []),
        "forms": sorted(set(rec.get("form", []))),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(slim, f)
    return slim
