"""Free historical ticker->CIK recovery for dead/renamed names — and an honest
measurement of how little of the survivorship hole it can close.

THE PROBLEM (H1 audit, research_log 2026-06-14). SEC's ``company_tickers.json``
is CURRENT-ONLY, so ~27% of the PIT S&P universe — the dead/acquired/renamed
names — has no ticker->CIK mapping. Those are exactly the names whose omission
caused trial #1's survivorship illusion (trial #2). A survivorship-SAFE H1 needs
their fundamentals, so it needs their CIKs.

THE FREE OPTIONS, AND WHY THEY MOSTLY FAIL.
- Name-based (SEC ``cik-lookup-data.txt`` maps every filer NAME, incl. former
  names, to a CIK): blocked here — our PIT universe (Wikipedia change table)
  carries only TICKERS, no company names, so there is nothing to look up.
- Ticker-based (EDGAR ``getcompany&ticker=`` resolves a ticker to a CIK): works
  ONLY while a ticker is still in EDGAR's ticker index, i.e. for recently-dead
  names. Older M&A/bankruptcy tickers are dropped from the index and return
  nothing. And tickers get REASSIGNED, so a hit can be a *living* company that
  later took the symbol — a false link a graded trial must not accept.

This module implements the ticker-based path with a reassignment-safety
classifier, so the recoverable fraction can be MEASURED (scripts/h1_cik_coverage)
rather than asserted. The measured answer is the point: free recovery is partial
and unsafe, which is why H1 waits on Compustat/CRSP. Pure parsers are pinned by
tests; the network calls are isolated and rate-limited per SEC fair-access.
"""
from __future__ import annotations

import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

socket.setdefaulttimeout(60)
_UA = "qr-alpha-lab research Jared@how.co"
_MIN_INTERVAL = 0.15
_last = [0.0]

_GETCOMPANY = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
               "&ticker={ticker}&type=10-K&dateb=&owner=include&count=1&output=atom")
_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"


# --------------------------------------------------------------------------- #
# Pure parsers (no network) — pinned by tests.
# --------------------------------------------------------------------------- #

def parse_getcompany_cik(atom_text: str) -> int | None:
    """Extract the CIK from an EDGAR ``getcompany`` atom response. Handles both
    the ``<cik>NNN</cik>`` company-info element and a ``CIK=NNN`` querystring;
    returns None when the ticker is unknown to EDGAR (empty/locked feed)."""
    if not atom_text:
        return None
    m = re.search(r"<cik>(\d+)</cik>", atom_text) or re.search(r"CIK=(\d+)", atom_text)
    return int(m.group(1)) if m else None


def last_10k_date(submissions: dict) -> str | None:
    """Most recent 10-K ``filingDate`` from a submissions JSON, or None."""
    recent = (submissions or {}).get("filings", {}).get("recent", {})
    dates = [fd for f, fd in zip(recent.get("form", []), recent.get("filingDate", []))
             if f == "10-K"]
    return max(dates) if dates else None


def classify_resolution(last_10k: str | None, deletion_date: str | None,
                        grace_days: int = 540) -> str:
    """Is a resolved CIK plausibly the DEAD name we wanted, or a reassignment?

    A dead name stops filing 10-Ks around when it leaves the index. If the CIK's
    last 10-K is on/before ``deletion_date + grace`` it is ``plausible_dead``; if
    it keeps filing well after, the ticker was likely REASSIGNED to a living
    company -> ``possible_reassignment`` (a false link a trial must reject). With
    no last 10-K or no deletion date -> ``unknown`` (shown, never guessed)."""
    if not last_10k or not deletion_date:
        return "unknown"
    import pandas as pd
    cutoff = pd.Timestamp(deletion_date) + pd.Timedelta(days=grace_days)
    return "plausible_dead" if pd.Timestamp(last_10k) <= cutoff else "possible_reassignment"


# --------------------------------------------------------------------------- #
# Network resolver (isolated, rate-limited).
# --------------------------------------------------------------------------- #

def _get(url: str, retries: int = 3) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    for attempt in range(retries):
        wait = _MIN_INTERVAL - (time.monotonic() - _last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                _last[0] = time.monotonic()
                return r.read().decode("utf-8", "ignore")
        except (urllib.error.HTTPError, urllib.error.URLError):
            _last[0] = time.monotonic()
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return ""
    return ""


@dataclass
class Resolution:
    ticker: str
    cik: int | None
    last_10k: str | None
    verdict: str            # resolved_via_edgar / unresolved


def resolve_ticker_cik(ticker: str, fetch=_get) -> Resolution:
    """Best-effort free ticker->CIK via EDGAR (recent dead names only). ``fetch``
    is injectable so the logic is testable without network."""
    atom = fetch(_GETCOMPANY.format(ticker=ticker))
    cik = parse_getcompany_cik(atom)
    if cik is None:
        return Resolution(ticker, None, None, "unresolved")
    import json
    sub = fetch(_SUBMISSIONS.format(cik=cik))
    try:
        last = last_10k_date(json.loads(sub)) if sub else None
    except Exception:
        last = None
    return Resolution(ticker, cik, last, "resolved_via_edgar")
