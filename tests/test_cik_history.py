"""Known-answer tests for the free historical ticker->CIK resolver. No network:
the parsers are pure and the network resolver takes an injectable fetch.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import cik_history as ch


def test_parse_getcompany_cik_atom_element():
    atom = "<company-info><cik>0000718877</cik><name>Activision</name></company-info>"
    assert ch.parse_getcompany_cik(atom) == 718877


def test_parse_getcompany_cik_querystring_fallback():
    assert ch.parse_getcompany_cik("...&CIK=815094&type=10-K...") == 815094


def test_parse_getcompany_cik_miss():
    assert ch.parse_getcompany_cik("") is None
    assert ch.parse_getcompany_cik("<feed>no company here</feed>") is None


def test_last_10k_date_picks_latest():
    sub = {"filings": {"recent": {
        "form": ["10-Q", "10-K", "8-K", "10-K"],
        "filingDate": ["2022-08-01", "2021-02-20", "2022-01-01", "2022-05-20"]}}}
    assert ch.last_10k_date(sub) == "2022-05-20"
    assert ch.last_10k_date({"filings": {"recent": {"form": ["8-K"], "filingDate": ["2020-01-01"]}}}) is None


def test_classify_resolution():
    # stopped filing around deletion -> plausibly the dead name
    assert ch.classify_resolution("2022-05-20", "2022-12-12") == "plausible_dead"
    # still filing 10-Ks years after deletion -> ticker reassigned (false link)
    assert ch.classify_resolution("2025-02-01", "2018-11-28") == "possible_reassignment"
    # missing inputs are shown, never guessed
    assert ch.classify_resolution(None, "2020-01-01") == "unknown"
    assert ch.classify_resolution("2020-01-01", None) == "unknown"


def test_resolve_ticker_cik_with_injected_fetch():
    def fake_fetch(url):
        if "getcompany" in url:
            return "<cik>0000718877</cik>"
        return json.dumps({"filings": {"recent": {
            "form": ["10-K"], "filingDate": ["2023-02-23"]}}})
    r = ch.resolve_ticker_cik("ATVI", fetch=fake_fetch)
    assert r.cik == 718877 and r.last_10k == "2023-02-23" and r.verdict == "resolved_via_edgar"


def test_resolve_ticker_cik_unresolved():
    r = ch.resolve_ticker_cik("DEADTICKER", fetch=lambda url: "")
    assert r.cik is None and r.verdict == "unresolved"
