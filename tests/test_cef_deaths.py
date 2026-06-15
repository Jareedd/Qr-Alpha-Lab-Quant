"""Dead-CEF census (quantlab.cef_deaths) — offline known-answer tests.

Pins the EFTS parser, the fund-name pre-screen, and the terminal-outcome
classifier against fixtures from the live EDGAR schema (2026-06-15). Network
fetchers are thin cache wrappers; the classification is where the census's
honesty lives, so it gets the same hostile pinning as H8's removal classifier.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import cef_deaths as cd


# --- name cleaning / EFTS parse -------------------------------------------- #

def test_clean_name_strips_ticker_and_cik():
    assert cd.clean_name("PIMCO Dynamic Income Fund  (PDI)  (CIK 0001510599)") \
        == "PIMCO Dynamic Income Fund"
    assert cd.clean_name("DISH Network CORP  (CIK 0001001082)") == "DISH Network CORP"


def test_parse_efts_takes_delisting_entity_not_exchange():
    payload = {"hits": {"hits": [
        {"_source": {
            "display_names": ["Virtus Stone Harbor EM Total Income  (CIK 0001550913)",
                              "NEW YORK STOCK EXCHANGE LLC  (CIK 0000876661)"],
            "ciks": ["0001550913", "0000876661"],
            "file_date": "2023-12-18"}},
    ]}}
    hits = cd.parse_efts_hits(payload)
    assert len(hits) == 1
    assert hits[0] == {"cik": "1550913",
                       "name": "Virtus Stone Harbor EM Total Income",
                       "date": "2023-12-18"}
    assert cd.parse_efts_hits({}) == []


# --- fund name pre-screen (recall-first) ----------------------------------- #

def test_fund_name_screen_recall_and_rejects_operating_cos():
    for nm in ("Nuveen Corporate Income 2023 Target Term Fund",
               "First Trust Dynamic Europe Equity Income Fund",
               "BlackRock Municipal Income Trust", "Oxford Square Capital Corp."):
        assert cd.is_fund_name(nm), nm
    for nm in ("DISH Network CORP", "Applied Molecular Transport Inc.",
               "ENETI INC."):
        assert not cd.is_fund_name(nm), nm


# --- terminal-outcome classifier ------------------------------------------- #

def test_term_fund_is_nav_event():
    r = cd.classify_death("Invesco High Income 2023 Target Term Fund",
                          {"N-2", "N-CSR", "N-CEN", "25-NSE"}, current_tickers=[])
    assert r["outcome"] == "term_liquidation"
    assert r["nav_event"] is True
    assert r["is_cef_or_bdc"] and not r["is_bdc"]


def test_n8f_dereg_is_liquidation_or_merger_nav_event():
    r = cd.classify_death("Virtus Stone Harbor EM Total Income",
                          {"N-2", "N-CSR", "N-8F", "25-NSE"}, current_tickers=[])
    assert r["outcome"] == "liquidation_or_merger"
    assert r["nav_event"] is True


def test_open_ending_when_still_filing_fund_reports_no_dereg():
    r = cd.classify_death("Some Closed-End Income Fund",
                          {"N-2", "N-CEN", "N-CSR", "25-NSE"}, current_tickers=[])
    assert r["outcome"] == "open_ending_or_conversion"
    assert r["nav_event"] is True


def test_still_listed_common_is_security_redemption_not_death():
    # fund still has a current ticker -> a preferred/note delisted, fund alive
    r = cd.classify_death("Oxford Square Capital Corp.",
                          {"N-2", "10-K", "10-Q", "25-NSE"},
                          current_tickers=["OXSQ"])
    assert r["outcome"] == "security_redemption_fund_alive"
    assert r["nav_event"] is None
    assert r["is_bdc"] is True            # N-2 + 10-K -> BDC


def test_non_n2_entity_is_not_a_cef():
    r = cd.classify_death("iSHARES TRUST", {"N-1A", "N-CEN", "497", "25-NSE"},
                          current_tickers=[])
    assert r["outcome"] == "not_a_closed_end_fund"
    assert r["nav_event"] is None
    assert r["is_cef_or_bdc"] is False


def test_delisted_unknown_is_shown_not_guessed():
    # N-2 fund, fully delisted, but no N-8F, no ongoing fund reports, not term:
    # we do NOT guess a NAV event -- unknown is surfaced.
    r = cd.classify_death("Mystery Holdings Fund", {"N-2", "25-NSE"},
                          current_tickers=[])
    assert r["outcome"] == "delisted_unknown"
    assert r["nav_event"] is None


def test_distress_text_detector():
    assert cd.text_has_distress("...failed to meet the minimum bid price...")
    assert cd.text_has_distress("the Fund filed for Chapter 11 bankruptcy")
    assert not cd.text_has_distress("the Fund completed its liquidation at NAV")
