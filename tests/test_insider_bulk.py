"""BulkInsiderSource — offline known-answer tests. No network.

The bulk Form 3/4/5 data sets are the SAME Form 4 data the crawl reads, delivered
as quarterly TSV ZIPs. These tests pin the PURE parsers + the join (``build_
transactions``) on tiny hand-built TSV byte blobs, so the parse/select logic is
validated without ever downloading a ZIP:

  * P/A open-market BUY kept; S/D open-market SALE separated into kind="S".
  * non-open-market codes EXCLUDED from both buys and sells: ``A`` (award/grant),
    ``M`` (option exercise), ``G`` (gift), ``F`` (shares withheld for tax).
  * DD-MMM-YYYY date parsing (filing + transaction dates).
  * multi-owner accession: FIRST owner by default (matches the crawl) vs
    ``all_owners=True`` (one row per owner).
  * missing price -> value is NaN (never crashes).
  * role mapping from free-text RPTOWNER_RELATIONSHIP (officer>director>tenpct).
  * filing-date index + EXACT output columns/order == InsiderSource.purchases().
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import insider_bulk as ib
from quantlab.insider_data import InsiderSource


# --------------------------------------------------------------------------- #
# Hand-built tiny TSV blobs (tab-separated, real headers, real DD-MMM-YYYY).
# --------------------------------------------------------------------------- #

def _tsv(header: list[str], rows: list[list[str]]) -> bytes:
    """Assemble a tab-separated byte blob with the given header + rows."""
    lines = ["\t".join(header)]
    lines += ["\t".join(r) for r in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


# SUBMISSION.tsv — three accessions for issuer CIK 0000000321 (ticker ABC):
#   ACC1 a plain Form 4; ACC2 a Form 4 with TWO owners; ACC3 a Form 5 that MUST be
#   dropped (DOCUMENT_TYPE filter keeps only 4 / 4/A).
_SUB_HEADER = ["ACCESSION_NUMBER", "FILING_DATE", "PERIOD_OF_REPORT",
               "DATE_OF_ORIG_SUB", "NO_SECURITIES_OWNED", "NOT_SUBJECT_SEC16",
               "FORM3_HOLDINGS_REPORTED", "FORM4_TRANS_REPORTED",
               "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERNAME",
               "ISSUERTRADINGSYMBOL", "REMARKS"]


def _sub_blob():
    rows = [
        ["ACC1", "05-FEB-2024", "31-JAN-2024", "", "", "", "", "1",
         "4", "0000000321", "ABC CORP", "ABC", ""],
        ["ACC2", "10-MAR-2024", "08-MAR-2024", "", "", "", "", "1",
         "4", "0000000321", "ABC CORP", "ABC", ""],
        ["ACC3", "12-MAR-2024", "10-MAR-2024", "", "", "", "", "1",
         "5", "0000000321", "ABC CORP", "ABC", ""],   # Form 5 -> dropped
    ]
    return _tsv(_SUB_HEADER, rows)


_RO_HEADER = ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME",
              "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE", "RPTOWNER_TXT",
              "RPTOWNER_STREET1", "RPTOWNER_STREET2", "RPTOWNER_CITY",
              "RPTOWNER_STATE", "RPTOWNER_ZIPCODE", "RPTOWNER_STATE_DESC",
              "FILE_NUMBER"]


def _ro_blob():
    rows = [
        # ACC1: single owner, Director,Officer -> role precedence picks officer.
        ["ACC1", "111", "DOE JANE", "Director,Officer", "CFO", "",
         "", "", "", "", "", "", ""],
        # ACC2: TWO owners. First = TenPercentOwner (role tenpct), second =
        # Director (role director). First-owner-by-default keeps only the first.
        ["ACC2", "222", "FIRST OWNER", "TenPercentOwner", "", "",
         "", "", "", "", "", "", ""],
        ["ACC2", "333", "SECOND OWNER", "Director", "", "",
         "", "", "", "", "", "", ""],
        ["ACC3", "444", "FIVE FILER", "Officer", "", "",
         "", "", "", "", "", "", ""],
    ]
    return _tsv(_RO_HEADER, rows)


_TR_HEADER = ["ACCESSION_NUMBER", "NONDERIV_TRANS_SK", "SECURITY_TITLE",
              "SECURITY_TITLE_FN", "TRANS_DATE", "TRANS_DATE_FN",
              "DEEMED_EXECUTION_DATE", "DEEMED_EXECUTION_DATE_FN",
              "TRANS_FORM_TYPE", "TRANS_CODE", "EQUITY_SWAP_INVOLVED",
              "EQUITY_SWAP_TRANS_CD_FN", "TRANS_TIMELINESS",
              "TRANS_TIMELINESS_FN", "TRANS_SHARES", "TRANS_SHARES_FN",
              "TRANS_PRICEPERSHARE", "TRANS_PRICEPERSHARE_FN",
              "TRANS_ACQUIRED_DISP_CD", "TRANS_ACQUIRED_DISP_CD_FN",
              "SHRS_OWND_FOLWNG_TRANS", "SHRS_OWND_FOLWNG_TRANS_FN",
              "VALU_OWND_FOLWNG_TRANS", "VALU_OWND_FOLWNG_TRANS_FN",
              "DIRECT_INDIRECT_OWNERSHIP", "DIRECT_INDIRECT_OWNERSHIP_FN",
              "NATURE_OF_OWNERSHIP", "NATURE_OF_OWNERSHIP_FN"]


def _tr_row(acc, sk, date, code, shares, price, acq):
    """One NONDERIV_TRANS row (only the columns we read are populated)."""
    r = [""] * len(_TR_HEADER)
    idx = {c: i for i, c in enumerate(_TR_HEADER)}
    r[idx["ACCESSION_NUMBER"]] = acc
    r[idx["NONDERIV_TRANS_SK"]] = sk
    r[idx["TRANS_DATE"]] = date
    r[idx["TRANS_CODE"]] = code
    r[idx["TRANS_SHARES"]] = shares
    r[idx["TRANS_PRICEPERSHARE"]] = price
    r[idx["TRANS_ACQUIRED_DISP_CD"]] = acq
    return r


def _tr_blob():
    rows = [
        # ACC1: a P/A open-market BUY (price 150.5) ...
        _tr_row("ACC1", "1", "31-JAN-2024", "P", "1000", "150.5", "A"),
        # ... an S/D open-market SALE ...
        _tr_row("ACC1", "2", "01-FEB-2024", "S", "200", "160.0", "D"),
        # ... and four NON-open-market codes that must be excluded from BOTH:
        _tr_row("ACC1", "3", "01-FEB-2024", "A", "50", "", "A"),    # award (no price)
        _tr_row("ACC1", "4", "01-FEB-2024", "M", "10", "5.0", "A"), # option exercise
        _tr_row("ACC1", "5", "01-FEB-2024", "G", "5", "", "D"),     # gift
        _tr_row("ACC1", "6", "01-FEB-2024", "F", "7", "150.0", "D"),# tax withholding
        # ACC2: a P/A BUY with NO price -> value must be NaN (not a crash).
        _tr_row("ACC2", "7", "08-MAR-2024", "P", "300", "", "A"),
        # ACC3 (Form 5, dropped at submission level): a P/A buy that must vanish.
        _tr_row("ACC3", "8", "10-MAR-2024", "P", "999", "1.0", "A"),
    ]
    return _tsv(_TR_HEADER, rows)


_CIKS = {"0000000321"}


# --------------------------------------------------------------------------- #
# Output schema contract — EXACTLY InsiderSource.purchases().
# --------------------------------------------------------------------------- #

def test_output_columns_match_crawl_exactly():
    # An empty crawl frame defines the canonical column order/index name.
    crawl_cols = list(InsiderSource().purchases("0").columns)
    buys = ib.build_transactions(_sub_blob(), _ro_blob(), _tr_blob(), _CIKS,
                                 kind="P")
    assert list(buys.columns) == crawl_cols
    assert list(buys.columns) == ["owner_name", "role", "shares", "value",
                                  "transaction_date", "ticker", "accession"]
    assert buys.index.name == "filed_date"


def test_empty_when_no_matching_cik():
    buys = ib.build_transactions(_sub_blob(), _ro_blob(), _tr_blob(),
                                 {"9999999999"}, kind="P")
    assert len(buys) == 0
    assert list(buys.columns) == ["owner_name", "role", "shares", "value",
                                  "transaction_date", "ticker", "accession"]
    assert buys.index.name == "filed_date"


# --------------------------------------------------------------------------- #
# P/A buy kept; S/D sale separated; non-open-market codes excluded.
# --------------------------------------------------------------------------- #

def test_open_market_buy_kept_and_priced():
    buys = ib.build_transactions(_sub_blob(), _ro_blob(), _tr_blob(), _CIKS,
                                 kind="P")
    # Two P/A buys survive: ACC1 (priced) + ACC2 (no price). The Form-5 ACC3 buy
    # is dropped at the submission filter.
    assert len(buys) == 2
    acc1 = buys[buys["accession"] == "ACC1"].iloc[0]
    assert acc1["shares"] == 1000.0
    assert acc1["value"] == 1000.0 * 150.5            # shares * price
    assert acc1["ticker"] == "ABC"


def test_sale_separated_into_kind_S():
    sells = ib.build_transactions(_sub_blob(), _ro_blob(), _tr_blob(), _CIKS,
                                  kind="S")
    # Exactly the single S/D disposal in ACC1 (the F tax row is S? no — code F,
    # disposed D, but F is not an open-market sale, so excluded).
    assert len(sells) == 1
    row = sells.iloc[0]
    assert row["shares"] == 200.0
    assert row["value"] == 200.0 * 160.0
    # the buy never leaks into sells.
    assert (sells["shares"] == 1000.0).sum() == 0


def test_non_open_market_codes_excluded_from_both():
    buys = ib.build_transactions(_sub_blob(), _ro_blob(), _tr_blob(), _CIKS,
                                 kind="P")
    sells = ib.build_transactions(_sub_blob(), _ro_blob(), _tr_blob(), _CIKS,
                                  kind="S")
    excluded_shares = {50.0, 10.0, 5.0, 7.0}   # A, M, G, F rows
    assert not (set(buys["shares"]) & excluded_shares)
    assert not (set(sells["shares"]) & excluded_shares)
    # A buy is only P/A; an M-code "acquired A" award-exercise is NOT a buy.
    assert (buys["shares"] == 10.0).sum() == 0


def test_form5_dropped_at_submission_filter():
    # ACC3 is a Form 5 with a P/A buy of 999 shares -> must not appear.
    buys = ib.build_transactions(_sub_blob(), _ro_blob(), _tr_blob(), _CIKS,
                                 kind="P")
    assert (buys["shares"] == 999.0).sum() == 0
    assert "ACC3" not in set(buys["accession"])


# SUBMISSION blob with a plain Form 4 (AMD1) AND a 4/A amendment (AMD2), both with
# a P/A buy, used to pin the amendment policy (faithfulness to the crawl).
def _sub_blob_amend():
    rows = [
        ["AMD1", "01-FEB-2024", "31-JAN-2024", "", "", "", "", "1",
         "4", "0000000321", "ABC CORP", "ABC", ""],
        ["AMD2", "05-FEB-2024", "31-JAN-2024", "", "", "", "", "1",
         "4/A", "0000000321", "ABC CORP", "ABC", ""],   # amendment
    ]
    return _tsv(_SUB_HEADER, rows)


def _ro_blob_amend():
    return _tsv(_RO_HEADER, [
        ["AMD1", "1", "OWNER ONE", "Director", "", "", "", "", "", "", "", "", ""],
        ["AMD2", "2", "OWNER TWO", "Director", "", "", "", "", "", "", "", "", ""],
    ])


def _tr_blob_amend():
    return _tsv(_TR_HEADER, [
        _tr_row("AMD1", "1", "31-JAN-2024", "P", "100", "10.0", "A"),
        _tr_row("AMD2", "2", "31-JAN-2024", "P", "200", "10.0", "A"),
    ])


def test_amendment_excluded_by_default_matches_crawl():
    # The crawl keeps only form == "4"; the bulk default must too. The 4/A buy
    # (200 shares) is DROPPED; only the plain Form-4 buy (100) survives.
    buys = ib.build_transactions(_sub_blob_amend(), _ro_blob_amend(),
                                 _tr_blob_amend(), _CIKS, kind="P")
    assert set(buys["shares"]) == {100.0}
    assert "AMD2" not in set(buys["accession"])


def test_amendment_included_when_opted_in():
    # include_amendments=True keeps the 4/A buy too (the one documented place the
    # bulk source legitimately diverges from the crawl).
    buys = ib.build_transactions(_sub_blob_amend(), _ro_blob_amend(),
                                 _tr_blob_amend(), _CIKS, kind="P",
                                 include_amendments=True)
    assert set(buys["shares"]) == {100.0, 200.0}
    assert {"AMD1", "AMD2"} <= set(buys["accession"])


# --------------------------------------------------------------------------- #
# DD-MMM-YYYY date parsing + filing-date index.
# --------------------------------------------------------------------------- #

def test_dd_mmm_yyyy_parsing_and_filing_date_index():
    buys = ib.build_transactions(_sub_blob(), _ro_blob(), _tr_blob(), _CIKS,
                                 kind="P")
    # filing-date index, sorted ascending.
    assert buys.index.is_monotonic_increasing
    acc1 = buys[buys["accession"] == "ACC1"].iloc[0]
    # ACC1 filed 05-FEB-2024, transaction 31-JAN-2024.
    assert buys[buys["accession"] == "ACC1"].index[0] == pd.Timestamp("2024-02-05")
    assert acc1["transaction_date"] == pd.Timestamp("2024-01-31")


def test_filing_date_index_never_transaction_date():
    # PIT contract: ACC1's transaction (31-JAN) is EARLIER than its filing
    # (05-FEB); the index must be the filing date, never the transaction date.
    buys = ib.build_transactions(_sub_blob(), _ro_blob(), _tr_blob(), _CIKS,
                                 kind="P")
    idx = buys[buys["accession"] == "ACC1"].index[0]
    txn = buys[buys["accession"] == "ACC1"].iloc[0]["transaction_date"]
    assert idx == pd.Timestamp("2024-02-05")
    assert idx > txn


# --------------------------------------------------------------------------- #
# Missing price -> NaN value (no crash).
# --------------------------------------------------------------------------- #

def test_missing_price_yields_nan_value():
    buys = ib.build_transactions(_sub_blob(), _ro_blob(), _tr_blob(), _CIKS,
                                 kind="P")
    acc2 = buys[buys["accession"] == "ACC2"].iloc[0]
    assert acc2["shares"] == 300.0
    assert np.isnan(acc2["value"])     # no price -> NaN, not 0, not crash


# --------------------------------------------------------------------------- #
# Multi-owner accession: first owner by default vs all_owners.
# --------------------------------------------------------------------------- #

def test_multi_owner_first_owner_by_default_matches_crawl():
    # ACC2 has two owners; default keeps only the FIRST (TenPercentOwner) =>
    # exactly one buy row for ACC2, owner_name "FIRST OWNER", role "tenpct".
    buys = ib.build_transactions(_sub_blob(), _ro_blob(), _tr_blob(), _CIKS,
                                 kind="P")
    acc2 = buys[buys["accession"] == "ACC2"]
    assert len(acc2) == 1
    assert acc2.iloc[0]["owner_name"] == "FIRST OWNER"
    assert acc2.iloc[0]["role"] == "tenpct"


def test_all_owners_keeps_every_owner_row():
    buys = ib.build_transactions(_sub_blob(), _ro_blob(), _tr_blob(), _CIKS,
                                 kind="P", all_owners=True)
    acc2 = buys[buys["accession"] == "ACC2"]
    # one transaction x two owners -> two rows.
    assert len(acc2) == 2
    assert set(acc2["owner_name"]) == {"FIRST OWNER", "SECOND OWNER"}
    assert set(acc2["role"]) == {"tenpct", "director"}


# --------------------------------------------------------------------------- #
# Role mapping from free-text RPTOWNER_RELATIONSHIP.
# --------------------------------------------------------------------------- #

def test_role_mapping_officer_precedence():
    # ACC1 owner is "Director,Officer" -> _role_tag precedence picks "officer".
    buys = ib.build_transactions(_sub_blob(), _ro_blob(), _tr_blob(), _CIKS,
                                 kind="P")
    acc1 = buys[buys["accession"] == "ACC1"].iloc[0]
    assert acc1["role"] == "officer"


def test_role_mapping_all_levels():
    # Pin each precedence level via a single-relationship owner per accession.
    sub = _tsv(_SUB_HEADER, [
        ["D1", "01-JAN-2024", "01-JAN-2024", "", "", "", "", "1", "4",
         "0000000321", "ABC", "ABC", ""],
        ["D2", "02-JAN-2024", "02-JAN-2024", "", "", "", "", "1", "4",
         "0000000321", "ABC", "ABC", ""],
        ["D3", "03-JAN-2024", "03-JAN-2024", "", "", "", "", "1", "4",
         "0000000321", "ABC", "ABC", ""],
        ["D4", "04-JAN-2024", "04-JAN-2024", "", "", "", "", "1", "4",
         "0000000321", "ABC", "ABC", ""],
    ])
    ro = _tsv(_RO_HEADER, [
        ["D1", "1", "OFF", "Officer", "", "", "", "", "", "", "", "", ""],
        ["D2", "2", "DIR", "Director", "", "", "", "", "", "", "", "", ""],
        ["D3", "3", "TEN", "TenPercentOwner", "", "", "", "", "", "", "", "", ""],
        ["D4", "4", "OTH", "Other", "", "", "", "", "", "", "", "", ""],
    ])
    tr = _tsv(_TR_HEADER, [
        _tr_row("D1", "1", "01-JAN-2024", "P", "1", "1.0", "A"),
        _tr_row("D2", "2", "02-JAN-2024", "P", "1", "1.0", "A"),
        _tr_row("D3", "3", "03-JAN-2024", "P", "1", "1.0", "A"),
        _tr_row("D4", "4", "04-JAN-2024", "P", "1", "1.0", "A"),
    ])
    buys = ib.build_transactions(sub, ro, tr, _CIKS, kind="P")
    role_by_acc = dict(zip(buys["accession"], buys["role"]))
    assert role_by_acc["D1"] == "officer"
    assert role_by_acc["D2"] == "director"
    assert role_by_acc["D3"] == "tenpct"
    assert role_by_acc["D4"] == "insider"      # "Other" -> none set -> insider


# --------------------------------------------------------------------------- #
# with_cik: opt-in issuer_cik column (the H12 price-alignment need).
# --------------------------------------------------------------------------- #

def test_with_cik_appends_issuer_cik_column():
    # Default: schema is EXACTLY the crawl's (no issuer_cik).
    base = ib.build_transactions(_sub_blob(), _ro_blob(), _tr_blob(), _CIKS,
                                 kind="P")
    assert "issuer_cik" not in base.columns
    assert list(base.columns) == ["owner_name", "role", "shares", "value",
                                  "transaction_date", "ticker", "accession"]

    # with_cik=True: append a 10-digit zero-padded issuer_cik, same rows/order.
    out = ib.build_transactions(_sub_blob(), _ro_blob(), _tr_blob(), _CIKS,
                                kind="P", with_cik=True)
    assert list(out.columns) == ["owner_name", "role", "shares", "value",
                                 "transaction_date", "ticker", "accession",
                                 "issuer_cik"]
    # the only difference vs the default is the extra column (same rows, order).
    assert len(out) == len(base)
    pd.testing.assert_frame_equal(out[base.columns], base)
    # every issuer_cik is the 10-digit zero-padded CIK 0000000321 (issuer ABC).
    assert set(out["issuer_cik"]) == {"0000000321"}
    acc1 = out[out["accession"] == "ACC1"].iloc[0]
    assert acc1["issuer_cik"] == "0000000321"
    assert acc1["ticker"] == "ABC"


# --------------------------------------------------------------------------- #
# kind validation.
# --------------------------------------------------------------------------- #

def test_invalid_kind_raises():
    import pytest
    with pytest.raises(ValueError):
        ib.build_transactions(_sub_blob(), _ro_blob(), _tr_blob(), _CIKS,
                              kind="X")
