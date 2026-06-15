"""H6 CEF data-layer (quantlab.cef_data) — offline known-answer tests.

Pins the pure parsers against fixtures captured from the live api/v3 schema
(2026-06-15) so the network is never touched in CI. Mirrors the perp_data /
borrow / revisions test pattern: fetchers are thin cache wrappers; the parsing
is where bugs hide, so the parsing is what gets pinned.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import cef, cef_data


# --- pricinghistory --------------------------------------------------------- #

# Two real PDI records (oldest + newest) from pricinghistory/PDI/All, plus an
# out-of-order duplicate to prove sorting + dedup.
_PH_PAYLOAD = {
    "Data": {
        "Period": "All",
        "Ticker": "PDI",
        "PriceHistory": [
            {"NAVData": 15.82, "DiscountData": 3.03, "Data": 16.30,
             "DataDate": "2026-06-12T00:00:00"},
            {"NAVData": 23.845, "DiscountData": 4.84, "Data": 25.00,
             "DataDate": "2012-05-25T00:00:00"},
            {"NAVData": 23.845, "DiscountData": 4.84, "Data": 25.00,
             "DataDate": "2012-05-25T00:00:00"},  # dup -> dropped
        ],
    }
}


def test_pricinghistory_parses_price_nav_discount_sorted():
    df = cef_data.parse_pricinghistory(_PH_PAYLOAD)
    assert list(df.columns) == ["price", "nav", "discount_cc"]
    assert df.index.is_monotonic_increasing            # sorted ascending
    assert len(df) == 2                                 # duplicate dropped
    assert df.index[0] == pd.Timestamp("2012-05-25")
    assert df["price"].iloc[-1] == pytest.approx(16.30)
    assert df["nav"].iloc[-1] == pytest.approx(15.82)


def test_cefconnect_discount_matches_our_primitive():
    # CEFConnect's DiscountData must equal quantlab.cef.discount (P-NAV)/NAV,
    # in PERCENT -- this is the cross-check that lets us trust the field.
    df = cef_data.parse_pricinghistory(_PH_PAYLOAD)
    ours = cef.discount(df[["price"]], df[["nav"]].rename(columns={"nav": "price"}))
    assert (ours["price"].iloc[-1] * 100) == pytest.approx(df["discount_cc"].iloc[-1], abs=0.01)


def test_pricinghistory_empty_is_empty_frame_not_error():
    # "Max"/unknown-period tokens return PriceHistory: [] -- a legitimate empty
    # answer, never an exception (the depth sample relies on this).
    df = cef_data.parse_pricinghistory({"Data": {"PriceHistory": []}})
    assert df.empty
    assert list(df.columns) == ["price", "nav", "discount_cc"]
    assert cef_data.parse_pricinghistory({}).empty       # missing keys too


# --- dailypricing snapshot -------------------------------------------------- #

# One real record (IFN) trimmed to the fields the census uses, plus a second
# fund with a missing numeric (null -> NaN) and a stale NAV.
_SNAPSHOT = [
    {"Ticker": "IFN", "Price": 10.98, "NAV": 12.20, "Discount": -10.0,
     "MarketCapUSDm": 468.318, "TotalAssetsUSDm": 520.354,
     "AvgDailyVolume": 1960020, "ExpenseRatio": 1.36,
     "NAVPublished": "2026-06-12T00:00:00", "LastUpdated": "2026-06-12T00:00:00",
     "DistributionFrequency": "Quarterly", "CategoryName": "Asia Equity"},
    {"Ticker": "XYZ", "Price": 5.00, "NAV": 6.00, "Discount": -16.667,
     "MarketCapUSDm": 80.0, "TotalAssetsUSDm": 120.0,
     "AvgDailyVolume": 10000, "ExpenseRatio": None,
     "NAVPublished": "2026-06-05T00:00:00", "LastUpdated": "2026-06-12T00:00:00",
     "DistributionFrequency": "Monthly", "CategoryName": "Senior Loan"},
]


def test_snapshot_indexes_by_ticker_and_types_numerics():
    df = cef_data.parse_snapshot(_SNAPSHOT)
    assert list(df.index) == ["IFN", "XYZ"]            # sorted
    assert df.loc["IFN", "MarketCapUSDm"] == pytest.approx(468.318)
    assert pd.isna(df.loc["XYZ", "ExpenseRatio"])      # null -> NaN, not "None"


def test_snapshot_derives_dollar_adv_and_nav_lag():
    df = cef_data.parse_snapshot(_SNAPSHOT)
    # dollar ADV = shares * price (the spec's >=$250k floor is in dollars)
    assert df.loc["IFN", "dollar_adv"] == pytest.approx(1960020 * 10.98)
    assert df.loc["XYZ", "dollar_adv"] == pytest.approx(10000 * 5.00)
    # NAV staleness: IFN fresh (0d), XYZ NAV is 7 days stale
    assert df.loc["IFN", "nav_lag_days"] == 0
    assert df.loc["XYZ", "nav_lag_days"] == 7


def test_snapshot_discount_matches_primitive():
    df = cef_data.parse_snapshot(_SNAPSHOT)
    # (5-6)/6 = -16.667% -> matches the served Discount field
    computed = (df.loc["XYZ", "Price"] - df.loc["XYZ", "NAV"]) / df.loc["XYZ", "NAV"]
    assert computed * 100 == pytest.approx(df.loc["XYZ", "Discount"], abs=0.01)


def test_snapshot_empty_raises():
    with pytest.raises(ValueError):
        cef_data.parse_snapshot([])
