"""H6 CEF data-loader parsing — known-answer test on a synthetic fixture.

Does NOT touch the network or the licensed cache; pins the pricinghistory
parser (the shape CEFConnect returns) so a vendor format drift fails loudly.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import cef_data


def test_parse_price_history_known_answer():
    payload = {"Data": [
        {"NAVData": 6.47, "DiscountData": -4.33, "Data": 6.19, "DataDate": "2018-01-02T00:00:00"},
        {"NAVData": 6.52, "DiscountData": -4.91, "Data": 6.20, "DataDate": "2018-01-01T00:00:00"},
        # duplicate date -> keep last
        {"NAVData": 6.54, "DiscountData": -4.43, "Data": 6.25, "DataDate": "2018-01-02T00:00:00"},
    ]}
    df = cef_data.parse_price_history(payload)
    assert list(df.columns) == ["px", "nav", "disc"]
    assert df.index.is_monotonic_increasing                 # sorted ascending
    assert df.index[0] == pd.Timestamp("2018-01-01")
    assert len(df) == 2                                      # deduped
    assert df.loc["2018-01-02", "px"] == 6.25               # kept the LAST 1/2 row
    # discount field really is (px-nav)/nav * 100, to the vendor's rounding
    px, nav = df.loc["2018-01-01", "px"], df.loc["2018-01-01", "nav"]
    assert abs((px - nav) / nav * 100 - df.loc["2018-01-01", "disc"]) < 0.1


def test_parse_price_history_empty():
    assert cef_data.parse_price_history({}).empty
    assert list(cef_data.parse_price_history({"Data": []}).columns) == ["px", "nav", "disc"]
