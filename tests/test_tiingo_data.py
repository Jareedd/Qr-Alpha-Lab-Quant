"""Tiingo parsers: known-answer tests, no network."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import tiingo_data as td


def test_parse_eod_prices_adjclose_indexed_sorted():
    rows = [
        {"date": "2024-01-03T00:00:00.000Z", "close": 184.0, "adjClose": 182.0,
         "volume": 1},
        {"date": "2024-01-02T00:00:00.000Z", "close": 185.6, "adjClose": 183.6,
         "volume": 2},
    ]
    s = td.parse_eod_prices(rows)
    assert list(s.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert s.tolist() == [183.6, 182.0]            # adjClose, chronological
    assert s.name == "adjClose" and s.index.name == "date"
    assert s.index.tz is None                       # tz stripped


def test_parse_eod_prices_field_override_and_empty():
    rows = [{"date": "2024-01-02T00:00:00.000Z", "close": 185.6, "adjClose": 183.6}]
    assert td.parse_eod_prices(rows, field="close").tolist() == [185.6]
    assert td.parse_eod_prices([]).empty


def test_parse_supported_tickers_keeps_delisted_with_date_range():
    csv_text = (
        "ticker,exchange,assetType,priceCurrency,startDate,endDate\n"
        "AAPL,NASDAQ,Stock,USD,1980-12-12,2026-06-23\n"
        "ABMD,NASDAQ,Stock,USD,1987-07-30,2023-01-03\n"   # delisted (past endDate)
        "SPY,NYSE ARCA,ETF,USD,1993-01-29,2026-06-23\n"     # not common stock
        "FXTON,NYSE,Stock,EUR,2015-01-01,2018-01-01\n"       # kept (US exch, stock)
    )
    df = td.parse_supported_tickers(csv_text)
    assert set(df["ticker"]) == {"AAPL", "ABMD", "FXTON"}    # ETF dropped
    abmd = df[df["ticker"] == "ABMD"].iloc[0]
    assert abmd["enddate"] == pd.Timestamp("2023-01-03")     # dead name retained
    assert abmd["startdate"] == pd.Timestamp("1987-07-30")


def test_parse_supported_tickers_us_equity_filter_off():
    csv_text = (
        "ticker,exchange,assetType,priceCurrency,startDate,endDate\n"
        "SPY,NYSE ARCA,ETF,USD,1993-01-29,2026-06-23\n"
    )
    df = td.parse_supported_tickers(csv_text, us_equity_only=False)
    assert "SPY" in set(df["ticker"])               # ETF kept when filter off
