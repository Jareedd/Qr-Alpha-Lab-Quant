"""Known-answer tests for quantlab.spot_data -- the spot-kline data layer for
the cash-and-carry feasibility audit. No network: we exercise the pure parser
and the perp->spot symbol mapping on tiny hand-built fixtures with known answers.

These pin (a) that headerless and headered monthly blobs parse identically and
positionally, and (b) the perp->spot ticker mapping (the 1000x-scaled-perp trap).
"""

import io
import os
import sys
import zipfile

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import spot_data


# A tiny, hand-built two-row spot 1d kline blob (the standard 12 Binance fields).
# open_time(ms), open, high, low, close, volume, close_time, quote_volume,
# count, taker_buy_volume, taker_buy_quote_volume, ignore
_DAY1_MS = 1609459200000  # 2021-01-01 00:00:00 UTC
_DAY2_MS = 1609545600000  # 2021-01-02 00:00:00 UTC
_HEADERLESS = (
    f"{_DAY1_MS},100.0,110.0,90.0,105.0,12.0,1609545599999,1260.0,7,6.0,630.0,0\n"
    f"{_DAY2_MS},105.0,120.0,104.0,118.0,20.0,1609631999999,2360.0,9,11.0,1298.0,0\n"
)
_HEADERED = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore\n" + _HEADERLESS
)


def test_parse_headerless_blob_known_answer():
    df = spot_data.parse_kline_csv(_HEADERLESS)
    assert df.shape == (2, 12)
    # positional column names (strings "0".."11")
    assert list(df.columns) == [str(i) for i in range(12)]
    # close is column 4; quote_volume is column 7
    assert float(df["4"].iloc[0]) == 105.0
    assert float(df["4"].iloc[1]) == 118.0
    assert float(df["7"].iloc[1]) == 2360.0


def test_parse_headered_blob_matches_headerless():
    # A header row must be detected and dropped, yielding identical data.
    df_h = spot_data.parse_kline_csv(_HEADERED)
    df_n = spot_data.parse_kline_csv(_HEADERLESS)
    assert df_h.shape == df_n.shape == (2, 12)
    pd.testing.assert_frame_equal(
        df_h.reset_index(drop=True), df_n.reset_index(drop=True)
    )


def test_parse_empty_blob_returns_empty():
    assert spot_data.parse_kline_csv("").empty
    assert spot_data.parse_kline_csv("\n\n").empty


def test_perp_to_spot_symbol_strips_scaled_multiplier():
    # The 1000x-scaled-perp trap: perp 1000PEPEUSDT tracks spot PEPEUSDT.
    assert spot_data.perp_to_spot_symbol("1000PEPEUSDT") == "PEPEUSDT"
    assert spot_data.perp_to_spot_symbol("1000SHIBUSDT") == "SHIBUSDT"
    assert spot_data.perp_to_spot_symbol("1000000MOGUSDT") == "MOGUSDT"
    assert spot_data.perp_to_spot_symbol("100BONKUSDT") == "BONKUSDT"
    # unscaled tickers are identity
    assert spot_data.perp_to_spot_symbol("BTCUSDT") == "BTCUSDT"
    assert spot_data.perp_to_spot_symbol("ETHUSDT") == "ETHUSDT"
    # a leading digit that is part of the name, not a 1000x multiplier, is kept
    assert spot_data.perp_to_spot_symbol("1INCHUSDT") == "1INCHUSDT"


def test_load_spot_klines_reads_cached_parquet(tmp_path):
    """End-to-end of the read path WITHOUT network: pre-seed the cache parquet
    the way _load_raw would write it, then assert load_spot_klines builds the
    date-indexed close/quote_volume frame with the right known values."""
    cache_dir = str(tmp_path)
    # _load_raw caches a positional frame under the SPOT symbol name.
    raw = spot_data.parse_kline_csv(_HEADERLESS)
    start, end = "2019-09-01", "2026-06-01"
    cache_file = os.path.join(cache_dir, f"klines_BTCUSDT_{start}_{end}.parquet")
    raw.to_parquet(cache_file)

    out = spot_data.load_spot_klines("BTCUSDT", start=start, end=end,
                                     cache_dir=cache_dir)
    assert out is not None
    assert list(out.columns) == ["close", "quote_volume"]
    assert out.index[0] == pd.Timestamp("2021-01-01")
    assert out.index[1] == pd.Timestamp("2021-01-02")
    assert float(out["close"].iloc[0]) == 105.0
    assert float(out["close"].iloc[1]) == 118.0
    assert float(out["quote_volume"].iloc[1]) == 2360.0


def test_load_spot_klines_handles_mixed_ms_and_us_timestamps(tmp_path):
    """Regression (2026-06-25): Binance switched kline open_time from MILLISECONDS
    to MICROSECONDS in early 2025, so a symbol spanning that boundary has a cache
    frame that MIXES units in one column. A single ``unit="ms"`` overflowed the µs
    rows to year ~56971 (OutOfBoundsDatetime) and errored ~every symbol. The loader
    must normalize per-row (µs values >= 1e14 -> /1000) and parse both correctly."""
    cache_dir = str(tmp_path)
    day_ms = 1609459200000        # 2021-01-01 00:00 UTC, MILLISECONDS (pre-switch)
    day_us = 1748736000000000     # 2025-06-01 00:00 UTC, MICROSECONDS (post-switch)
    blob = (
        f"{day_ms},100.0,110.0,90.0,105.0,12.0,1609545599999,1260.0,7,6.0,630.0,0\n"
        f"{day_us},105.0,120.0,104.0,118.0,20.0,1748822399999999,2360.0,9,11.0,1298.0,0\n"
    )
    raw = spot_data.parse_kline_csv(blob)
    start, end = "2019-09-01", "2026-06-01"
    raw.to_parquet(os.path.join(cache_dir, f"klines_BTCUSDT_{start}_{end}.parquet"))

    out = spot_data.load_spot_klines("BTCUSDT", start=start, end=end,
                                     cache_dir=cache_dir)
    assert out is not None
    # BOTH rows parse to the right calendar day (no overflow), units normalized.
    assert out.index[0] == pd.Timestamp("2021-01-01")
    assert out.index[1] == pd.Timestamp("2025-06-01")
    assert float(out["close"].iloc[1]) == 118.0


def test_load_spot_klines_maps_scaled_perp_to_spot_cache(tmp_path):
    """Passing a 1000x-scaled PERP ticker must read the UNSCALED spot pair's
    cache (perp_to_spot_symbol applied before lookup)."""
    cache_dir = str(tmp_path)
    raw = spot_data.parse_kline_csv(_HEADERLESS)
    start, end = "2019-09-01", "2026-06-01"
    # cache is keyed by the SPOT symbol (PEPEUSDT), not the perp (1000PEPEUSDT)
    cache_file = os.path.join(cache_dir, f"klines_PEPEUSDT_{start}_{end}.parquet")
    raw.to_parquet(cache_file)

    out = spot_data.load_spot_klines("1000PEPEUSDT", start=start, end=end,
                                     cache_dir=cache_dir)
    assert out is not None
    assert float(out["close"].iloc[1]) == 118.0


def test_load_spot_klines_missing_returns_none(tmp_path):
    """A pair with an empty cached parquet (the never-listed-spot answer
    _load_raw writes) must return None so the caller SKIPS it."""
    cache_dir = str(tmp_path)
    start, end = "2019-09-01", "2026-06-01"
    cache_file = os.path.join(cache_dir, f"klines_NOSUCHUSDT_{start}_{end}.parquet")
    pd.DataFrame().to_parquet(cache_file)
    out = spot_data.load_spot_klines("NOSUCHUSDT", start=start, end=end,
                                     cache_dir=cache_dir)
    assert out is None
