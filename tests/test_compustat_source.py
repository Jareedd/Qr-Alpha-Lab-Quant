"""CompustatSource adapter: known-answer tests on synthetic WRDS-format extracts.

No network, no WRDS access — fixtures are tiny CSVs written to tmp_path in the
exact schema the adapter documents. The properties that matter most and are
pinned here: filing-date PIT (and REFUSAL to fall back to period-end),
survivorship safety (dead names retained), correct Compustat-mnemonic mapping,
the gross_profit fallback, and end-to-end integration with the real consumer
(fundamentals.pit_feature_panels).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import fundamentals as fnd
from quantlab.fundamentals_data import CompustatSource


def _write_fundamentals(d, rows, cols):
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(os.path.join(d, "fundamentals.csv"), index=False)


def _standard_extract(d, with_gp=False, with_freq=True):
    cols = ["ticker", "filed"] + (["freq"] if with_freq else []) + \
        ["at", "ni", "oancf", "revt", "cogs"] + (["gp"] if with_gp else [])

    def row(t, filed, freq, at, ni, oancf, revt, cogs, gp=None):
        r = [t, filed] + ([freq] if with_freq else []) + [at, ni, oancf, revt, cogs]
        if with_gp:
            r += [gp]
        return r

    rows = [
        row("AAA", "2020-03-01", "A", 1000, 100, 120, 500, 300, 200),
        row("AAA", "2021-03-01", "A", 1100, 110, 130, 550, 320, 230),
        row("BBB", "2020-03-15", "A", 2000, -50, 10, 800, 600, 200),  # later delists
    ]
    _write_fundamentals(d, rows, cols)


def _write_prices(d):
    dates = pd.bdate_range("2020-01-31", "2021-12-31", freq="BME")
    px = pd.DataFrame({"AAA": np.linspace(100, 140, len(dates)),
                       "BBB": np.linspace(50, 70, len(dates))}, index=dates)
    px.loc[px.index > "2021-06-30", "BBB"] = np.nan      # BBB delists mid-2021
    px.index.name = "date"
    px.to_csv(os.path.join(d, "prices.csv"))


def test_field_series_pit_indexed_and_mapped(tmp_path):
    _standard_extract(str(tmp_path))
    src = CompustatSource(data_dir=str(tmp_path))
    a = src.field_series("AAA", "assets")
    assert list(a.index) == [pd.Timestamp("2020-03-01"), pd.Timestamp("2021-03-01")]
    assert a.tolist() == [1000.0, 1100.0]
    assert a.index.name == "filed"
    assert src.field_series("AAA", "net_income").tolist() == [100.0, 110.0]
    assert src.field_series("AAA", "cfo").tolist() == [120.0, 130.0]
    assert src.field_series("AAA", "revenue").tolist() == [500.0, 550.0]
    assert src.field_series("AAA", "cogs").tolist() == [300.0, 320.0]


def test_gross_profit_revt_minus_cogs_when_no_gp(tmp_path):
    _standard_extract(str(tmp_path), with_gp=False)
    src = CompustatSource(data_dir=str(tmp_path))
    gp = src.field_series("AAA", "gross_profit")
    assert gp.tolist() == [200.0, 230.0]            # 500-300, 550-320


def test_gross_profit_uses_gp_column_when_present(tmp_path):
    _standard_extract(str(tmp_path), with_gp=True)
    src = CompustatSource(data_dir=str(tmp_path))
    # gp column says 200/230 explicitly; ticker AAA revt-cogs happens to match,
    # so use a ticker where they'd differ: overwrite via a custom extract.
    rows_cols = ["ticker", "filed", "at", "ni", "oancf", "revt", "cogs", "gp"]
    _write_fundamentals(str(tmp_path),
                        [["CCC", "2020-03-01", 10, 1, 1, 100, 90, 42]], rows_cols)
    src2 = CompustatSource(data_dir=str(tmp_path))
    assert src2.field_series("CCC", "gross_profit").tolist() == [42.0]  # gp, not 10


def test_missing_filed_column_refuses_lookahead(tmp_path):
    # only datadate (period end), no filing date -> must REFUSE (law #1)
    _write_fundamentals(str(tmp_path),
                        [["AAA", "2020-12-31", 1000, 100, 120, 500, 300]],
                        ["ticker", "datadate", "at", "ni", "oancf", "revt", "cogs"])
    src = CompustatSource(data_dir=str(tmp_path))
    with pytest.raises(ValueError, match="filed"):
        src.field_series("AAA", "assets")


def test_missing_extract_raises_with_guidance(tmp_path):
    src = CompustatSource(data_dir=str(tmp_path))      # empty dir
    with pytest.raises(FileNotFoundError, match="WRDS"):
        src.field_series("AAA", "assets")


def test_annual_only_filters_when_freq_present(tmp_path):
    cols = ["ticker", "filed", "freq", "at", "ni", "oancf", "revt", "cogs"]
    _write_fundamentals(str(tmp_path), [
        ["AAA", "2020-03-01", "A", 1000, 100, 120, 500, 300],
        ["AAA", "2020-08-01", "Q", 1050, 30, 35, 140, 85],   # quarterly
    ], cols)
    src = CompustatSource(data_dir=str(tmp_path))
    annual = src.field_series("AAA", "assets", annual_only=True)
    assert annual.tolist() == [1000.0]                       # Q row excluded
    allf = src.field_series("AAA", "assets", annual_only=False)
    assert allf.tolist() == [1000.0, 1050.0]


def test_survivorship_dead_names_retained(tmp_path):
    _standard_extract(str(tmp_path))
    _write_prices(str(tmp_path))
    src = CompustatSource(data_dir=str(tmp_path))
    assert src.survivorship_safe is True
    assert src.universe() == ["AAA", "BBB"]                  # BBB retained though it dies
    bbb = src.field_series("BBB", "assets")
    assert bbb.tolist() == [2000.0]                          # dead name's history present


def test_prices_pit_reindex_and_delisting(tmp_path):
    _standard_extract(str(tmp_path))
    _write_prices(str(tmp_path))
    src = CompustatSource(data_dir=str(tmp_path))
    asof = pd.bdate_range("2021-01-31", "2021-12-31", freq="BME")
    px = src.prices(["AAA", "BBB"], asof)
    assert list(px.columns) == ["AAA", "BBB"]
    assert px["AAA"].notna().all()                           # survivor priced throughout
    assert px.loc[px.index <= "2021-06-30", "BBB"].notna().all()
    assert px.loc[px.index > "2021-06-30", "BBB"].isna().all()  # NaN after delist
    assert src.end == "2021-12-30" or src.end.startswith("2021-12")


def test_unknown_field_raises(tmp_path):
    _standard_extract(str(tmp_path))
    src = CompustatSource(data_dir=str(tmp_path))
    with pytest.raises(KeyError):
        src.field_series("AAA", "ebitda")


def test_integration_with_pit_feature_panels(tmp_path):
    # the real consumer: pit_feature_panels must assemble GP/A + accruals/A from
    # the adapter, PIT-aligned to month-end asof dates.
    _standard_extract(str(tmp_path))
    src = CompustatSource(data_dir=str(tmp_path))
    asof = pd.bdate_range("2020-06-30", "2021-12-31", freq="BME")
    panels = fnd.pit_feature_panels(src, ["AAA", "BBB"], asof)
    gp_a = panels["gp_a"]
    # AAA GP/A: 200/1000 = 0.20 from the 2020-03 filing, then 230/1100 ≈ 0.209
    # after the 2021-03 filing (forward-filled to each month-end).
    assert gp_a.loc["2020-06-30", "AAA"] == pytest.approx(0.20, abs=1e-9)
    assert gp_a.loc["2021-12-31", "AAA"] == pytest.approx(230 / 1100, abs=1e-9)
    acc = panels["accruals_a"]
    # AAA accruals/A from 2020-03: (ni - cfo)/at = (100-120)/1000 = -0.02
    assert acc.loc["2020-06-30", "AAA"] == pytest.approx(-0.02, abs=1e-9)


# --- review-driven regressions (BLOCKER/MAJOR/MINOR from the 2026-06-24 review) #

def test_prices_wide_parquet_is_not_misread(tmp_path):
    # B1: a natural wide prices.parquet (DatetimeIndex x ticker) must be read as
    # wide -- NOT have its first ticker column consumed as the index and the
    # price values parsed as 1970 dates (a silent empty backtest). The CSV path
    # masked this because to_csv writes the index as a column.
    _standard_extract(str(tmp_path))
    dates = pd.bdate_range("2020-01-31", "2021-12-31", freq="BME")
    px = pd.DataFrame({"AAA": np.linspace(100, 140, len(dates)),
                       "BBB": np.linspace(50, 70, len(dates))}, index=dates)
    px.index.name = "date"
    px.to_parquet(os.path.join(str(tmp_path), "prices.parquet"))
    src = CompustatSource(data_dir=str(tmp_path))
    assert src.end.startswith("2021-12")          # NOT 1970-01-01
    assert set(src.universe()) == {"AAA", "BBB"}   # no ticker dropped into the index
    out = src.prices(["AAA", "BBB"], dates[-3:])
    assert out["AAA"].notna().all() and list(out.columns) == ["AAA", "BBB"]


def test_prices_ticker_first_parquet_refused(tmp_path):
    # a parquet whose first column is a (non-date) ticker and which has no date
    # column must RAISE, not silently parse prices as 1970 dates.
    _standard_extract(str(tmp_path))
    bad = pd.DataFrame({"AAA": [100.0, 101.0], "BBB": [50.0, 51.0]})  # RangeIndex
    bad.to_parquet(os.path.join(str(tmp_path), "prices.parquet"))
    src = CompustatSource(data_dir=str(tmp_path))
    with pytest.raises(ValueError, match="date"):
        src.prices(["AAA"], pd.bdate_range("2020-01-31", periods=2, freq="BME"))


def test_gross_profit_partial_nan_gp_falls_back_per_row(tmp_path):
    # M1: gp is sparse in real funda. A NaN gp row whose revt-cogs IS computable
    # must NOT be dropped (committing to the gp column for all rows loses it).
    cols = ["ticker", "filed", "at", "ni", "oancf", "revt", "cogs", "gp"]
    _write_fundamentals(str(tmp_path), [
        ["AAA", "2020-03-01", 1000, 100, 120, 500, 300, 200],     # gp present
        ["AAA", "2021-03-01", 1100, 110, 130, 550, 320, np.nan],  # gp NaN -> 230
    ], cols)
    src = CompustatSource(data_dir=str(tmp_path))
    assert src.field_series("AAA", "gross_profit").tolist() == [200.0, 230.0]


def test_blank_filed_refused_as_lookahead(tmp_path):
    # M2: a real value with a blank/unparseable filing date is look-ahead (law #1)
    # -- refuse at load, never admit it at a NaT index.
    _write_fundamentals(str(tmp_path), [
        ["AAA", "2020-03-01", 1000, 100, 120, 500, 300],
        ["AAA", "", 9999, 1, 1, 1, 1],                            # blank filed
    ], ["ticker", "filed", "at", "ni", "oancf", "revt", "cogs"])
    src = CompustatSource(data_dir=str(tmp_path))
    with pytest.raises(ValueError, match="look-ahead"):
        src.field_series("AAA", "assets")


def test_prices_offgrid_asof_no_stale_post_delist(tmp_path):
    # m3: an off-grid asof (not on the price index) must NOT ffill a dead name's
    # last price past its delisting -- a survivorship leak.
    _standard_extract(str(tmp_path))
    _write_prices(str(tmp_path))                      # BBB delists after 2021-06-30
    src = CompustatSource(data_dir=str(tmp_path))
    asof = pd.DatetimeIndex(["2021-05-15", "2021-07-15"])  # both OFF the BME grid
    px = src.prices(["AAA", "BBB"], asof)
    assert pd.notna(px.loc["2021-05-15", "BBB"])      # alive -> priced (ffill ok)
    assert pd.isna(px.loc["2021-07-15", "BBB"])       # dead -> NaN, no stale carry
    assert px.loc["2021-07-15", "AAA"] == pytest.approx(px["AAA"].dropna().iloc[-1])


def test_same_filed_duplicate_tiebreak_by_period_end(tmp_path):
    # m4: two rows filed the SAME day -> the LATER period-end (datadate) wins,
    # regardless of row order in the file.
    _write_fundamentals(str(tmp_path), [
        ["AAA", "2021-03-01", "2020-12-31", 1100],    # correct (later period)
        ["AAA", "2021-03-01", "2019-12-31", 999],     # stale, appears LAST in file
    ], ["ticker", "filed", "datadate", "at"])
    src = CompustatSource(data_dir=str(tmp_path))
    assert src.field_series("AAA", "assets").tolist() == [1100.0]
