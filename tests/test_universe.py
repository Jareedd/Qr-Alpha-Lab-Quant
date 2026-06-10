"""Point-in-time membership reconstruction: tested offline with a known history.

No network in tests -- a hand-built changes table with a known answer key
exercises the backward walk, interval boundaries (effective date inclusive),
ticker normalization, and the coverage report.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import metrics, universe


def _toy_history():
    # Today: {A, B, C}. Walking backward:
    #   2020-06-01: C added, D removed  -> before: {A, B, D}
    #   2015-03-02: B added, E removed  -> before: {A, D, E}
    current = pd.DataFrame({"ticker": ["A", "B", "C"]})
    changes = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-06-01", "2015-03-02"]),
            "added": ["C", "B"],
            "removed": ["D", "E"],
        }
    )
    return current, changes


def test_backward_reconstruction_known_answer():
    current, changes = _toy_history()
    iv = universe.build_membership_intervals(current, changes, start="2012-01-01")
    dates = pd.DatetimeIndex(["2013-01-02", "2016-01-04", "2021-01-04"])
    mask = universe.membership_mask(dates, pd.Index(list("ABCDE")), iv)

    assert set(mask.columns[mask.loc["2013-01-02"]]) == {"A", "D", "E"}
    assert set(mask.columns[mask.loc["2016-01-04"]]) == {"A", "B", "D"}
    assert set(mask.columns[mask.loc["2021-01-04"]]) == {"A", "B", "C"}


def test_effective_date_is_inclusive_for_the_add():
    current, changes = _toy_history()
    iv = universe.build_membership_intervals(current, changes, start="2012-01-01")
    on_change = pd.DatetimeIndex(["2020-05-29", "2020-06-01"])
    mask = universe.membership_mask(on_change, pd.Index(list("ABCDE")), iv)
    assert not mask.loc["2020-05-29", "C"] and mask.loc["2020-05-29", "D"]
    assert mask.loc["2020-06-01", "C"] and not mask.loc["2020-06-01", "D"]


def test_one_sided_changes_handled():
    # Index size changes / data gaps produce add-only or remove-only rows.
    current = pd.DataFrame({"ticker": ["A", "B"]})
    changes = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-01-02", "2016-01-04"]),
            "added": ["B", None],
            "removed": [None, "Z"],
        }
    )
    iv = universe.build_membership_intervals(current, changes, start="2015-01-01")
    dates = pd.DatetimeIndex(["2015-06-01", "2017-01-03", "2019-01-02"])
    mask = universe.membership_mask(dates, pd.Index(["A", "B", "Z"]), iv)
    assert set(mask.columns[mask.loc["2015-06-01"]]) == {"A", "Z"}
    assert set(mask.columns[mask.loc["2017-01-03"]]) == {"A"}
    assert set(mask.columns[mask.loc["2019-01-02"]]) == {"A", "B"}


def test_ticker_normalization_matches_yfinance_style():
    assert universe._normalize_ticker("BRK.B") == "BRK-B"
    assert universe._normalize_ticker(" bf.b ") == "BF-B"


def test_all_members_and_coverage_report():
    current, changes = _toy_history()
    iv = universe.build_membership_intervals(current, changes, start="2012-01-01")
    members = universe.all_members_in_window(iv)
    assert members == ["A", "B", "C", "D", "E"]

    prices = pd.DataFrame(
        {"A": [1.0, 2.0], "B": [1.0, 2.0], "C": [np.nan, np.nan]},
        index=pd.DatetimeIndex(["2020-01-02", "2020-01-03"]),
    )
    cov = universe.coverage_report(members, prices)
    assert cov["n_members_ever"] == 5
    assert cov["n_with_price_data"] == 2  # C exists but is all-NaN
    assert cov["missing_tickers"] == ["C", "D", "E"]


def test_newey_west_corrects_overlap_inflation():
    rng = np.random.default_rng(0)
    # IID series: NW t-stat ~ naive t-stat.
    iid = pd.Series(rng.normal(0.02, 1.0, 4000))
    naive = iid.mean() / iid.sem()
    nw = metrics.newey_west_tstat(iid, lags=21)
    assert abs(nw - naive) / abs(naive) < 0.15

    # 21-day moving sum: massive positive autocorrelation (like overlapping
    # 21d labels). NW must shrink the t-stat substantially; naive overstates.
    overlap = pd.Series(rng.normal(0.02, 1.0, 4000)).rolling(21).sum().dropna()
    naive_o = overlap.mean() / overlap.sem()
    nw_o = metrics.newey_west_tstat(overlap, lags=21)
    assert nw_o < 0.5 * naive_o
