"""H1 quality harness — offline known-answer tests (parsers, features, signal,
machinery gate, source adapters). No network.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import fundamentals as fnd
from quantlab import fundamentals_data as fdat


# --- SEC parsers ----------------------------------------------------------- #

def test_parse_company_concept_filing_date_pit_and_form_filter():
    payload = {"units": {"USD": [
        {"end": "2022-12-31", "val": 100, "form": "10-K", "filed": "2023-02-15"},
        {"end": "2021-12-31", "val": 90, "form": "10-K", "filed": "2023-02-15"},  # same filed, older end
        {"end": "2023-03-31", "val": 110, "form": "10-Q", "filed": "2023-05-01"},
        {"end": "2023-06-30", "val": 120, "form": "8-K", "filed": "2023-07-10"},   # filtered
    ]}}
    s = fdat.parse_company_concept(payload)
    assert list(s.index) == [pd.Timestamp("2023-02-15"), pd.Timestamp("2023-05-01")]
    assert s.loc["2023-02-15"] == 100      # latest period END kept for that filing
    assert s.loc["2023-05-01"] == 110
    assert fdat.parse_company_concept({"units": {}}).empty


def test_parse_ticker_cik_map_zero_pads():
    payload = {"0": {"cik_str": 320193, "ticker": "aapl", "title": "Apple"},
               "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft"}}
    m = fdat.parse_ticker_cik_map(payload)
    assert m["AAPL"] == "0000320193" and m["MSFT"] == "0000789019"


def test_source_adapters_flags_and_slot():
    assert fdat.FreeSECSource().survivorship_safe is False
    assert fdat.CompustatSource().survivorship_safe is True
    with pytest.raises(NotImplementedError):
        fdat.CompustatSource().field_series("AAPL", "assets")


# --- features / signal ----------------------------------------------------- #

def test_gp_and_accruals_over_assets():
    idx = pd.to_datetime(["2022-02-15", "2023-02-15"])
    gp = pd.Series([30.0, 40.0], index=idx)
    assets = pd.Series([100.0, 100.0], index=idx)
    assert list(fnd.gp_over_assets(gp, assets)) == [0.30, 0.40]
    ni = pd.Series([10.0], index=[pd.Timestamp("2023-02-15")])
    cfo = pd.Series([7.0], index=[pd.Timestamp("2023-02-15")])
    a = pd.Series([100.0], index=[pd.Timestamp("2023-02-15")])
    assert fnd.accruals_over_assets(ni, cfo, a).iloc[0] == pytest.approx(0.03)


def test_quality_signal_high_profitability_high_accruals():
    d = pd.to_datetime(["2023-02-15"])
    gp_a = pd.DataFrame({"A": [0.30], "B": [0.15], "C": [0.05]}, index=d)
    sig = fnd.quality_signal(gp_a)
    assert sig.loc[d[0], "A"] > sig.loc[d[0], "B"] > sig.loc[d[0], "C"]
    # accruals penalize: make A accrual-heavy -> its score drops
    acc = pd.DataFrame({"A": [0.20], "B": [0.0], "C": [-0.20]}, index=d)
    sig2 = fnd.quality_signal(gp_a, acc)
    assert sig2.loc[d[0], "A"] < sig.loc[d[0], "A"]


def test_quality_weights_long_high_short_low_dollar_neutral():
    d = pd.to_datetime(["2023-02-15"])
    sig = pd.DataFrame([list(range(10))], index=d, columns=[f"F{i}" for i in range(10)],
                       dtype=float)
    w = fnd.quality_weights(sig, quantile=0.2).loc[d[0]]
    assert w.sum() == pytest.approx(0.0, abs=1e-12)
    assert w["F9"] > 0 and w["F8"] > 0      # highest quality LONG
    assert w["F0"] < 0 and w["F1"] < 0      # lowest quality SHORT


def test_machinery_gate_planted_quality_beats_null():
    gate = fnd.machinery_gate(seeds=(7, 11, 23))
    assert gate["passed"], gate["diffs"]
    assert min(gate["planted_sr"]) > 0.5
    assert max(abs(s) for s in gate["null_sr"]) < 0.6
