"""SurvivorshipSafeSECSource — offline known-answer tests. NO NETWORK.

Pins the survivorship recovery that is the whole point of this source: a DEAD
ticker absent from SEC's current ticker->CIK map is recovered via the company
NAME (disambiguated by its S&P removal date), and ``field_series`` then returns
the stubbed filing-date concept series on that recovered CIK — with semantics
identical to FreeSECSource. Also pins survivorship_safe and the PIT universe.

Every network/API-key surface is stubbed: fetch_sp500_tables, the S&P names
table, FreeSECSource.ticker_cik / ._concept_frame, NameCikResolver.operating_cik,
and TiingoSource.prices. No socket is ever opened.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import sec_xwalk_source as sxs
from quantlab import universe as uni

# --- Fixtures: a tiny PIT world with one LIVE and one DEAD name ------------- #

_LIVE = "AAPL"          # in the current ticker->CIK map
_DEAD = "CELG"          # delisted (Celgene, acquired 2019) -> name recovery
_DEAD_NAME = "Celgene Corp"
_DEAD_REMOVAL = pd.Timestamp("2019-11-21")
_LIVE_CIK = "0000320193"
_DEAD_CIK = "0000816284"


def _fake_tables():
    """(current, changes) with AAPL current and CELG removed on a known date."""
    current = pd.DataFrame({"ticker": [_LIVE], "sector": ["Information Technology"]})
    changes = pd.DataFrame(
        {
            "date": [_DEAD_REMOVAL, pd.Timestamp("2015-03-01")],
            "added": [None, _LIVE],
            "removed": [_DEAD, None],
            "reason": ["Acquired by Bristol-Myers Squibb", None],
        }
    )
    return current, changes


def _build(monkeypatch):
    """Construct the source with all construction-time network calls stubbed."""
    monkeypatch.setattr(uni, "fetch_sp500_tables", lambda *a, **k: _fake_tables())
    monkeypatch.setattr(
        sxs, "fetch_sp500_security_names",
        lambda *a, **k: {_LIVE: "Apple Inc.", _DEAD: _DEAD_NAME},
    )
    return sxs.SurvivorshipSafeSECSource(start="2010-01-01", end="2024-01-01")


# --- Concept-frame fakes (stand in for SEC XBRL) --------------------------- #

def _concept_frame_for(self, cik, tag):
    """Fake FreeSECSource._concept_frame (bound-method signature: self, cik, tag):
    only the DEAD company's Assets tag has data, indexed by filing date with
    value/form/end (the real schema)."""
    if cik == _DEAD_CIK and tag == "Assets":
        idx = pd.to_datetime(["2017-02-15", "2018-02-15"])
        return pd.DataFrame(
            {
                "value": [1000.0, 1100.0],
                "form": ["10-K", "10-K"],
                "end": ["2016-12-31", "2017-12-31"],
            },
            index=idx,
        )
    return pd.DataFrame(columns=["value", "form", "end"])


def _patch_collaborators(monkeypatch, src):
    """Stub the lazy collaborators' methods on the instances they create."""
    # FreeSECSource: AAPL maps (current), CELG does NOT (the hole).
    monkeypatch.setattr(
        type(src.freesec), "ticker_cik",
        lambda self, t: {_LIVE: _LIVE_CIK}.get(t.upper()),
    )
    monkeypatch.setattr(type(src.freesec), "_concept_frame", _concept_frame_for)
    # NameCikResolver: name recovery returns the dead operating CIK, and the
    # removal date must be passed through as the dead_by disambiguator.
    seen = {}

    def _operating_cik(self, name, dead_by=None):
        seen["name"] = name
        seen["dead_by"] = dead_by
        return _DEAD_CIK if "celgene" in name.lower() else None

    monkeypatch.setattr(type(src.resolver), "operating_cik", _operating_cik)
    return seen


# --- Tests ----------------------------------------------------------------- #

def test_survivorship_safe_flag_true():
    assert sxs.SurvivorshipSafeSECSource.survivorship_safe is True


def test_universe_is_pit_member_set(monkeypatch):
    src = _build(monkeypatch)
    # CELG was a member (it was removed in-window) and AAPL is current -> both.
    assert set(src.universe()) == {_LIVE, _DEAD}


def test_live_ticker_resolves_via_current_map(monkeypatch):
    src = _build(monkeypatch)
    _patch_collaborators(monkeypatch, src)
    assert src._cik_for(_LIVE) == _LIVE_CIK


def test_dead_ticker_recovered_via_name_crosswalk(monkeypatch):
    src = _build(monkeypatch)
    seen = _patch_collaborators(monkeypatch, src)
    # CELG is NOT in the current map -> must be recovered by name.
    assert src._cik_for(_DEAD) == _DEAD_CIK
    assert seen["name"] == _DEAD_NAME
    # the S&P removal date is passed as the dead_by disambiguator (not as data).
    assert seen["dead_by"] == str(_DEAD_REMOVAL.date())


def test_field_series_on_recovered_cik_matches_freesec_semantics(monkeypatch):
    src = _build(monkeypatch)
    _patch_collaborators(monkeypatch, src)
    s = src.field_series(_DEAD, "assets")
    assert list(s.index) == [pd.Timestamp("2017-02-15"), pd.Timestamp("2018-02-15")]
    assert list(s.values) == [1000.0, 1100.0]
    assert s.index.name == "filed"


def test_field_series_annual_only_filters_forms(monkeypatch):
    src = _build(monkeypatch)
    _patch_collaborators(monkeypatch, src)
    # The fake Assets frame is all 10-K, so annual_only keeps both rows.
    s = src.field_series(_DEAD, "assets", annual_only=True)
    assert len(s) == 2


def test_field_series_empty_when_unresolvable(monkeypatch):
    src = _build(monkeypatch)
    _patch_collaborators(monkeypatch, src)
    # An unknown ticker resolves to no CIK -> empty series, no crash.
    src._names["ZZZZ"] = "Nonexistent Holdings"
    src._members = list(src._members) + ["ZZZZ"]
    s = src.field_series("ZZZZ", "assets")
    assert s.empty


def test_cik_resolution_is_cached(monkeypatch):
    src = _build(monkeypatch)
    seen = _patch_collaborators(monkeypatch, src)
    calls = {"n": 0}
    orig = type(src.resolver).operating_cik

    def _counting(self, name, dead_by=None):
        calls["n"] += 1
        return orig(self, name, dead_by=dead_by)

    monkeypatch.setattr(type(src.resolver), "operating_cik", _counting)
    src._cik_cache.clear()
    src._cik_for(_DEAD)
    src._cik_for(_DEAD)
    assert calls["n"] == 1            # second lookup served from cache


def test_prices_reindexes_to_asof_grid(monkeypatch):
    src = _build(monkeypatch)
    _patch_collaborators(monkeypatch, src)

    daily = pd.DataFrame(
        {_LIVE: [10.0, 11.0, 12.0], _DEAD: [5.0, 5.5, 6.0]},
        index=pd.to_datetime(["2018-01-31", "2018-02-28", "2018-03-31"]),
    )

    class _FakeTiingo:
        def prices(self, tickers, start, end):
            return daily[[c for c in daily.columns if c in tickers]]

    src._tiingo = _FakeTiingo()
    asof = pd.to_datetime(["2018-02-15", "2018-03-15"])
    px = src.prices([_LIVE, _DEAD], asof)
    assert list(px.index) == list(asof)
    # forward-filled from the last daily print on/before each asof date.
    assert px.loc["2018-02-15", _LIVE] == 10.0      # ffill of 2018-01-31
    assert px.loc["2018-03-15", _DEAD] == 5.5       # ffill of 2018-02-28


def test_start_end_properties(monkeypatch):
    src = _build(monkeypatch)
    assert src.start == "2010-01-01"
    assert src.end == "2024-01-01"
