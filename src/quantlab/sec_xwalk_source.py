"""Survivorship-SAFE free fundamentals source for H1 — SEC XBRL + name crosswalk.

The wall (audit 2026-06-14): SEC's free ticker->CIK map (``company_tickers.json``)
is CURRENT-ONLY, so dead/renamed S&P names (Celgene, Monsanto, Xilinx, ...) are
unmapped — ~73% coverage, the survivorship hole reprised. ``FreeSECSource`` is
therefore declared survivorship-BLOCKED and the H1 runner refuses a graded trial
on it BY DESIGN.

This module closes that hole WITHOUT paid data by composing three existing,
separately-tested pieces — it re-implements none of their logic:

1. ``FreeSECSource``      — the SEC XBRL company-facts reader (concept frames,
                            current ticker->CIK map, form filters, tag fallbacks).
2. ``NameCikResolver``    — the name-keyed crosswalk that recovers dead operating
                            entities (94% of dead names per the latest audit),
                            gating out namesakes/ticker-reusers via filing history.
3. ``TiingoSource``       — delisting-inclusive EOD prices (dead names carry
                            history to their final print).

The recovery: when the current ticker->CIK map has no entry (the dead-name case),
fall back to the company NAME and resolve via ``NameCikResolver.operating_cik``,
disambiguating the dead operating entity with its S&P index-removal date.

LAW #1 (no lookahead). Fundamentals enter strictly by FILING date — every
``field_series`` value is ``filed``-indexed, identical to ``FreeSECSource``. The
index-removal date is used ONLY to pick the correct dead operating CIK among
namesakes (it disambiguates the entity, it is never a feature and never gates
data values). Prices are reindexed forward-filled to the rebalance grid.

Because each dead name now resolves to a real, filing-date-indexed history,
``survivorship_safe = True`` and the H1 DATA GATE passes — a FREE, clean trial
#12 the day this is run with trials enabled. (Build-only here: ZERO trials.)
"""
from __future__ import annotations

import pandas as pd

from quantlab import universe
from quantlab.cik_crosswalk import NameCikResolver, fetch_sp500_security_names
from quantlab.fundamentals_data import (
    ANNUAL_FORMS,
    CACHE,
    FIELD_CONCEPTS,
    PERIODIC_FORMS,
    FreeSECSource,
    FundamentalsSource,
    _read_concept_frame,
)
from quantlab.tiingo_data import TiingoSource


class SurvivorshipSafeSECSource(FundamentalsSource):
    """Free, survivorship-safe fundamentals: SEC XBRL keyed by a name-recovered
    CIK, with delisting-inclusive Tiingo prices and a point-in-time S&P 500
    member list. Semantics of ``field_series`` are identical to
    ``FreeSECSource`` — only CIK resolution differs (current map first, then the
    dead-name crosswalk)."""

    survivorship_safe = True

    def __init__(
        self,
        start: str = "2010-01-01",
        end: str | None = None,
        cache_dir: str = CACHE,
    ):
        """Build PIT membership and the dead-name resolution scaffolding.

        Assumptions:
        - ``start`` bounds the PIT membership reconstruction and price pulls.
        - ``end`` defaults to today (UTC-naive) so prices span the full window.
        - Network-touching members (FreeSECSource map, NameCikResolver index,
          TiingoSource) are constructed LAZILY so this object — and the tests —
          need no network or API key until a field/price is actually requested.
        """
        self._start = pd.Timestamp(start)
        self._end = pd.Timestamp(end) if end is not None else pd.Timestamp.utcnow().normalize()
        self.cache_dir = cache_dir

        # Lazy collaborators (network / API key only on first real use).
        self._freesec: FreeSECSource | None = None
        self._resolver: NameCikResolver | None = None
        self._tiingo: TiingoSource | None = None

        # PIT S&P 500 membership intervals (point-in-time safe: change effective
        # dates were announced in advance).
        current, changes = universe.fetch_sp500_tables()
        self._intervals = universe.build_membership_intervals(
            current, changes, start=str(self._start.date())
        )
        self._members = universe.all_members_in_window(self._intervals)

        # {ticker: removal_date} — the LATEST date a ticker appears in 'removed'.
        # Used ONLY to disambiguate the dead operating entity (never as data).
        removed: dict[str, pd.Timestamp] = {}
        for _, row in changes.iterrows():
            tkr = row.get("removed")
            date = row.get("date")
            if pd.notna(tkr) and pd.notna(date):
                date = pd.Timestamp(date)
                if tkr not in removed or date > removed[tkr]:
                    removed[tkr] = date
        self._removal_date = removed

        # {ticker: company name} — current constituents PLUS dead names from the
        # changes table (the free source of departed companies' names).
        self._names = fetch_sp500_security_names()

        # Cache of resolved CIKs (ticker -> cik|None) to avoid repeat work.
        self._cik_cache: dict[str, str | None] = {}

    # ------------------------------------------------------------------ #
    # Lazy collaborators.
    # ------------------------------------------------------------------ #

    @property
    def freesec(self) -> FreeSECSource:
        if self._freesec is None:
            self._freesec = FreeSECSource(cache_dir=self.cache_dir)
        return self._freesec

    @property
    def resolver(self) -> NameCikResolver:
        if self._resolver is None:
            self._resolver = NameCikResolver(cache_dir=self.cache_dir)
        return self._resolver

    @property
    def tiingo(self) -> TiingoSource:
        if self._tiingo is None:
            self._tiingo = TiingoSource()
        return self._tiingo

    # ------------------------------------------------------------------ #
    # CIK resolution — the survivorship recovery.
    # ------------------------------------------------------------------ #

    def _cik_for(self, ticker: str) -> str | None:
        """Resolve ``ticker`` to a CIK. Current SEC ticker->CIK map FIRST (the
        live, unambiguous case); on a miss (the dead/renamed name) fall back to
        the company NAME via ``NameCikResolver.operating_cik``, disambiguating
        the dead operating entity with its S&P removal date. Cached. ``None`` if
        neither path resolves."""
        tkr = ticker.upper()
        if tkr in self._cik_cache:
            return self._cik_cache[tkr]

        cik = self.freesec.ticker_cik(tkr)               # current map first
        if cik is None:
            name = self._names.get(tkr)
            if name is not None:
                dead_by = self._removal_date.get(tkr)
                dead_by_str = str(dead_by.date()) if dead_by is not None else None
                cik = self.resolver.operating_cik(name, dead_by=dead_by_str)

        self._cik_cache[tkr] = cik
        return cik

    # ------------------------------------------------------------------ #
    # FundamentalsSource interface.
    # ------------------------------------------------------------------ #

    def field_series(
        self, ticker: str, field: str, *, annual_only: bool = False
    ) -> pd.Series:
        """Filing-date-indexed values for a logical field — same semantics as
        ``FreeSECSource.field_series``, but on the survivorship-recovered CIK.

        Iterate ``FIELD_CONCEPTS[field]`` (namespace, tag, unit) candidates, read
        each via ``FreeSECSource._concept_frame`` (us-gaap/USD positionally, other
        namespaces — e.g. dei shares — by keyword), filter to ``ANNUAL_FORMS`` if
        ``annual_only`` else ``PERIODIC_FORMS``, and return the first non-empty
        ``filed``-indexed ``value`` series. Empty Series if the CIK does not
        resolve or no concept carries data."""
        cik = self._cik_for(ticker)
        if cik is None:
            return pd.Series(dtype=float, name="value")     # truly unrecoverable
        forms = ANNUAL_FORMS if annual_only else PERIODIC_FORMS
        for namespace, tag, unit in FIELD_CONCEPTS[field]:
            frame = _read_concept_frame(self.freesec, cik, tag, namespace, unit)
            if frame.empty:
                s = pd.Series(dtype=float, name="value")
            else:
                filtered = frame[frame["form"].isin(forms)]
                s = filtered["value"].copy()
                s.index.name = "filed"
            if not s.empty:
                return s
        return pd.Series(dtype=float, name="value")

    def universe(self) -> list[str]:
        """All point-in-time S&P 500 members in the window (survivorship-safe:
        includes names later removed)."""
        return list(self._members)

    def _daily_prices(self, universe: list[str]) -> pd.DataFrame:
        """Raw wide (date x ticker) DAILY adjusted-price frame from Tiingo over
        [start, end], delisting-inclusive, restricted to tickers Tiingo carries.
        The shared pull behind ``prices`` (quarterly grid) and ``prices_monthly``
        (month-end grid) so neither re-fetches and they agree on the universe."""
        wide = self.tiingo.prices(
            list(universe), str(self._start.date()), str(self._end.date())
        )
        if wide.empty:
            return pd.DataFrame()
        cols = [c for c in wide.columns if c in set(universe) or c.upper() in {u.upper() for u in universe}]
        return wide[cols] if cols else wide

    def prices(self, universe: list[str], asof: pd.DatetimeIndex) -> pd.DataFrame:
        """Wide (asof x ticker) adjusted-price frame: delisting-inclusive Tiingo
        EOD over [start, end], reindexed forward-filled to the rebalance grid
        ``asof``, restricted to tickers Tiingo actually carries."""
        wide = self._daily_prices(universe)
        if wide.empty:
            return pd.DataFrame(index=asof)
        return wide.reindex(asof, method="ffill")

    def prices_monthly(self, universe: list[str]) -> pd.DataFrame:
        """Wide (month-end x ticker) DAILY->monthly price grid — the LAST daily
        adjusted close in each calendar month, NOT reindexed onto the quarterly
        ``asof`` grid.

        Why this exists (B3): the HML value-loading is a trailing rolling beta of
        MONTHLY returns. ``prices`` is already reindexed to the QUARTERLY rebalance
        grid, so resampling THAT to month-end yields a sparse, mostly-NaN series ->
        all-NaN HML betas -> the NEUTRAL arm silently degenerates to a plain demean
        (NEUTRAL == RAW, a false 'not value-collinear' read). The HML regression
        must see the genuine monthly return series, built from the daily grid
        BEFORE any quarterly reindex. Point-in-time: month-end m carries the close
        known at m; nothing peeks forward."""
        wide = self._daily_prices(universe)
        if wide.empty:
            return pd.DataFrame()
        return wide.resample("ME").last()

    # ------------------------------------------------------------------ #
    # Window properties (consumed by the runner).
    # ------------------------------------------------------------------ #

    @property
    def start(self) -> str:
        return str(self._start.date())

    @property
    def end(self) -> str:
        return str(self._end.date())
