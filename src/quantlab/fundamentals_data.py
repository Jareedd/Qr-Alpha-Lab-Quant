"""SEC XBRL fundamentals layer for H1 (quality tilts) — filing-date point-in-time.

Free and authoritative: SEC's XBRL company-facts API. The hard part of H1 is PIT
safety — a fundamental may only enter the universe by its FILING date (when the
market could first know it), never its period END. Every series here is indexed
by ``filed``.

Logical fields assembled (with tag fallbacks, since issuers tag inconsistently):
Assets, NetIncome, CFO, Revenue, CostOfGoodsSold → features GP/A and accruals/A
(in quantlab.fundamentals).

**KNOWN LIMITATION, measured in the H1 audit (research_log 2026-06-14):** SEC's
``company_tickers.json`` map is CURRENT-ONLY, so dead/renamed names are unmapped
(~73% coverage) — the survivorship hole, reprised. A survivorship-SAFE H1 run
therefore needs a historical ticker→CIK map or a paid source (Compustat/CRSP).
This module exposes a ``FundamentalsSource`` interface with a free-SEC
implementation (research/dev, survivorship-limited) and a ``CompustatSource``
adapter SLOT for when WRDS access lands. The H1 runner refuses a real trial on
the free source BY DESIGN — building the machine now so it is one command from a
clean trial #12 the day institutional data arrives.
"""
from __future__ import annotations

import gzip
import json
import os
import socket
import time
import urllib.error
import urllib.request

import pandas as pd

socket.setdefaulttimeout(120)
_UA = "qr-alpha-lab research Jared@how.co"          # SEC fair-access: real contact
CACHE = os.path.join("data_cache", "fundamentals")
_MIN_INTERVAL = 0.12
_last = [0.0]

# Logical field -> ordered us-gaap tag candidates (first that has data wins).
FIELD_TAGS = {
    "assets": ["Assets"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues", "SalesRevenueNet"],
    "cogs": ["CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfRevenue"],
    "gross_profit": ["GrossProfit"],
}
PERIODIC_FORMS = ("10-K", "10-Q", "10-K/A", "10-Q/A")
ANNUAL_FORMS = ("10-K", "10-K/A")


def _get(url: str, timeout: int = 60, retries: int = 4) -> bytes:
    import time as _t
    req = urllib.request.Request(
        url, headers={"User-Agent": _UA, "Accept-Encoding": "gzip, deflate"})
    for attempt in range(retries):
        wait = _MIN_INTERVAL - (_t.monotonic() - _last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
            _last[0] = _t.monotonic()
            return raw
        except urllib.error.HTTPError as e:
            _last[0] = _t.monotonic()
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
        except urllib.error.URLError:
            _last[0] = _t.monotonic()
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------- #
# Pure parsers (no network) — pinned by tests.
# --------------------------------------------------------------------------- #

def parse_company_concept_frame(
    payload: dict, forms: tuple[str, ...] = PERIODIC_FORMS,
) -> pd.DataFrame:
    """SEC ``companyconcept`` JSON → filing-date-indexed rows with ``value``,
    ``form``, and ``end`` columns.

    The free SEC path needs this metadata so H1 can request ANNUAL-ONLY flow
    numerators (10-K/10-KA) while leaving assets as every known stock value.
    When a filing date carries several period values, keep the row with the
    latest period ``end`` — the freshest figure disclosed in that filing.
    """
    units = (payload or {}).get("units", {}) or {}
    rows = units.get("USD") or []
    recs = []
    for r in rows:
        if r.get("form") in forms and r.get("filed") and r.get("val") is not None:
            recs.append(
                (r["filed"], r.get("end", ""), r["form"], float(r["val"]))
            )
    if not recs:
        return pd.DataFrame(columns=["value", "form", "end"])
    df = pd.DataFrame(
        recs, columns=["filed", "end", "form", "value"]
    ).sort_values(["filed", "end"])
    df = df.drop_duplicates("filed", keep="last")
    out = df.set_index(pd.to_datetime(df["filed"]))[["value", "form", "end"]]
    out.index.name = "filed"
    return out[~out.index.duplicated(keep="last")].sort_index()


def parse_company_concept(
    payload: dict, forms: tuple[str, ...] = PERIODIC_FORMS,
) -> pd.Series:
    """SEC ``companyconcept`` JSON → filing-date-indexed USD values.

    Kept as a Series wrapper for callers/tests that only need the values.
    """
    frame = parse_company_concept_frame(payload, forms=forms)
    if frame.empty:
        return pd.Series(dtype=float, name="value")
    out = frame["value"].copy()
    out.index.name = "filed"
    return out


def parse_ticker_cik_map(payload: dict) -> dict[str, str]:
    """SEC ``company_tickers.json`` → {TICKER: zero-padded-CIK}. CURRENT-ONLY —
    the survivorship limitation documented above."""
    out = {}
    for rec in (payload or {}).values():
        t, cik = rec.get("ticker"), rec.get("cik_str")
        if t and cik is not None:
            out[t.upper()] = str(int(cik)).zfill(10)
    return out


# --------------------------------------------------------------------------- #
# Source interface + implementations.
# --------------------------------------------------------------------------- #

class FundamentalsSource:
    """Interface the H1 harness consumes. ``field_series(ticker, field)`` returns
    a filing-date-indexed Series for a logical field in FIELD_TAGS."""

    survivorship_safe: bool = False

    def field_series(
        self, ticker: str, field: str, *, annual_only: bool = False
    ) -> pd.Series:  # pragma: no cover
        raise NotImplementedError


class FreeSECSource(FundamentalsSource):
    """Free SEC XBRL. Research/dev only — **survivorship-limited** (current-only
    ticker→CIK map; ~73% coverage per the audit). NOT valid for a graded trial."""

    survivorship_safe = False

    def __init__(self, cache_dir: str = CACHE):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._map: dict[str, str] | None = None

    def ticker_cik(self, ticker: str) -> str | None:
        if self._map is None:
            path = os.path.join(self.cache_dir, "company_tickers.json")
            if os.path.exists(path):
                payload = json.loads(open(path, encoding="utf-8").read())
            else:
                payload = json.loads(_get("https://www.sec.gov/files/company_tickers.json"))
                open(path, "w", encoding="utf-8").write(json.dumps(payload))
            self._map = parse_ticker_cik_map(payload)
        return self._map.get(ticker.upper())

    def _concept_frame(self, cik: str, tag: str) -> pd.DataFrame:
        path = os.path.join(self.cache_dir, f"cc_{cik}_{tag}.parquet")
        if os.path.exists(path):
            cached = pd.read_parquet(path)
            if {"value", "form", "end"}.issubset(cached.columns):
                cached.index.name = "filed"
                return cached[["value", "form", "end"]].sort_index()
        url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
        try:
            frame = parse_company_concept_frame(json.loads(_get(url)))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                frame = pd.DataFrame(columns=["value", "form", "end"])
            else:
                raise
        frame.to_parquet(path)
        return frame

    def field_series(
        self, ticker: str, field: str, *, annual_only: bool = False
    ) -> pd.Series:
        cik = self.ticker_cik(ticker)
        if cik is None:
            return pd.Series(dtype=float, name="value")     # unmapped (the hole)
        forms = ANNUAL_FORMS if annual_only else PERIODIC_FORMS
        for tag in FIELD_TAGS[field]:
            frame = self._concept_frame(cik, tag)
            if frame.empty:
                s = pd.Series(dtype=float, name="value")
            else:
                filtered = frame[frame["form"].isin(forms)]
                s = filtered["value"].copy()
                s.index.name = "filed"
            if not s.empty:
                return s
        return pd.Series(dtype=float, name="value")


# Logical field -> Compustat (funda) mnemonic. gross_profit falls back to
# revt - cogs when the `gp` column is absent (matches FreeSECSource._gross_profit).
COMPUSTAT_MNEMONIC = {
    "assets": "at",          # Assets - Total
    "net_income": "ni",      # Net Income (Loss)
    "cfo": "oancf",          # Operating Activities - Net Cash Flow
    "revenue": "revt",       # Revenue - Total  (alt: sale)
    "cogs": "cogs",          # Cost of Goods Sold
    "gross_profit": "gp",    # Gross Profit (often absent -> revt - cogs)
}


class CompustatSource(FundamentalsSource):
    """Survivorship-safe, filing-date-PIT fundamentals from WRDS Compustat/CRSP,
    read from LOCAL EXTRACTS (no live WRDS API needed — you export once and drop
    the files in ``data_dir``). Dead/renamed names are RETAINED (that is the whole
    point vs free SEC); fundamentals enter only by FILING date, never period end.

    Expected extracts in ``data_dir`` (parquet preferred, CSV accepted):

      fundamentals.parquet  — long, one row per (firm, filing). REQUIRED columns:
        ``ticker``  : security ticker (matched upper-cased to the PIT universe)
        ``filed``   : the FILING / point-in-time AVAILABILITY date (NOT
                      ``datadate``/period-end — using period-end is look-ahead and
                      is REFUSED, law #1). From Compustat PIT/Snapshot, or merge
                      ``rdq`` from fundq, or the 10-K filing date.
        ``at``, ``ni``, ``oancf``, ``revt``, ``cogs``  : Compustat mnemonics.
        OPTIONAL: ``gp`` (gross profit; else computed revt - cogs),
                  ``freq`` ('A'/'Q'; when present, ``annual_only`` keeps 'A').
      prices.parquet        — wide (date index x ticker), delisting-inclusive
                              adjusted prices from CRSP (dead names carry history
                              to their delist date, then NaN).

    WRDS recipe (documented so the export is reproducible):
      - Fundamentals: Compustat ``funda`` (consol=C, indfmt=INDL, datafmt=STD,
        popsrc=D) joined to a filing date — PIT-Snapshot ``pitdate``, or ``rdq``
        from ``fundq`` keyed on (gvkey, datadate). gvkey->ticker via the CRSP/
        Compustat Merged link (``ccmxpf_lnkhist``), keeping dead gvkeys.
      - Prices: CRSP ``msf``/``dsf`` total-return-adjusted price, with delisting
        returns (``msedelist``) applied — survivorship-safe by construction.

    The H1 runner's DATA GATE passes (survivorship_safe=True); if the extracts are
    absent it fails LOUDLY at first use with the export instructions, never
    silently."""

    survivorship_safe = True

    def __init__(
        self, data_dir: str = os.path.join("data_cache", "compustat"),
        start: str | None = None, end: str | None = None,
    ):
        self.data_dir = data_dir
        self._fund: pd.DataFrame | None = None
        self._px: pd.DataFrame | None = None
        self._start = start
        self._end = end

    # -- extract loaders (clear failures; no silent fallback) ----------------- #
    def _read(self, stem: str) -> pd.DataFrame:
        pq = os.path.join(self.data_dir, f"{stem}.parquet")
        csv = os.path.join(self.data_dir, f"{stem}.csv")
        if os.path.exists(pq):
            return pd.read_parquet(pq)
        if os.path.exists(csv):
            return pd.read_csv(csv)
        raise FileNotFoundError(
            f"Compustat '{stem}' extract not found at {pq} or {csv}. Export it "
            "from WRDS (see CompustatSource docstring for the exact funda/CRSP "
            "recipe) and drop it in the data_dir. This is the only thing standing "
            "between the proven H1 harness and trial #12.")

    def _fundamentals(self) -> pd.DataFrame:
        if self._fund is None:
            df = self._read("fundamentals")
            df.columns = [str(c).lower() for c in df.columns]
            if "ticker" not in df.columns:
                raise ValueError("fundamentals extract lacks a 'ticker' column.")
            if "filed" not in df.columns:
                raise ValueError(
                    "fundamentals extract lacks a 'filed' (filing/PIT availability "
                    "date) column. Refusing to fall back to datadate/period-end — "
                    "that is look-ahead (law #1). Supply filing dates (Compustat "
                    "PIT/Snapshot, rdq from fundq, or 10-K filing date).")
            df["ticker"] = df["ticker"].astype(str).str.upper()
            df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
            bad = df["filed"].isna()
            if bad.any():
                raise ValueError(
                    f"{int(bad.sum())} fundamentals rows have missing/unparseable "
                    "'filed' dates; a value with no known availability date is "
                    "look-ahead (law #1). Fix the extract's filing dates.")
            self._fund = df
        return self._fund

    def _prices(self) -> pd.DataFrame:
        if self._px is None:
            df = self._read("prices")
            if isinstance(df.index, pd.DatetimeIndex):
                pass  # already wide (date index x ticker) -- the preferred parquet form
            else:
                cols = {str(c).lower(): c for c in df.columns}
                if {"date", "ticker"}.issubset(cols) and ("prc" in cols or "price" in cols):
                    val = cols.get("prc") or cols.get("price")
                    df = df.pivot_table(index=cols["date"], columns=cols["ticker"], values=val)
                elif "date" in cols:
                    df = df.set_index(cols["date"])
                else:
                    # last resort: first column as the date index, but REFUSE if it
                    # is not date-like. A NUMERIC column is the trap: pd.to_datetime
                    # turns price floats into 1970 epoch-nanosecond dates (valid, not
                    # NaT), which would silently empty the backtest -- so reject
                    # numeric first columns outright, and string junk via the NaN test.
                    first = df.columns[0]
                    col = df[first]
                    idx = pd.to_datetime(col, errors="coerce")
                    if pd.api.types.is_numeric_dtype(col) or idx.isna().mean() > 0.5:
                        raise ValueError(
                            "prices extract: first column is not date-like and no "
                            "'date'/'ticker' columns found; refusing to misread a "
                            "ticker column as the date index.")
                    df = df.set_index(first)
            df.index = pd.to_datetime(df.index)
            df.columns = [str(c).upper() for c in df.columns]
            self._px = df.sort_index()
        return self._px

    # -- FundamentalsSource interface ----------------------------------------- #
    def field_series(
        self, ticker: str, field: str, *, annual_only: bool = False
    ) -> pd.Series:
        if field not in COMPUSTAT_MNEMONIC:
            raise KeyError(f"unknown field {field!r}; expected one of {list(COMPUSTAT_MNEMONIC)}")
        fund = self._fundamentals()
        sub = fund[fund["ticker"] == ticker.upper()]
        if annual_only and "freq" in fund.columns:
            sub = sub[sub["freq"].astype(str).str.upper().str.startswith("A")]
        if sub.empty:
            return pd.Series(dtype=float, name="value")
        # Within-`filed` tiebreak by period-end so a later fiscal period wins over
        # an earlier one filed the same day (10-K vs 10-K/A); period-end orders
        # ONLY, it is never the index (using it as the index would be look-ahead).
        end_col = next((c for c in ("datadate", "end", "period_end") if c in sub.columns), None)
        sub = sub.sort_values(["filed"] + ([end_col] if end_col else []), kind="stable")
        if field == "gross_profit":
            # per-ROW fallback: use `gp` where present, else revt-cogs for that row
            # (committing to the gp column for all rows would silently drop the
            # fully-computable firm-years where gp is NaN -- gp is sparse in funda).
            gp_col = sub["gp"] if "gp" in sub.columns else pd.Series(float("nan"), index=sub.index)
            computed = ((sub["revt"] - sub["cogs"]) if {"revt", "cogs"}.issubset(sub.columns)
                        else pd.Series(float("nan"), index=sub.index))
            vals = gp_col.where(gp_col.notna(), computed)
        else:
            mnem = COMPUSTAT_MNEMONIC[field]
            if mnem not in sub.columns:
                return pd.Series(dtype=float, name="value")
            vals = sub[mnem]
        s = pd.Series(vals.to_numpy(dtype=float), index=pd.to_datetime(sub["filed"]),
                      name="value").dropna()
        s = s[~s.index.duplicated(keep="last")].sort_index()  # latest period-end per filed
        s.index.name = "filed"
        return s

    # -- universe + prices (consumed by run_fundamentals._run_trial) ---------- #
    def universe(self) -> list[str]:
        """Survivorship-safe membership: every ticker that ever appears in the
        fundamentals OR price extracts (dead names included)."""
        names = set(self._fundamentals()["ticker"])
        try:
            names |= set(self._prices().columns)
        except FileNotFoundError:
            pass
        return sorted(names)

    def prices(self, universe: list[str], asof: pd.DatetimeIndex) -> pd.DataFrame:
        """Delisting-inclusive prices reindexed to ``asof`` (last price at or
        before each date). Dead names carry NaN after their final print."""
        px = self._prices()
        cols = [t for t in universe if t in px.columns]
        out = px[cols].reindex(asof, method="ffill")
        # Never carry a delisted name's last price PAST its final print: an
        # off-grid asof (e.g. month-end grid vs a daily price series) would
        # otherwise ffill a stale post-delist price -- a survivorship leak. Cap
        # each column at its last valid date.
        last = px[cols].apply(lambda s: s.last_valid_index())
        for c in cols:
            if pd.notna(last[c]):
                out.loc[out.index > last[c], c] = float("nan")
        return out

    @property
    def start(self) -> str:
        if self._start is not None:
            return self._start
        return str(self._prices().index.min().date())  # data-driven, not a hard floor

    @property
    def end(self) -> str:
        if self._end is not None:
            return self._end
        return str(self._prices().index.max().date())
