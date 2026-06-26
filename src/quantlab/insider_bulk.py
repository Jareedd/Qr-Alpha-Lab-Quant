"""SEC bulk quarterly insider-transaction data sets (Form 3/4/5) — a faithful,
drop-in alternative to the one-filing-at-a-time crawl in ``quantlab.insider_data``.

WHY THIS EXISTS. ``InsiderSource`` enumerates a company's Form 4 accessions from
the submissions API and fetches/parses each accession's RAW ``ownershipDocument``
XML — correct, but it costs one HTTP request per accession (≈200k requests to
cover a broad universe). SEC publishes the SAME structured Form 3/4/5 data as
quarterly bulk ZIPs (≈64 files back to 2006), so the entire corpus is ≈64
downloads instead. This module reads those ZIPs and surfaces open-market
PURCHASES (and SALES) with EXACTLY the schema ``InsiderSource.purchases()``
returns, so the H10 signal layer can swap data sources without any other change.

FAITHFULNESS (the whole point). The bulk data and the raw-XML crawl are the same
underlying Form 4 filings, so they must agree. Two deliberate choices keep them
in lock-step:
  * DOCUMENT_TYPE is filtered to {``4``, ``4/A``} — the crawl's
    ``parse_form4_accessions`` keeps ``form == "4"`` (the submissions API reports
    a 4/A amendment under its own ``form`` string, but the cached crawl corpus
    used for the cross-check is dominated by plain ``4``; see the cross-check
    script for the exact reconciliation and any residual amendment diffs).
  * ONE owner per accession by default = the FIRST ``REPORTINGOWNER`` row, because
    ``parse_form4_xml`` collapses a multi-owner filing to its first
    ``reportingOwner``. ``all_owners=True`` keeps every owner row instead (useful
    for joint-filing analysis, but it will NOT match the crawl — documented).

PIT SAFETY. The filing date (SUBMISSION.FILING_DATE) is the point-in-time instant
the market could first know — Form 4 is due within two business days of the
transaction. EVERY surfaced frame is indexed by FILING DATE, never the (earlier)
transaction date. The transaction date is carried as a column for diagnostics
only; it must never drive the index or any as-of join.

NO NETWORK AT IMPORT. Downloads happen lazily inside ``transactions``/the
``_quarter_zip`` downloader; parsing is split into PURE functions
(``parse_*_tsv``, ``build_transactions``) that take raw bytes so tests never hit
the network. Downloads are SEQUENTIAL and paced by the shared
``insider_data._get`` rate limiter (SEC fair-access < 10 req/s); ZIPs are cached
under ``data_cache/insider_bulk/`` so a re-run is free.
"""
from __future__ import annotations

import io
import os
import zipfile

import pandas as pd

# Reuse the EXACT rate limiter / retry-backoff policy and role logic from the
# crawl layer — do not duplicate the limiter (one IP, one SEC budget).
from quantlab.insider_data import _get, _role_tag

CACHE = os.path.join("data_cache", "insider_bulk")

# Verified bulk data-set URL pattern. ~8-14 MB/ZIP, data back to 2006, range
# requests supported. {y} is the 4-digit year, {q} is 1..4.
_ZIP_URL = ("https://www.sec.gov/files/structureddata/data/"
            "insider-transactions-data-sets/{y}q{q}_form345.zip")

# The output schema is pinned EQUAL to InsiderSource.purchases().
_OUT_COLS = ["owner_name", "role", "shares", "value", "transaction_date",
             "ticker", "accession"]

# Open-market legs (same definition as the crawl's P/A buy, S/D sale).
_KIND_FILTER = {
    "P": ("P", "A"),   # open-market purchase: TRANS_CODE==P AND ACQ_DISP==A
    "S": ("S", "D"),   # open-market sale:     TRANS_CODE==S AND ACQ_DISP==D
}


def _norm_cik(cik) -> str:
    """A CIK as the 10-digit zero-padded string SEC uses everywhere (the bulk
    SUBMISSION.ISSUERCIK is already zero-padded; normalise raw ints/strs to it)."""
    return str(int(cik)).zfill(10)


# --------------------------------------------------------------------------- #
# PURE parsers (no network) — known-answer-testable on tiny TSV byte blobs.
# --------------------------------------------------------------------------- #

def _read_tsv(blob: bytes) -> pd.DataFrame:
    """Read a bulk TSV byte blob as all-string columns (numerics coerced later).

    Large real tables: ``dtype=str`` avoids pandas guessing mixed types per
    column, and ``keep_default_na=False`` keeps an empty field as ``""`` (not
    NaN) so missing-vs-present is explicit. Returns an empty frame for an empty
    blob. PURE: no network, no I/O beyond the in-memory buffer."""
    if not blob or not blob.strip():
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(blob), sep="\t", dtype=str,
                       keep_default_na=False, na_values=[])


def _parse_bulk_date(s: pd.Series) -> pd.Series:
    """Parse the bulk ``DD-MMM-YYYY`` dates (e.g. ``31-JAN-2024``) to Timestamps.

    The format is unambiguous and fixed in the data sets; uppercase month
    abbreviations parse fine with ``%d-%b-%Y``. Blank/garbage -> NaT (coerce)."""
    return pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")


def parse_submission_tsv(blob: bytes,
                         include_amendments: bool = False) -> pd.DataFrame:
    """SUBMISSION.tsv -> one row per Form 4 (optionally 4/A) accession with the
    issuer identity and parsed filing date.

    Columns out: ``accession, filed, issuer_cik, ticker``.

    AMENDMENT POLICY (faithfulness). By DEFAULT keeps only ``DOCUMENT_TYPE == "4"``
    — exactly matching the crawl, whose ``parse_form4_accessions`` keeps
    ``form == "4"`` and so EXCLUDES 4/A amendments. With
    ``include_amendments=True`` it ALSO keeps ``"4/A"`` (the
    spec's {``4``,``4/A``} set). 3 / 3/A / 5 / 5/A are always dropped (the crawl is
    Form-4 only). This single flag is the ONE place the bulk source can legitimately
    diverge from the crawl: an open-market buy reported only on a 4/A amendment
    (verified live: AON director Knight's accession 0001179110-16-019798) appears
    in the bulk source with amendments on, but never in the crawl. Default off =
    drop-in faithful; turn on to capture amendment-only corrections.

    ``issuer_cik`` is normalised to the 10-digit zero-padded string. ``filed`` is
    the FILING date (the PIT instant). PURE."""
    df = _read_tsv(blob)
    cols = ["accession", "filed", "issuer_cik", "ticker"]
    if df.empty or "ACCESSION_NUMBER" not in df.columns:
        return pd.DataFrame(columns=cols)
    allowed = ["4", "4/A"] if include_amendments else ["4"]
    keep = df["DOCUMENT_TYPE"].isin(allowed)
    df = df[keep]
    out = pd.DataFrame({
        "accession": df["ACCESSION_NUMBER"].astype(str).str.strip(),
        "filed": _parse_bulk_date(df["FILING_DATE"]),
        "issuer_cik": df["ISSUERCIK"].astype(str).str.strip().apply(
            lambda c: _norm_cik(c) if c not in ("", "nan") else c),
        "ticker": df["ISSUERTRADINGSYMBOL"].astype(str).str.strip().replace(
            {"": None, "nan": None, "NONE": None}),
    })
    return out.reset_index(drop=True)


def parse_reportingowner_tsv(blob: bytes) -> pd.DataFrame:
    """REPORTINGOWNER.tsv -> owner identity + role flags per (accession, owner).

    Columns out: ``accession, owner_seq, owner_name, is_officer, is_director,
    is_tenpct``. ``owner_seq`` is the 0-based position of the owner row WITHIN its
    accession (preserving file order) so a caller can pick the FIRST owner to
    match the crawl. Role flags are derived from the free-text
    ``RPTOWNER_RELATIONSHIP`` (e.g. ``Director,Officer``) by case-insensitive
    substring match on Officer/Director/TenPercentOwner — equivalent to the XML
    boolean flags the crawl reads. PURE."""
    df = _read_tsv(blob)
    cols = ["accession", "owner_seq", "owner_name",
            "is_officer", "is_director", "is_tenpct"]
    if df.empty or "ACCESSION_NUMBER" not in df.columns:
        return pd.DataFrame(columns=cols)
    acc = df["ACCESSION_NUMBER"].astype(str).str.strip()
    rel = df["RPTOWNER_RELATIONSHIP"].fillna("").astype(str).str.lower()
    out = pd.DataFrame({
        "accession": acc,
        # position within the accession block = file order (stable cumcount).
        "owner_seq": df.groupby(acc, sort=False).cumcount(),
        "owner_name": df["RPTOWNERNAME"].astype(str).str.strip(),
        "is_officer": rel.str.contains("officer", regex=False),
        "is_director": rel.str.contains("director", regex=False),
        "is_tenpct": rel.str.contains("tenpercentowner", regex=False),
    })
    return out.reset_index(drop=True)


def parse_nonderiv_trans_tsv(blob: bytes) -> pd.DataFrame:
    """NONDERIV_TRANS.tsv -> one row per non-derivative transaction with numerics
    coerced.

    Columns out: ``accession, code, acq_disp, shares, price, transaction_date``.
    ``shares``/``price`` are floats (NaN if blank/garbage — a missing price is
    legitimate on some rows). ``transaction_date`` is a Timestamp (the bulk
    ``DD-MMM-YYYY``). Derivative tables are intentionally ignored — H10 keys on
    open-market common-stock (non-derivative) transactions. PURE."""
    df = _read_tsv(blob)
    cols = ["accession", "code", "acq_disp", "shares", "price",
            "transaction_date"]
    if df.empty or "ACCESSION_NUMBER" not in df.columns:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame({
        "accession": df["ACCESSION_NUMBER"].astype(str).str.strip(),
        "code": df["TRANS_CODE"].astype(str).str.strip(),
        "acq_disp": df["TRANS_ACQUIRED_DISP_CD"].astype(str).str.strip(),
        "shares": pd.to_numeric(df["TRANS_SHARES"], errors="coerce"),
        "price": pd.to_numeric(df["TRANS_PRICEPERSHARE"], errors="coerce"),
        "transaction_date": _parse_bulk_date(df["TRANS_DATE"]),
    })
    return out.reset_index(drop=True)


def build_transactions(sub_blob: bytes, owner_blob: bytes, trans_blob: bytes,
                       ciks: set[str], kind: str = "P",
                       all_owners: bool = False,
                       include_amendments: bool = False,
                       with_cik: bool = False) -> pd.DataFrame:
    """Join the three parsed bulk tables into a FILING-DATE-indexed open-market
    transactions frame, for the given issuer CIK set, of one kind.

    This is the PURE core (no network): it takes the three TSV byte blobs, parses
    them, restricts to ``ciks`` (10-digit zero-padded strings), keeps the open-
    market leg for ``kind`` (``P`` = P/A buys, ``S`` = S/D sales), and joins owner
    identity onto every transaction row of an accession.

    Owner collapsing (faithfulness): by default ONLY the FIRST reporting owner of
    each accession is kept (``owner_seq == 0``) — this matches the crawl, whose
    ``parse_form4_xml`` uses the first ``reportingOwner``. With
    ``all_owners=True`` every owner row is kept (so a multi-owner accession yields
    one transaction row per owner) — this will NOT match the crawl and is for
    joint-filing analysis only.

    Amendments (faithfulness): ``include_amendments`` (default False) is forwarded
    to ``parse_submission_tsv`` — off = Form-4-only (matches the crawl); on also
    keeps 4/A amendments (the only documented source of a legitimate crawl/bulk
    buy difference).

    Output columns are EXACTLY ``InsiderSource.purchases()``: ``owner_name, role,
    shares, value, transaction_date, ticker, accession``; index name
    ``filed_date``. ``value = shares * price`` (NaN if price missing). ``role``
    via the shared ``_role_tag`` (officer > director > tenpct > insider).

    ``with_cik`` (default False, opt-in): when True, APPEND a 10-digit zero-padded
    ``issuer_cik`` column (the SUBMISSION.ISSUERCIK), so a caller can re-key the
    cross-section on a CIK-resolved universe symbol rather than the issuer's (then-
    current) trading symbol. Default False keeps the schema EXACTLY
    ``InsiderSource.purchases()`` (so the crawl cross-check + existing tests stay
    green). It changes NO other behavior — the same rows, the same order, only an
    extra column.

    PIT: indexed by ``filed_date`` only — a transaction enters at its filing
    instant, never its earlier transaction date."""
    if kind not in _KIND_FILTER:
        raise ValueError(f"kind must be one of {sorted(_KIND_FILTER)}, got {kind!r}")
    code, acq = _KIND_FILTER[kind]

    sub = parse_submission_tsv(sub_blob, include_amendments=include_amendments)
    owners = parse_reportingowner_tsv(owner_blob)
    trans = parse_nonderiv_trans_tsv(trans_blob)

    out_cols = _OUT_COLS + ["issuer_cik"] if with_cik else _OUT_COLS
    empty = pd.DataFrame(columns=out_cols,
                         index=pd.DatetimeIndex([], name="filed_date"))
    if sub.empty or trans.empty:
        return empty

    ciks = {_norm_cik(c) for c in ciks}
    sub = sub[sub["issuer_cik"].isin(ciks)]
    if sub.empty:
        return empty

    # open-market leg only.
    trans = trans[(trans["code"] == code) & (trans["acq_disp"] == acq)]
    if trans.empty:
        return empty

    # owner rows: first-owner-by-default to match the crawl.
    if owners.empty:
        owners = pd.DataFrame(columns=["accession", "owner_seq", "owner_name",
                                       "is_officer", "is_director", "is_tenpct"])
    if not all_owners:
        owners = owners[owners["owner_seq"] == 0]

    # Restrict every table to the accessions of interest before joining (cheap).
    accs = set(sub["accession"])
    trans = trans[trans["accession"].isin(accs)]
    owners = owners[owners["accession"].isin(accs)]
    if trans.empty:
        return empty

    # transactions x owners (many-to-many only when all_owners and >1 owner);
    # first-owner default makes this one owner per accession.
    merged = trans.merge(owners, on="accession", how="left")
    # attach filing date + issuer identity (ticker, and cik when requested) from
    # submissions.
    sub_cols = ["accession", "filed", "ticker"]
    if with_cik:
        sub_cols.append("issuer_cik")
    merged = merged.merge(sub[sub_cols], on="accession", how="left")

    # role tag via the SHARED crawl logic (officer>director>tenpct>insider).
    def _role(row) -> str:
        return _role_tag({
            "is_officer": bool(row.get("is_officer")),
            "is_director": bool(row.get("is_director")),
            "is_tenpct": bool(row.get("is_tenpct")),
        })

    out = pd.DataFrame({
        "filed_date": merged["filed"],
        "owner_name": merged["owner_name"],
        "role": merged.apply(_role, axis=1) if len(merged) else [],
        "shares": merged["shares"],
        "value": merged["shares"] * merged["price"],   # NaN if price missing
        "transaction_date": merged["transaction_date"],
        "ticker": merged["ticker"],
        "accession": merged["accession"],
    })
    if with_cik:
        out["issuer_cik"] = merged["issuer_cik"].values
    out = out.dropna(subset=["filed_date"]).set_index("filed_date").sort_index()
    out.index.name = "filed_date"
    return out[out_cols]


# --------------------------------------------------------------------------- #
# Live source (network: sequential, rate-limited via the shared _get; cached).
# --------------------------------------------------------------------------- #

class BulkInsiderSource:
    """Read SEC bulk quarterly Form 3/4/5 data sets and surface open-market Form 4
    PURCHASES/SALES with the SAME schema as ``InsiderSource`` — a drop-in, far
    cheaper data source (≈64 ZIPs vs ≈200k requests for the same Form 4 corpus).

    Downloads are SEQUENTIAL and paced by the shared ``insider_data._get`` limiter
    (SEC fair-access < 10 req/s, one IP); each quarter ZIP is cached under
    ``cache_dir``. NO network at import time — only when a method needs a quarter.
    """

    def __init__(self, cache_dir: str = CACHE, start: str = "2010-01-01",
                 end: str | None = None):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()

    # -- which quarters span [start, end] -- #
    def _quarters(self) -> list[tuple[int, int]]:
        """List of (year, quarter) tuples whose calendar quarter overlaps
        [start, end]. PIT-neutral helper (pure date arithmetic)."""
        qs: list[tuple[int, int]] = []
        y, q = self.start.year, (self.start.month - 1) // 3 + 1
        ey, eq = self.end.year, (self.end.month - 1) // 3 + 1
        while (y, q) <= (ey, eq):
            qs.append((y, q))
            q += 1
            if q > 4:
                q, y = 1, y + 1
        return qs

    # -- one quarter ZIP (cached, paced via shared _get) -- #
    def _quarter_zip(self, year: int, q: int) -> bytes | None:
        """Download (or read from cache) one quarter's ``{y}q{q}_form345.zip``.

        Sequential + rate-limited via the shared ``_get`` (never parallelised).
        Returns the raw ZIP bytes, or None if the quarter is not published yet
        (404 — e.g. a quarter past the latest release). Cached as-is on disk so a
        re-run is free. NO date logic here — just I/O."""
        path = os.path.join(self.cache_dir, f"{year}q{q}_form345.zip")
        if os.path.exists(path):
            return open(path, "rb").read()
        url = _ZIP_URL.format(y=year, q=q)
        try:
            raw = _get(url, timeout=120)
        except Exception as e:  # 404 (not-yet-published) or hard network failure
            import urllib.error
            if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                return None
            raise
        open(path, "wb").write(raw)
        return raw

    @staticmethod
    def _split_zip(raw: bytes) -> tuple[bytes, bytes, bytes]:
        """Extract the three needed TSV byte blobs (SUBMISSION, REPORTINGOWNER,
        NONDERIV_TRANS) from a quarter ZIP. Missing member -> empty blob."""
        z = zipfile.ZipFile(io.BytesIO(raw))
        names = set(z.namelist())

        def _maybe(name: str) -> bytes:
            return z.read(name) if name in names else b""

        return (_maybe("SUBMISSION.tsv"),
                _maybe("REPORTINGOWNER.tsv"),
                _maybe("NONDERIV_TRANS.tsv"))

    def transactions(self, ciks, kind: str = "P",
                     all_owners: bool = False,
                     include_amendments: bool = False,
                     with_cik: bool = False) -> pd.DataFrame:
        """Open-market Form 4 transactions for the given issuer CIK set, across the
        quarters spanning [start, end], as a FILING-DATE-indexed DataFrame.

        ``ciks``: an iterable of CIKs (raw int/str ok; normalised to 10-digit
        zero-padded). ``kind``: ``"P"`` = open-market buys (TRANS_CODE P, ACQ A),
        ``"S"`` = open-market sells (TRANS_CODE S, DISP D) — non-open-market codes
        (awards ``A``, option exercises ``M``, gifts ``G``, tax ``F`` …) are
        excluded from both. ``all_owners``: default False keeps only the FIRST
        reporting owner per accession (matches the crawl); True keeps every owner
        row (does NOT match the crawl — joint-filing analysis only).
        ``include_amendments``: default False = Form-4-only (matches the crawl);
        True also keeps 4/A amendments (the one documented place the bulk source
        can legitimately carry a buy the crawl does not).
        ``with_cik``: default False keeps the schema EXACTLY
        ``InsiderSource.purchases()``; True appends a 10-digit ``issuer_cik``
        column so the caller can re-key the cross-section on a CIK-resolved
        universe symbol (the H12 price-alignment need).

        Output columns EXACTLY ``InsiderSource.purchases()``: ``owner_name, role,
        shares, value, transaction_date, ticker, accession``; index name
        ``filed_date``; ``value = shares * price`` (NaN if price missing). With
        ``with_cik=True`` an extra ``issuer_cik`` column is appended.

        Downloads each spanning quarter sequentially (rate-limited, cached). A
        quarter that 404s (not yet published) is skipped. Result is filtered to
        the [start, end] filing-date window after assembly.

        PIT: indexed by ``filed_date`` only — every row enters at its filing
        instant, never its earlier transaction date (Form 4 is due within two
        business days, so the filing date is the near-PIT instant the market could
        first act on)."""
        cik_set = {_norm_cik(c) for c in ciks}
        out_cols = _OUT_COLS + ["issuer_cik"] if with_cik else _OUT_COLS
        frames = []
        for (year, q) in self._quarters():
            raw = self._quarter_zip(year, q)
            if raw is None:
                continue
            sub_b, own_b, tr_b = self._split_zip(raw)
            frame = build_transactions(sub_b, own_b, tr_b, cik_set,
                                       kind=kind, all_owners=all_owners,
                                       include_amendments=include_amendments,
                                       with_cik=with_cik)
            if len(frame):
                frames.append(frame)
        if not frames:
            return pd.DataFrame(
                columns=out_cols,
                index=pd.DatetimeIndex([], name="filed_date"))
        out = pd.concat(frames).sort_index()
        # clamp to the requested filing-date window (a quarter ZIP can carry a
        # filing dated a day or two into the next calendar quarter).
        out = out[(out.index >= self.start) & (out.index <= self.end)]
        out.index.name = "filed_date"
        return out

    # -- single-CIK convenience mirroring InsiderSource -- #
    def purchases(self, cik: str, all_owners: bool = False,
                  include_amendments: bool = False) -> pd.DataFrame:
        """Open-market PURCHASES (P/A) for ONE issuer CIK, filing-date-indexed.

        Mirrors ``InsiderSource.purchases`` but takes a RAW CIK (ticker->CIK
        resolution is the caller's job, exactly as InsiderSource behaves when given
        a digit string). Same columns/shape as the crawl. Defaults
        (``all_owners=False``, ``include_amendments=False``) match the crawl
        exactly. PIT: filing-date index."""
        return self.transactions([cik], kind="P", all_owners=all_owners,
                                 include_amendments=include_amendments)

    def sells(self, cik: str, all_owners: bool = False,
              include_amendments: bool = False) -> pd.DataFrame:
        """Open-market SELLS (S/D) for ONE issuer CIK, filing-date-indexed.

        Symmetric to ``purchases`` (same columns/shape); only the open-market leg
        differs. Takes a RAW CIK. Defaults match the crawl. PIT: filing-date index
        — a sale enters at its filing instant, never its earlier transaction
        date."""
        return self.transactions([cik], kind="S", all_owners=all_owners,
                                 include_amendments=include_amendments)
