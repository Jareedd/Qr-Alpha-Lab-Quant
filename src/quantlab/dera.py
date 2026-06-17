"""SEC DERA Financial Statement Data Sets — a free, survivorship-safe,
filing-date-PIT fundamentals source keyed on CIK.

WHY THIS MATTERS FOR H1. The H1 blocker (audit 2026-06-14, CIK measurement
2026-06-15) is that SEC's company_tickers.json is current-only, so dead names are
unmapped — and `companyfacts`/`companyconcept` are convenient but the practical
H1 question is bulk coverage. The DERA quarterly datasets contain EVERY filer's
numeric facts (dead companies included, because filings persist under their CIK)
keyed on CIK + filing date — exactly the survivorship-safe, point-in-time shape
H1 needs.

WHAT DERA DOES AND DOES NOT SOLVE — read before believing it unblocks H1.
  * SOLVES: fundamentals coverage. Given a CIK, DERA num.txt has Assets, NetIncome,
    CFO, Revenue, CoGS etc. for all filers back to 2009, filing-date stamped.
  * DOES NOT SOLVE: the dead-ticker -> CIK crosswalk. DERA num.txt is NUMERIC
    only; dei:TradingSymbol is a TEXT fact and is absent, and DERA sub.txt carries
    the company NAME but no ticker. So mapping a dead S&P TICKER to its CIK still
    needs a name/ticker bridge — the same bottleneck measured at ~73%->75% on free
    sources (cik_history.py). DERA improves the fundamentals side; it does not, by
    itself, close survivorship. That distinction is the honest finding.

Pure parsers (tab-delimited, header-driven) so they are testable on fixtures
without downloading the multi-hundred-MB quarterly archives.
"""
from __future__ import annotations

import io

import pandas as pd

SUB_COLS = ["adsh", "cik", "name", "form", "period", "fy", "fp", "filed"]
NUM_COLS = ["adsh", "tag", "version", "ddate", "qtrs", "uom", "value"]


def _read_tsv(buf, usecols=None) -> pd.DataFrame:
    """DERA tables are tab-separated with a header row. ``buf`` is a path or a
    file-like / string. Only ``usecols`` are kept (memory-friendly at scale)."""
    src = io.StringIO(buf) if isinstance(buf, str) and "\t" in buf else buf
    return pd.read_csv(src, sep="\t", usecols=usecols, dtype=str,
                       encoding="utf-8", on_bad_lines="skip")


def parse_sub(buf) -> pd.DataFrame:
    """sub.txt -> submissions keyed by accession (adsh): cik, name, form, filed."""
    df = _read_tsv(buf, usecols=lambda c: c in set(SUB_COLS))
    df["cik"] = pd.to_numeric(df["cik"], errors="coerce").astype("Int64")
    df["filed"] = pd.to_datetime(df["filed"], format="%Y%m%d", errors="coerce")
    return df


def parse_num(buf) -> pd.DataFrame:
    """num.txt -> numeric facts: adsh, tag, ddate (period end), qtrs, value."""
    df = _read_tsv(buf, usecols=lambda c: c in set(NUM_COLS))
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["ddate"] = pd.to_datetime(df["ddate"], format="%Y%m%d", errors="coerce")
    return df


def pit_value(sub: pd.DataFrame, num: pd.DataFrame, cik: int, tag: str,
              asof, forms: tuple[str, ...] = ("10-K", "10-Q")) -> float | None:
    """Filing-date point-in-time value of ``tag`` for ``cik`` as known at ``asof``.

    Selects the filer's periodic-report filings with ``filed <= asof``, joins the
    requested ``tag`` from num.txt, and returns the value from the LATEST such
    filing (freshest figure the market could know at asof). None if unavailable.
    This is the survivorship-safe, PIT primitive H1's harness would consume —
    identical contract to FreeSECSource.field_series but sourced from bulk DERA.
    """
    asof = pd.Timestamp(asof)
    filings = sub[(sub["cik"] == cik) & (sub["form"].isin(forms))
                  & (sub["filed"] <= asof)]
    if filings.empty:
        return None
    facts = num[num["tag"] == tag].merge(
        filings[["adsh", "filed"]], on="adsh", how="inner")
    if facts.empty:
        return None
    facts = facts.sort_values(["filed", "ddate"])
    return float(facts.iloc[-1]["value"])


def filer_ciks(sub: pd.DataFrame) -> set[int]:
    """All distinct filer CIKs present (the survivorship-safe filer universe)."""
    return set(int(c) for c in sub["cik"].dropna().unique())
