"""INTEGRITY CROSS-CHECK: bulk Form 3/4/5 data sets vs the raw-XML Form 4 crawl.

The bulk quarterly ZIPs and the one-filing-at-a-time crawl are the SAME underlying
Form 4 filings. This script PROVES the ``BulkInsiderSource`` reproduces the crawl's
open-market BUYS for four names whose raw-XML crawl is already CACHED under
``data_cache/insider/`` (AAL, AON, BHF, CCL), so the crawl side needs little/no new
network. Only the bulk quarters spanning these four names' active buy windows are
downloaded (NOT all 64).

METHOD (per CIK):
  1. crawl buys  = InsiderSource().purchases(cik)          # reads the cache.
  2. bulk buys   = BulkInsiderSource(start, end).transactions([cik], kind="P").
  3. Restrict BOTH to the overlapping filing-date window = the quarters the bulk
     source covers ∩ the crawl's buy span.
  4. Compare the SETS of buys keyed on
       (filed_date, owner_name_normalized, transaction_date, shares).
     Report counts per side, symmetric-difference size, and example diffs.

NORMALIZATION (documented, benign-only): owner_name is upper-cased + internal
whitespace collapsed (the bulk feed and the XML occasionally differ in case /
double spaces for the SAME person). ``value`` is NOT in the key: the bulk feed
rounds TRANS_PRICEPERSHARE to whole cents while the raw XML keeps full precision
(e.g. 28.18 vs 28.1799), a benign price-formatting difference — so we key on
shares + dates + owner, never value.

PIT: both sides are filing-date-indexed; the comparison window is a filing-date
window. Output -> results/insider_bulk_crosscheck.json + a CROSSCHECK: line.
"""
from __future__ import annotations

import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab.insider_bulk import BulkInsiderSource
from quantlab.insider_data import InsiderSource

# The four names whose raw-XML crawl is already cached.
NAMES = {
    "0000006201": "AAL",
    "0000315293": "AON",
    "0001685040": "BHF",
    "0000815097": "CCL",
}


def _norm_owner(name) -> str:
    """Benign owner-name normalization: upper-case + collapse internal whitespace.

    The ONLY normalization applied to the comparison key. Documented as benign:
    the bulk TSV and the raw XML carry the same person's name with occasional
    case / double-space differences (e.g. 'Isom Robert D Jr' vs 'ISOM ROBERT D
    JR'); collapsing those is not a data difference."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    return re.sub(r"\s+", " ", str(name).strip()).upper()


def _key_set(df: pd.DataFrame) -> set[tuple]:
    """Set of buy keys (filed_date, owner_norm, transaction_date, shares) from a
    purchases frame. filed_date/transaction_date -> date (drop intraday); shares
    rounded to 4dp to dodge float noise."""
    keys = set()
    for filed, row in df.iterrows():
        fd = pd.Timestamp(filed).date() if pd.notna(filed) else None
        td = (pd.Timestamp(row["transaction_date"]).date()
              if pd.notna(row["transaction_date"]) else None)
        sh = round(float(row["shares"]), 4) if pd.notna(row["shares"]) else None
        keys.add((fd, _norm_owner(row["owner_name"]), td, sh))
    return keys


def _quarter(ts: pd.Timestamp) -> tuple[int, int]:
    return (ts.year, (ts.month - 1) // 3 + 1)


def main() -> int:
    crawl = InsiderSource()

    # 1) crawl buys per CIK (cache-only) + the union of buy quarters to download.
    crawl_buys: dict[str, pd.DataFrame] = {}
    union_quarters: set[tuple[int, int]] = set()
    spans: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for cik in NAMES:
        cb = crawl.purchases(cik)
        crawl_buys[cik] = cb
        if len(cb):
            spans[cik] = (cb.index.min(), cb.index.max())
            for d in cb.index:
                union_quarters.add(_quarter(pd.Timestamp(d)))

    # Overall download window = earliest..latest crawl-buy quarter across the 4.
    # (Downloads are restricted to exactly the buy-quarters below, NOT all 64.)
    all_filed = pd.concat([cb for cb in crawl_buys.values() if len(cb)])
    start = all_filed.index.min().normalize()
    end = all_filed.index.max().normalize()

    bulk = BulkInsiderSource(start=str(start.date()), end=str(end.date()))
    # Pre-download ONLY the quarters that contain a crawl buy (the union). The
    # crawl cache is COMPLETE for these CIKs (verified: every accession has cached
    # XML), so a bulk-only buy could only appear in one of these quarters or be a
    # genuine extra accession; downloading the union catches every crawl buy and
    # any bulk extra in those quarters.
    quarters = sorted(union_quarters)
    print(f"Downloading {len(quarters)} bulk quarters (union of buy windows): "
          f"{quarters}")
    for (y, q) in quarters:
        bulk._quarter_zip(y, q)   # sequential, rate-limited, cached.

    # 2) per-CIK comparison over the overlap window.
    report: dict = {"names": {}, "normalization": {
        "owner_name": "upper-case + collapse internal whitespace (benign)",
        "key": "(filed_date, owner_name_norm, transaction_date, shares)",
        "value_excluded": "bulk rounds price to cents; XML keeps full precision",
    }, "download_quarters": [f"{y}q{q}" for (y, q) in quarters]}

    overall_match = True
    total_only_crawl = 0
    total_only_bulk = 0

    for cik, tkr in NAMES.items():
        cb = crawl_buys[cik]
        if not len(cb):
            report["names"][cik] = {"ticker": tkr, "note": "no crawl buys"}
            continue
        # bulk buys for this CIK across the (already downloaded) window.
        bb = bulk.transactions([cik], kind="P")

        # overlap window = crawl buy span ∩ bulk window (the downloaded quarters).
        lo, hi = spans[cik]
        cb_w = cb[(cb.index >= lo) & (cb.index <= hi)]
        bb_w = bb[(bb.index >= lo) & (bb.index <= hi)]

        ks_crawl = _key_set(cb_w)
        ks_bulk = _key_set(bb_w)
        only_crawl = ks_crawl - ks_bulk
        only_bulk = ks_bulk - ks_crawl
        symdiff = len(only_crawl) + len(only_bulk)

        match = symdiff == 0
        overall_match = overall_match and match
        total_only_crawl += len(only_crawl)
        total_only_bulk += len(only_bulk)

        def _fmt(keys):
            out = []
            for (fd, ow, td, sh) in sorted(
                    keys, key=lambda k: (str(k[0]), k[1])):
                out.append({"filed_date": str(fd), "owner": ow,
                            "transaction_date": str(td), "shares": sh})
            return out

        report["names"][cik] = {
            "ticker": tkr,
            "window": [str(lo.date()), str(hi.date())],
            "crawl_buys": int(len(cb_w)),
            "bulk_buys": int(len(bb_w)),
            "crawl_keys": len(ks_crawl),
            "bulk_keys": len(ks_bulk),
            "symmetric_difference": symdiff,
            "only_in_crawl_examples": _fmt(only_crawl)[:5],
            "only_in_bulk_examples": _fmt(only_bulk)[:5],
            "match": match,
        }
        print(f"  {tkr} ({cik}): crawl={len(cb_w)} bulk={len(bb_w)} "
              f"keys crawl={len(ks_crawl)} bulk={len(ks_bulk)} "
              f"symdiff={symdiff} {'MATCH' if match else 'MISMATCH'}")

    report["overall_match"] = overall_match
    report["total_only_in_crawl"] = total_only_crawl
    report["total_only_in_bulk"] = total_only_bulk

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "insider_bulk_crosscheck.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nWrote {out_path}")
    if overall_match:
        print("CROSSCHECK: MATCH")
        return 0
    print(f"CROSSCHECK: MISMATCH (only_in_crawl={total_only_crawl}, "
          f"only_in_bulk={total_only_bulk})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
