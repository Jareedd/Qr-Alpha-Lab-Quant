"""Point-in-time S&P 500 membership, reconstructed from Wikipedia.

Why this module exists: backtesting on *today's* index members lets the long
side free-ride on hindsight -- every name is known to have survived (and
usually thrived). That is survivorship bias, and it silently inflates returns
(McLean-Pontiff-scale effects). The fix is to know who was in the index ON
each historical date and only trade those names at that time.

Method: Wikipedia's "List of S&P 500 companies" page maintains (a) the
current constituent table and (b) a changes table (effective date, ticker
added, ticker removed) going back decades. Starting from today's membership
and walking the changes BACKWARD (undo each addition, redo each removal)
yields the membership set over any past interval. Change effective dates are
announced in advance, so gating trades by effective date is point-in-time
safe.

Honest limitations (do not delete -- quantify):
- The changes table is community-maintained: dense and reliable for recent
  decades, sparser before ~2000. Keep backtest start >= 2005 (we use 2010).
- A point-in-time *membership mask* does not conjure up price data for dead
  companies: names removed via bankruptcy or acquisition often have no Yahoo
  history. ``coverage_report`` counts exactly how many member-names lack
  price data so the residual bias is a number in the write-up, not a secret.
- Ticker reuse across decades (same symbol, different company) is not
  resolved; it is rare within a post-2010 window.
"""

from __future__ import annotations

import io
import os
import urllib.request

import pandas as pd

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_UA = "qr-alpha-lab/0.1 (student research project)"


def _normalize_ticker(t: str) -> str:
    """Wikipedia uses BRK.B / BF.B; yfinance wants BRK-B / BF-B."""
    return str(t).strip().upper().replace(".", "-")


def fetch_sp500_tables(cache_dir: str = "data_cache") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (current_members, changes) from Wikipedia, cached locally.

    current_members: columns [ticker]; changes: columns [date, added, removed]
    with one row per ticker movement (a single announcement covering an add
    and a removal becomes one row with both filled).
    """
    os.makedirs(cache_dir, exist_ok=True)
    cur_path = os.path.join(cache_dir, "sp500_current.parquet")
    chg_path = os.path.join(cache_dir, "sp500_changes.parquet")
    if os.path.exists(cur_path) and os.path.exists(chg_path):
        return pd.read_parquet(cur_path), pd.read_parquet(chg_path)

    req = urllib.request.Request(WIKI_URL, headers={"User-Agent": _UA})
    html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
    tables = pd.read_html(io.StringIO(html))

    current = pd.DataFrame({"ticker": tables[0]["Symbol"].map(_normalize_ticker)})

    raw = tables[1].copy()
    raw.columns = ["date", "added", "added_name", "removed", "removed_name", "reason"]
    changes = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["date"], format="%B %d, %Y", errors="coerce"),
            "added": raw["added"].map(lambda t: _normalize_ticker(t) if pd.notna(t) else None),
            "removed": raw["removed"].map(lambda t: _normalize_ticker(t) if pd.notna(t) else None),
        }
    ).dropna(subset=["date"])

    current.to_parquet(cur_path)
    changes.to_parquet(chg_path)
    return current, changes


def build_membership_intervals(
    current: pd.DataFrame,
    changes: pd.DataFrame,
    start: str = "2010-01-01",
) -> list[tuple[pd.Timestamp, pd.Timestamp, frozenset]]:
    """Reconstruct membership backward from today.

    Returns a list of (interval_start, interval_end, members) covering
    [start, far-future], where each interval has constant membership and
    interval_start is inclusive, interval_end exclusive. Walking backward:
    before a change date, the added ticker was NOT a member and the removed
    ticker WAS.
    """
    start_ts = pd.Timestamp(start)
    members = set(current["ticker"])
    chg = changes.sort_values("date", ascending=False)

    intervals = []
    upper = pd.Timestamp("2262-01-01")  # effectively +inf for daily data
    for date, grp in chg.groupby("date", sort=False):
        if date <= start_ts:
            break
        intervals.append((date, upper, frozenset(members)))
        upper = date
        for added in grp["added"].dropna():
            members.discard(added)
        for removed in grp["removed"].dropna():
            members.add(removed)
    intervals.append((start_ts, upper, frozenset(members)))
    return intervals[::-1]  # chronological order


def membership_mask(
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    intervals: list[tuple[pd.Timestamp, pd.Timestamp, frozenset]],
) -> pd.DataFrame:
    """Boolean (date x ticker) frame: was this name in the index on this date?"""
    mask = pd.DataFrame(False, index=dates, columns=tickers)
    valid = set(tickers)
    for lo, hi, members in intervals:
        cols = sorted(members & valid)
        if cols:
            mask.loc[(dates >= lo) & (dates < hi), cols] = True
    return mask


def all_members_in_window(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp, frozenset]],
) -> list[str]:
    """Every ticker that was a member at any point in the window."""
    names: set[str] = set()
    for _, _, members in intervals:
        names |= members
    return sorted(names)


def coverage_report(member_tickers: list[str], prices: pd.DataFrame) -> dict:
    """Quantify the residual survivorship bias: members with no price data.

    Dead companies (bankruptcy, acquisition) often vanish from free data
    sources. We cannot trade what we cannot price, so those names drop out of
    the backtest -- this measures how big that hole is instead of hiding it.
    """
    have = [t for t in member_tickers if t in prices.columns and prices[t].notna().any()]
    missing = sorted(set(member_tickers) - set(have))
    return {
        "n_members_ever": len(member_tickers),
        "n_with_price_data": len(have),
        "n_missing_price_data": len(missing),
        "missing_tickers": missing,
        "pct_covered": round(len(have) / max(len(member_tickers), 1), 4),
    }
