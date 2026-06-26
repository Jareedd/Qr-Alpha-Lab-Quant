"""Binance SPOT daily-kline data layer for the cash-and-carry feasibility audit.

This is the spot mirror of ``perp_data`` (futures/um). It points at the public
Binance SPOT dumps:

    https://data.binance.vision/data/spot/monthly/klines/{SYM}/1d/{SYM}-1d-{YYYY-MM}.zip

and caches per-symbol monthly files to parquet exactly like ``perp_data`` so the
audit is reproducible offline after one sequential download. NOTHING here is a
feature or a label -- it is data assembly for the long-spot leg of a delta-neutral
cash-and-carry book (long $1 spot, short $1 perp, collect funding).

Why a SEPARATE module rather than parameterising perp_data: CLAUDE.md forbids
modifying perp_data.py, and the spot/perp URL shapes differ (spot has no
``fundingRate`` field and no ``um`` segment). Keeping them apart means the audit
cannot accidentally mutate the registered trial's data layer.

PERP->SPOT symbol mapping (the subtle part). Binance lists some perps under a
SCALED ticker -- e.g. the perp ``1000PEPEUSDT`` tracks 1000x the spot ``PEPEUSDT``;
``1000SHIBUSDT`` -> ``SHIBUSDT``; ``1000000MOGUSDT`` -> ``MOGUSDT``. The spot pair,
when it exists, is listed UNSCALED. ``perp_to_spot_symbol`` strips a leading
numeric ``1000...``/``100...`` multiplier so we look the spot pair up correctly.
A perp with no corresponding spot pair (perp-only listing, or delisted/never-listed
spot) yields None from ``load_spot_klines`` and the caller SKIPS it. We never
fabricate a spot series -- a pair that did not exist is simply absent (PIT).

Kline columns are the standard 12-field Binance layout; OLD months are headerless
and NEW months carry a header, identical to perp_data, so we parse positionally
and drop a header row if present.
"""

from __future__ import annotations

import io
import os
import re
import socket
import time
import urllib.error
import urllib.request
import zipfile

import pandas as pd

# Belt-and-braces against hung sockets (mirrors perp_data): no socket op blocks
# past this even if the machine sleeps mid-request.
socket.setdefaulttimeout(120)

_UA = "Mozilla/5.0 (qr-alpha-lab research)"
DUMP = "https://data.binance.vision/data/spot/monthly"
LIST_API = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
CACHE = os.path.join("data_cache", "spot")

# Standard Binance kline columns (headerless old files -> positional fallback).
_KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume",
               "close_time", "quote_volume", "count", "taker_buy_volume",
               "taker_buy_quote_volume", "ignore"]


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    return urllib.request.urlopen(req, timeout=timeout).read()


def perp_to_spot_symbol(perp_symbol: str) -> str:
    """Map a USDT-perp ticker to the spot pair it tracks.

    Binance lists thin/cheap coins as a SCALED perp (e.g. ``1000PEPEUSDT`` =
    1000x ``PEPEUSDT``) while the spot pair is unscaled. Strip a leading numeric
    multiplier (``1000``, ``10000``, ``1000000``, ``100``...) so the spot lookup
    targets the real pair. For symbols with no multiplier this is the identity.

    Note: this only changes the SYMBOL we fetch. The price SCALE difference
    (perp quotes 1000x the spot) is irrelevant to the carry math because every
    leg uses daily RETURNS (ratios), and a constant scale factor cancels in a
    ratio. So we do not rescale prices -- we only need the right spot pair's
    return series.
    """
    m = re.fullmatch(r"(1000+|100)([A-Z][A-Z0-9]*USDT)", perp_symbol)
    if m:
        return m.group(2)
    return perp_symbol


def _available_months(symbol: str, cache_dir: str = CACHE) -> list[str]:
    """The YYYY-MM months Binance actually has SPOT 1d klines for this symbol,
    via ONE S3 directory listing (so a never-listed pair costs one request, not
    80+ probes). Cached to a small text file alongside the parquets so a resumed
    run does not re-list. Returns [] if the pair was never listed on spot."""
    os.makedirs(cache_dir, exist_ok=True)
    listing_cache = os.path.join(cache_dir, f"_months_{symbol}.txt")
    if os.path.exists(listing_cache):
        with open(listing_cache, encoding="utf-8") as f:
            return [m for m in f.read().split() if m]

    months: list[str] = []
    marker = ""
    while True:
        url = (f"{LIST_API}?prefix=data/spot/monthly/klines/{symbol}/1d/"
               + (f"&marker={marker}" if marker else ""))
        try:
            xml = _get(url, timeout=30).decode("utf-8", "replace")
        except urllib.error.URLError:
            break
        months += re.findall(r"-(\d{4}-\d{2})\.zip</Key>", xml)
        if "<IsTruncated>true</IsTruncated>" not in xml:
            break
        keys = re.findall(r"<Key>([^<]+)</Key>", xml)
        if not keys:
            break
        marker = keys[-1]
    months = sorted(set(months))
    with open(listing_cache, "w", encoding="utf-8") as f:
        f.write("\n".join(months))
    return months


def parse_kline_csv(text: str) -> pd.DataFrame:
    """Parse one monthly spot-kline CSV blob into a positional DataFrame.

    Pure (no I/O) so it is known-answer testable on a hand-built blob. Older
    months are headerless; newer ones carry a header row -- detect and drop it,
    then read positionally so columns are uniform integers across months
    (otherwise pandas.concat aligns string- and int-named columns separately and
    silently shreds any symbol spanning the header-format boundary -- the exact
    bug perp_data documents).
    """
    text = text.strip("\n")
    if not text:
        return pd.DataFrame()
    first = text.splitlines()[0].lower()
    skip = 1 if first.startswith(("open_time", "open time")) else 0
    df = pd.read_csv(io.StringIO(text), header=None, skiprows=skip)
    df.columns = [str(c) for c in df.columns]  # positional 0..n, as strings
    return df


def _load_raw(symbol, start, end, cache_dir) -> pd.DataFrame | None:
    """Download+cache one symbol's monthly spot files; return a concatenated
    positional frame, or None if the pair never traded in the window (every
    month absent -- a natural never-listed/delisted-spot signal)."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"klines_{symbol}_{start}_{end}.parquet")
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
        return df if len(df) else None

    lo, hi = start[:7], end[:7]
    want = [m for m in _available_months(symbol, cache_dir) if lo <= m <= hi]
    frames = []
    for ym in want:
        url = f"{DUMP}/klines/{symbol}/1d/{symbol}-1d-{ym}.zip"
        try:
            raw = _get(url, timeout=60)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue  # listed but file missing (rare)
            raise
        except urllib.error.URLError:
            time.sleep(1.0)
            continue
        z = zipfile.ZipFile(io.BytesIO(raw))
        frames.append(parse_kline_csv(z.read(z.namelist()[0]).decode()))
        time.sleep(0.05)  # polite to a free public bucket; sequential by design

    if not frames:
        pd.DataFrame().to_parquet(cache)  # cache the "never traded" answer
        return None
    out = pd.concat(frames, ignore_index=True)
    out.columns = [str(c) for c in out.columns]
    out.to_parquet(cache)
    return out


def load_spot_klines(symbol, start="2019-09-01", end=None, cache_dir=CACHE):
    """(date-indexed) spot close + quote_volume for one symbol, or None.

    ``symbol`` may be a perp ticker; it is mapped via ``perp_to_spot_symbol``
    before lookup, so callers can pass the perp universe directly. Returns None
    when the spot pair has no data in the window (caller SKIPS it). Index is
    UTC-normalised daily timestamps; duplicate days keep the last bar.
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-01")
    spot_symbol = perp_to_spot_symbol(symbol)
    df = _load_raw(spot_symbol, start, end, cache_dir)
    if df is None or df.empty:
        return None
    if df.columns[0] != "open_time":  # positional (headerless old months)
        df.columns = _KLINE_COLS[: df.shape[1]]
    # Binance switched kline timestamps from MILLISECONDS to MICROSECONDS in early
    # 2025, so a frame spanning that boundary MIXES units within one column. A single
    # unit="ms" then overflows the µs rows to year ~56971 (OutOfBoundsDatetime).
    # Normalize per-row to ms first: ms timestamps for 2019-2026 are ~1e12-1.8e12,
    # µs are ~1.5e15, so anything >= 1e14 is µs -> divide by 1000.
    ot = pd.to_numeric(df["open_time"], errors="coerce")
    ot = ot.where(ot < 1e14, ot / 1000.0)
    df["date"] = pd.to_datetime(ot, unit="ms").dt.normalize()
    out = df.groupby("date").agg(
        close=("close", "last"), quote_volume=("quote_volume", "sum")
    )
    out = out[~out.index.duplicated(keep="last")]
    # PIT trim: never return rows outside the requested window.
    out = out[(out.index >= pd.Timestamp(start)) & (out.index <= pd.Timestamp(end))]
    return out if len(out) else None
