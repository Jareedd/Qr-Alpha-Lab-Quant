"""Binance USDT-perp data layer for H2 (trial #8) — public dumps only.

Source: data.binance.vision (USDT-margined futures, prefix 'futures/um').
Verified reachable 2026-06-13: 876+ symbols listed INCLUDING delisted
contracts (LUNAUSDT present), so the point-in-time universe with dead
names is buildable for free — the central claim of the H2 registration.

Three things this module provides, all cached to parquet so the trial is
reproducible offline after one download:

- ``list_usdt_perp_symbols()``: every USDT-margined perp symbol ever
  listed (delisted ones included — that is the whole point; omitting them
  is the survivorship bias trial #2 taught this project to refuse).
- ``load_klines(symbol)`` / ``load_funding(symbol)``: per-symbol daily
  bars (close, quote_volume) and 8h funding settlements.
- ``build_panels(symbols, ...)``: (date x symbol) mark-price, dollar-
  volume, and DAILY-funding panels. A contract is NaN outside its trading
  life, so listing/delisting fall out of the data, not out of a hand list.

Funding sign convention (Binance): positive funding = longs pay shorts.
Daily funding = the SUM of that UTC day's settlements (3 at 8h), i.e. the
total a long pays that day. A long's funding-inclusive daily total return
is therefore ``mark_return - daily_funding`` — the label H2 requires, and
the object the synthetic carry world encodes.

Nothing here is a feature or a label; it is data assembly. PIT-safety
arguments for the features/labels live in the carry harness.
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

# Belt-and-braces against hung sockets (the machine sleeping mid-request
# froze a download run for ~14h); no socket op blocks past this.
socket.setdefaulttimeout(120)

_UA = "Mozilla/5.0 (qr-alpha-lab research)"
DUMP = "https://data.binance.vision/data/futures/um/monthly"
LIST_API = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
CACHE = os.path.join("data_cache", "perp")


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    return urllib.request.urlopen(req, timeout=timeout).read()


def list_usdt_perp_symbols(cache_dir: str = CACHE) -> list[str]:
    """All USDT-margined perp symbols ever listed (paginated S3 listing).

    Cached to a text file; delete it to refresh. USDT-quoted only (the
    registration's universe); USDC/coin-margined contracts excluded.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "symbols_usdt.txt")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return [s for s in f.read().split() if s]

    symbols: list[str] = []
    marker = ""
    while True:
        url = (f"{LIST_API}?delimiter=/&prefix=data/futures/um/monthly/klines/"
               + (f"&marker={marker}" if marker else ""))
        xml = _get(url).decode("utf-8", "replace")
        symbols += re.findall(
            r"<Prefix>data/futures/um/monthly/klines/([^/]+)/</Prefix>", xml
        )
        truncated = "<IsTruncated>true</IsTruncated>" in xml
        if not truncated:
            break
        # NextMarker, or fall back to the last key seen.
        m = re.search(r"<NextMarker>([^<]+)</NextMarker>", xml)
        marker = (m.group(1) if m else
                  f"data/futures/um/monthly/klines/{symbols[-1]}/")
        time.sleep(0.2)

    # USDT-quoted, and a sanity filter: real Binance symbols are
    # uppercase-alnum (e.g. 1000PEPEUSDT) — drops any XML/listing artifact.
    usdt = sorted({s for s in symbols
                   if s.endswith("USDT") and re.fullmatch(r"[A-Z0-9]+", s)})
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(usdt))
    return usdt


def _months(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m")
            for d in pd.date_range(start, end, freq="MS")]


def _available_months(symbol: str, kind: str) -> list[str]:
    """The YYYY-MM months Binance actually has for this symbol+field, via
    ONE directory listing instead of probing every calendar month (which
    turned a delisted contract's handful of files into 80+ wasted 404s).
    Naturally bounds each symbol to its true listing..delisting life."""
    sub = f"klines/{symbol}/1d" if kind == "klines" else f"fundingRate/{symbol}"
    months: list[str] = []
    marker = ""
    while True:
        url = (f"{LIST_API}?prefix=data/futures/um/monthly/{sub}/"
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
    return sorted(set(months))


def _load_field(
    symbol: str, kind: str, start: str, end: str, cache_dir: str
) -> pd.DataFrame | None:
    """Download+cache one symbol's monthly files for 'klines' or
    'fundingRate'; return a parsed frame or None if the symbol never
    traded in the window (every month 404s — a natural delisting signal)."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"{kind}_{symbol}_{start}_{end}.parquet")
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
        return df if len(df) else None

    lo, hi = start[:7], end[:7]
    want = [m for m in _available_months(symbol, kind) if lo <= m <= hi]
    frames = []
    for ym in want:
        if kind == "klines":
            url = f"{DUMP}/klines/{symbol}/1d/{symbol}-1d-{ym}.zip"
        else:
            url = f"{DUMP}/fundingRate/{symbol}/{symbol}-fundingRate-{ym}.zip"
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
        text = z.read(z.namelist()[0]).decode()
        # CRITICAL: older months are headerless, newer ones carry a header.
        # Read EVERY month positionally (header=None) and drop a header row
        # if present, so columns are uniform integers across all months --
        # otherwise concat aligns string-named and int-named columns
        # separately and silently shreds any symbol spanning the boundary.
        first = text.splitlines()[0].lower()
        skip = 1 if first.startswith(("open_time", "calc_time")) else 0
        frames.append(pd.read_csv(io.StringIO(text), header=None, skiprows=skip))
        time.sleep(0.05)

    if not frames:
        pd.DataFrame().to_parquet(cache)  # cache the "never traded" answer
        return None
    out = pd.concat(frames, ignore_index=True)
    out.columns = [str(c) for c in out.columns]  # positional 0..n, as strings
    out.to_parquet(cache)
    return out


# Binance kline columns (no header in old files): positional fallback.
_KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume",
               "close_time", "quote_volume", "count", "taker_buy_volume",
               "taker_buy_quote_volume", "ignore"]
_FUNDING_COLS = ["calc_time", "funding_interval_hours", "last_funding_rate"]


def load_klines(symbol, start="2019-09-01", end=None, cache_dir=CACHE):
    """(date-indexed) close + quote_volume for one symbol, or None."""
    end = end or pd.Timestamp.today().strftime("%Y-%m-01")
    df = _load_field(symbol, "klines", start, end, cache_dir)
    if df is None:
        return None
    if df.columns[0] != "open_time":  # headerless old months
        df.columns = _KLINE_COLS[: df.shape[1]]
    df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.normalize()
    out = df.groupby("date").agg(
        close=("close", "last"), quote_volume=("quote_volume", "sum")
    )
    return out[~out.index.duplicated(keep="last")]


def load_funding(symbol, start="2019-09-01", end=None, cache_dir=CACHE):
    """Daily funding (SUM of that day's 8h settlements) for one symbol."""
    end = end or pd.Timestamp.today().strftime("%Y-%m-01")
    df = _load_field(symbol, "fundingRate", start, end, cache_dir)
    if df is None:
        return None
    if df.columns[0] != "calc_time":
        df.columns = _FUNDING_COLS[: df.shape[1]]
    df["date"] = pd.to_datetime(df["calc_time"], unit="ms").dt.normalize()
    return df.groupby("date")["last_funding_rate"].sum()


def build_panels(
    symbols: list[str],
    start: str = "2019-09-01",
    end: str | None = None,
    cache_dir: str = CACHE,
    progress: bool = True,
) -> dict[str, pd.DataFrame]:
    """(date x symbol) price, dollar-volume, and daily-funding panels.

    A symbol contributes only the dates it actually traded; everything
    else is NaN, so listing/delisting are represented honestly. Symbols
    with no klines in the window are dropped (and listed in attrs).
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-01")
    closes, vols, funds, missing, errored = {}, {}, {}, [], []
    for i, sym in enumerate(symbols):
        if progress and i % 25 == 0:
            print(f"  [perp_data] {i}/{len(symbols)} {sym}", flush=True)
        # Per-symbol isolation: one bad symbol (a transient HTTP error, a
        # malformed file) must never kill a 729-symbol run. It is recorded
        # and skipped; re-running picks it up from cache or retries it.
        try:
            k = load_klines(sym, start, end, cache_dir)
            if k is None or k.empty:
                missing.append(sym)
                continue
            closes[sym] = k["close"].astype(float)
            vols[sym] = k["quote_volume"].astype(float)
            f = load_funding(sym, start, end, cache_dir)
            if f is not None:
                funds[sym] = f.astype(float)
        except Exception as exc:  # noqa: BLE001 -- resilience over purity here
            errored.append(sym)
            print(f"  [perp_data] SKIP {sym}: {type(exc).__name__} "
                  f"{str(exc)[:80]}", flush=True)

    price = pd.DataFrame(closes).sort_index()
    panels = {
        "price": price,
        "dollar_volume": pd.DataFrame(vols).reindex_like(price),
        "funding": pd.DataFrame(funds).reindex_like(price),
    }
    for p in panels.values():
        p.attrs["missing_symbols"] = missing
        p.attrs["errored_symbols"] = errored
        p.attrs["data_end"] = end
    return panels
