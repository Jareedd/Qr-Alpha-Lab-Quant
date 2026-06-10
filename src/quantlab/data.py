"""Data loading: real prices via yfinance (with local cache) or synthetic panels."""

from __future__ import annotations

import hashlib
import os

import pandas as pd

# A default universe of liquid US large caps + sector ETFs (free data, survivorship-
# biased by construction -- see README "Known limitations").
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "MA",
    "UNH", "HD", "PG", "KO", "PEP", "MRK", "ABBV", "XOM", "CVX", "WMT",
    "BAC", "DIS", "CSCO", "ADBE", "CRM", "NFLX", "INTC", "AMD", "QCOM", "TXN",
    "HON", "CAT", "BA", "GE", "MMM", "UPS", "RTX", "LMT", "GS", "MS",
    "C", "WFC", "T", "VZ", "CMCSA", "PFE", "JNJ", "LLY", "TMO", "ABT",
    "XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE",
]


def _download_field(
    tickers: list[str],
    field: str,
    start: str,
    end: str | None,
    cache_dir: str,
    min_coverage: float,
    chunk_size: int,
) -> pd.DataFrame:
    """Chunked yfinance download of one OHLCV field, cached to parquet."""
    os.makedirs(cache_dir, exist_ok=True)
    # Key on ticker *content*, not count: two different universes of the same
    # size must never silently share a cache file.
    digest = hashlib.md5(",".join(sorted(tickers)).encode()).hexdigest()[:10]
    # "prices" kept as the Close prefix so pre-refactor caches stay valid.
    prefix = "prices" if field == "Close" else field.lower()
    key = f"{prefix}_{digest}_{start}_{end or 'latest'}_{min_coverage}.parquet"
    cache_path = os.path.join(cache_dir, key)
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "yfinance is required for real data: pip install yfinance. "
            "For offline testing use synthetic data (see quantlab.synthetic)."
        ) from exc

    frames = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        raw = yf.download(chunk, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            continue
        part = raw[field] if isinstance(raw.columns, pd.MultiIndex) else raw[[field]]
        if not isinstance(raw.columns, pd.MultiIndex):
            part.columns = chunk[:1]
        frames.append(part)
    out = pd.concat(frames, axis=1).sort_index()
    out = out.loc[:, ~out.columns.duplicated()]
    out = out.dropna(how="all").dropna(axis=1, how="all")
    if min_coverage > 0:
        out = out.dropna(axis=1, thresh=int(len(out) * min_coverage))
    out.to_parquet(cache_path)
    return out


def load_prices(
    tickers: list[str] | None = None,
    start: str = "2010-01-01",
    end: str | None = None,
    cache_dir: str = "data_cache",
    min_coverage: float = 0.9,
    chunk_size: int = 100,
) -> pd.DataFrame:
    """Return a (date x ticker) DataFrame of adjusted close prices.

    Downloads via yfinance (in chunks, to be polite to the API at large
    universe sizes) and caches to parquet so repeated runs are offline.

    ``min_coverage``: drop columns with less than this fraction of non-NaN
    rows. The 0.9 default suits a static always-listed universe; pass 0.0 for
    point-in-time universes, where names that IPO'd or delisted mid-window
    are EXACTLY the ones survivorship-bias work needs to keep.
    """
    tickers = tickers or DEFAULT_UNIVERSE
    return _download_field(
        tickers, "Close", start, end, cache_dir, min_coverage, chunk_size
    )


def load_volumes(
    tickers: list[str] | None = None,
    start: str = "2010-01-01",
    end: str | None = None,
    cache_dir: str = "data_cache",
    chunk_size: int = 100,
) -> pd.DataFrame:
    """Share volumes (for dollar-ADV / impact modeling), cached like prices.

    No coverage filter: missing volume simply means a name falls back to the
    cross-sectional median ADV inside the impact model (and is counted in
    adv_coverage).
    """
    tickers = tickers or DEFAULT_UNIVERSE
    return _download_field(tickers, "Volume", start, end, cache_dir, 0.0, chunk_size)
