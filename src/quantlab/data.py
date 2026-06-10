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


def load_prices(
    tickers: list[str] | None = None,
    start: str = "2010-01-01",
    end: str | None = None,
    cache_dir: str = "data_cache",
) -> pd.DataFrame:
    """Return a (date x ticker) DataFrame of adjusted close prices.

    Downloads via yfinance and caches to parquet so repeated runs are offline.
    """
    tickers = tickers or DEFAULT_UNIVERSE
    os.makedirs(cache_dir, exist_ok=True)
    # Key on ticker *content*, not count: two different universes of the same
    # size must never silently share a cache file.
    digest = hashlib.md5(",".join(sorted(tickers)).encode()).hexdigest()[:10]
    key = f"prices_{digest}_{start}_{end or 'latest'}.parquet"
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

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    prices = prices.dropna(how="all").dropna(axis=1, thresh=int(len(prices) * 0.9))
    prices.to_parquet(cache_path)
    return prices
