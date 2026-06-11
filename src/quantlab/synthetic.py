"""Synthetic price panels for pipeline validation.

Two modes, both essential research hygiene:

- ``planted``: returns contain a small, known momentum-like predictable component.
  A correct pipeline must RECOVER it (positive out-of-sample Sharpe, DSR > 0.5).
- ``noise``: returns are pure noise with realistic vol clustering and correlation.
  A correct pipeline must NOT find anything (DSR should fail to reject luck).

If your pipeline passes both checks, backtest numbers on real data become
meaningfully interpretable. If it "finds alpha" in noise, you have leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_panel(
    n_assets: int = 60,
    n_days: int = 3000,
    mode: str = "planted",
    seed: int = 7,
    signal_strength: float = 0.03,
) -> pd.DataFrame:
    """Return a (date x ticker) price panel.

    Returns are built as: market beta + sector factor + idiosyncratic noise,
    with GARCH-like volatility regimes. In ``planted`` mode, a fraction of each
    asset's idiosyncratic return is predictable from its own trailing 12-1 month
    performance (cross-sectional momentum), scaled by ``signal_strength``.
    """
    if mode not in {"planted", "noise"}:
        raise ValueError(f"mode must be 'planted' or 'noise', got {mode!r}")
    rng = np.random.default_rng(seed)

    n_sectors = 6
    sectors = rng.integers(0, n_sectors, n_assets)
    betas = rng.uniform(0.6, 1.4, n_assets)

    # Volatility regimes (slow-moving, common to the market).
    base_vol = 0.010
    vol = np.empty(n_days)
    vol[0] = base_vol
    for t in range(1, n_days):
        vol[t] = 0.97 * vol[t - 1] + 0.03 * base_vol + 0.002 * abs(rng.standard_normal())

    mkt = rng.standard_normal(n_days) * vol + 0.0002
    sector_f = rng.standard_normal((n_days, n_sectors)) * 0.004

    idio_vol = rng.uniform(0.012, 0.025, n_assets)
    rets = np.zeros((n_days, n_assets))
    lookback, skip = 252, 21

    for t in range(n_days):
        idio = rng.standard_normal(n_assets) * idio_vol
        r = betas * mkt[t] + sector_f[t, sectors] + idio
        if mode == "planted" and t > lookback:
            past = rets[t - lookback : t - skip].sum(axis=0)
            z = (past - past.mean()) / (past.std() + 1e-12)
            r += signal_strength * z * idio_vol  # weak, vol-scaled momentum
        rets[t] = r

    dates = pd.bdate_range("2012-01-02", periods=n_days)
    tickers = [f"SYN{i:03d}" for i in range(n_assets)]
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    panel = pd.DataFrame(prices, index=dates, columns=tickers)
    # Ground truth for risk-neutralization tests; attrs ride along with the
    # frame without changing the (long-stable) return signature.
    panel.attrs["sectors"] = {t: f"S{sectors[i]}" for i, t in enumerate(tickers)}
    panel.attrs["betas"] = {t: float(betas[i]) for i, t in enumerate(tickers)}
    return panel


def inject_delisting_returns(
    prices: pd.DataFrame,
    delist_return: float,
    end_buffer_days: int = 5,
) -> pd.DataFrame:
    """SCENARIO TOOL, not data: append one SYNTHETIC final return to every
    name whose price series ends before the panel does.

    Free data drops the delisting return -- the final, usually ugly, move of
    a dying stock (Shumway 1997: ~-30% for performance delistings). The
    honest way to handle that hole is NOT to impute it (delistings are
    missing *because* the company died -- any model fit on survivors imputes
    survivor-like values, which is survivorship bias re-injected by ML), but
    to BOUND it: re-run the backtest under explicit worst-case assumptions
    and report the spread. This function builds those scenario worlds.

    It lives in synthetic.py on purpose (research law #7: fabricated values
    exist only here and are always labeled). Outputs that use it must carry
    a scenario tag -- run_pipeline appends ``_dlret±NN`` to every artifact.

    Mechanics: for each ticker whose last valid price is at least
    ``end_buffer_days`` trading days before the panel's end (i.e., it died
    mid-window rather than just missing a print), one synthetic price is
    placed on the next trading day: ``last_price * (1 + delist_return)``.
    That single return then flows through features, labels and the
    backtest's P&L exactly as a real final print would.

    The count of touched names rides along in ``attrs['delist_injected']``.
    """
    out = prices.copy()
    n_days = len(out.index)
    injected = 0
    for col in out.columns:
        s = out[col]
        last = s.last_valid_index()
        if last is None:
            continue  # never priced at all -- nothing to extend
        pos = out.index.get_loc(last)
        if pos >= n_days - end_buffer_days:
            continue  # still trading (or died too close to the end to tell)
        out.iloc[pos + 1, out.columns.get_loc(col)] = float(s.loc[last]) * (
            1.0 + delist_return
        )
        injected += 1
    out.attrs = dict(prices.attrs)
    out.attrs["delist_injected"] = injected
    out.attrs["delist_return"] = float(delist_return)
    return out
