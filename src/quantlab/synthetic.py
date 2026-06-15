"""Synthetic price panels for pipeline validation.

Two modes, both essential research hygiene:

- ``planted``: returns contain a small, known momentum-like predictable component.
  A correct pipeline must RECOVER it (positive out-of-sample Sharpe, DSR > 0.5).
- ``noise``: returns are pure noise with realistic vol clustering and correlation.
  A correct pipeline must NOT find anything (DSR should fail to reject luck).
- ``planted_regime``: the planted momentum signal exists ONLY in a persistent
  low-volatility regime (a hidden 2-state Markov chain; high-vol states run
  2.5x market vol with the signal switched off). Ground truth rides in
  ``attrs["regimes"]``. This is the falsification world for regime-detection
  machinery (quantlab.regime): a causal detector must add value here and a
  leaky one must be caught -- BEFORE either is allowed near real data.

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
    if mode not in {"planted", "noise", "planted_regime"}:
        raise ValueError(
            f"mode must be 'planted', 'noise' or 'planted_regime', got {mode!r}"
        )
    rng = np.random.default_rng(seed)

    # planted_regime only: a persistent hidden chain (expected duration
    # ~100 trading days per state). Drawn from a SEPARATE rng so the
    # planted/noise draw sequences -- and therefore the falsification-gate
    # baselines -- stay byte-identical to before this mode existed.
    regimes = np.zeros(n_days, dtype=int)
    if mode == "planted_regime":
        regime_rng = np.random.default_rng(seed + 777)
        stay = 0.99
        for t in range(1, n_days):
            if regime_rng.random() >= stay:
                regimes[t] = 1 - regimes[t - 1]
            else:
                regimes[t] = regimes[t - 1]

    n_sectors = 6
    sectors = rng.integers(0, n_sectors, n_assets)
    betas = rng.uniform(0.6, 1.4, n_assets)
    if mode == "planted_regime":
        # Uniform betas, AFTER the draw (so planted/noise rng sequences are
        # untouched). Reason, measured during development: with dispersed
        # betas, beta-ESTIMATION error interacting with the 2.5x vol regime
        # manufactures momentum-vs-residual-label IC of ~+0.13 in stressed
        # states on signal-free data -- an artifact that exactly masked the
        # planted on/off differential. A falsification world must have a
        # clean known answer; the artifact itself is documented in the log
        # (it is a live caution for any real-data regime claim).
        betas = np.ones(n_assets)

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
        # Stressed regime (planted_regime only): market vol jumps 2.5x --
        # the observable footprint a volatility-regime detector keys on.
        mkt_t = mkt[t] * (2.5 if regimes[t] else 1.0)
        r = betas * mkt_t + sector_f[t, sectors] + idio
        if mode in ("planted", "planted_regime") and t > lookback:
            past = rets[t - lookback : t - skip].sum(axis=0)
            z = (past - past.mean()) / (past.std() + 1e-12)
            # In planted_regime the signal exists ONLY in the calm state:
            # ground truth for "momentum works conditionally".
            on = 1.0 if (mode == "planted" or regimes[t] == 0) else 0.0
            r += on * signal_strength * z * idio_vol  # weak, vol-scaled momentum
        rets[t] = r

    dates = pd.bdate_range("2012-01-02", periods=n_days)
    tickers = [f"SYN{i:03d}" for i in range(n_assets)]
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    panel = pd.DataFrame(prices, index=dates, columns=tickers)
    # Ground truth for risk-neutralization tests; attrs ride along with the
    # frame without changing the (long-stable) return signature.
    panel.attrs["sectors"] = {t: f"S{sectors[i]}" for i, t in enumerate(tickers)}
    panel.attrs["betas"] = {t: float(betas[i]) for i, t in enumerate(tickers)}
    if mode == "planted_regime":
        # Ground truth (0 = calm/signal-on, 1 = stressed/signal-off): the
        # known answer regime-detection machinery is falsified against.
        panel.attrs["regimes"] = pd.Series(regimes, index=dates, name="regime")
    return panel


def make_perp_panel(
    n_assets: int = 30,
    n_days: int = 1500,
    mode: str = "planted_carry",
    seed: int = 7,
    carry_premium: float = 0.3,
) -> pd.DataFrame:
    """Synthetic perpetual-futures world for the H2 falsification gate.

    Returns a (date x contract) MARK PRICE panel; the funding-rate panel
    rides in ``attrs["funding"]`` (daily rate, sign convention: positive =
    longs pay shorts, the crypto-perp norm).

    The carry-specific subtlety this world exists to encode: the NULL of a
    carry strategy is not "funding doesn't exist" — funding is plainly
    observable — it is "funding is fully priced": the mark price drifts in
    favor of the side that pays, so funding income is exactly offset and a
    funding-ranked book earns nothing. The two modes:

    - ``planted_carry``: a fraction ``carry_premium`` (gamma) of funding is
      a TRUE premium. Price return r = beta*mkt + idio + (1-gamma)*F, so a
      long's funding-inclusive total return is r - F = beta*mkt + idio -
      gamma*F: shorting high-funding contracts harvests gamma*F. The H2
      machinery must RECOVER this.
    - ``priced_carry``: gamma = 0 — same world, funding fully compensated
      by drift; total returns are unpredictable from funding. The machinery
      must find NOTHING here (finding carry in this world = the harness is
      broken or the label is funding-exclusive, the exact bug this world
      guards against).

    Funding dynamics: persistent AR(1) per contract (phi=0.97) around a
    positive cross-sectional mean (~11%/yr — the perp baseline), with
    dispersion and occasional negative-funding contracts, so trailing
    funding ranks are slow-moving and realistic.

    Like everything in this module: SYNTHETIC, for harness validation only,
    and labeled as such (law #7).
    """
    if mode not in {"planted_carry", "priced_carry"}:
        raise ValueError(
            f"mode must be 'planted_carry' or 'priced_carry', got {mode!r}"
        )
    rng = np.random.default_rng(seed)
    gamma = carry_premium if mode == "planted_carry" else 0.0

    betas = rng.uniform(0.8, 1.2, n_assets)  # vs the crypto market factor
    idio_vol = rng.uniform(0.02, 0.05, n_assets)  # perps are wild
    mkt = rng.standard_normal(n_days) * 0.025

    # Funding: AR(1) around a positive mean, daily units.
    f_mean = rng.normal(0.0003, 0.0004, n_assets)  # some contracts negative
    funding = np.empty((n_days, n_assets))
    funding[0] = f_mean
    shocks = rng.standard_normal((n_days, n_assets)) * 0.0004
    for t in range(1, n_days):
        funding[t] = f_mean + 0.97 * (funding[t - 1] - f_mean) + shocks[t]

    idio = rng.standard_normal((n_days, n_assets)) * idio_vol
    # Mark-price return: funding is (1-gamma) priced into drift; the
    # remaining gamma is the planted premium a short-the-payers book earns.
    rets = betas[None, :] * mkt[:, None] + idio + (1.0 - gamma) * funding

    dates = pd.bdate_range("2019-01-01", periods=n_days)
    contracts = [f"PERP{i:02d}" for i in range(n_assets)]
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=contracts
    )
    prices.attrs["funding"] = pd.DataFrame(
        funding, index=dates, columns=contracts
    )
    prices.attrs["carry_gamma"] = float(gamma)
    prices.attrs["mode"] = mode
    return prices


def make_cef_panel(
    n_funds: int = 120,
    n_weeks: int = 520,
    mode: str = "planted_reversion",
    seed: int = 7,
    reversion_phi: float = 0.95,
) -> pd.DataFrame:
    """Synthetic closed-end-fund world for the H6 (trial #11) falsification gate.

    Returns a (week x fund) TOTAL-RETURN price panel (distributions reinvested);
    the NAV and discount panels ride in ``attrs["nav"]`` and ``attrs["discount"]``.
    By construction ``discount = (price - nav)/nav`` exactly, so the harness's
    ``cef.discount`` recovers it.

    The H6-specific subtlety this world encodes: the NULL is not "no discount" —
    discounts plainly exist — it is "discounts do not REVERT" (a random walk). A
    discount-z book earns only if wide-discount extremes narrow on average. Two
    paired modes (identical draws except the reversion coefficient):

    - ``planted_reversion``: each fund's discount is AR(1) with ``reversion_phi``
      < 1, so it mean-reverts toward its own level; a wide-discount extreme (low
      z) drifts back up -> the price rises faster than NAV -> a LONG-low-z /
      SHORT-high-z book harvests the reversion. The H6 machinery must RECOVER it.
    - ``random_walk``: ``phi = 1`` — same shocks, but the discount is a random
      walk with NO reversion; forward returns are unpredictable from the
      discount z. The machinery must find NOTHING here (finding reversion in this
      world = the harness is broken — the exact failure this world guards).

    SYNTHETIC, harness-validation only, labeled as such (law #7).
    """
    if mode not in {"planted_reversion", "random_walk"}:
        raise ValueError(
            f"mode must be 'planted_reversion' or 'random_walk', got {mode!r}"
        )
    rng = np.random.default_rng(seed)
    phi = reversion_phi if mode == "planted_reversion" else 1.0

    # NAV total-return path: a common market factor + idiosyncratic, weekly.
    betas = rng.uniform(0.6, 1.2, n_funds)
    idio_vol = rng.uniform(0.010, 0.025, n_funds)
    mkt = rng.standard_normal(n_weeks) * 0.02
    nav_rets = (betas[None, :] * mkt[:, None]
                + rng.standard_normal((n_weeks, n_funds)) * idio_vol)
    nav = 100.0 * np.exp(np.cumsum(nav_rets, axis=0))

    # Discount process. Shocks are drawn ONCE and shared across modes (paired
    # control): only phi differs, so planted vs random-walk see identical luck.
    d0 = rng.uniform(-0.15, 0.05, n_funds)          # typical CEF discount spread
    dshock = rng.standard_normal((n_weeks, n_funds)) * 0.015
    d = np.empty((n_weeks, n_funds))
    d[0] = d0
    for t in range(1, n_weeks):
        d[t] = np.clip(phi * d[t - 1] + dshock[t], -0.6, 0.4)

    price_tr = nav * (1.0 + d)
    weeks = pd.bdate_range("2012-01-06", periods=n_weeks, freq="W-FRI")
    funds = [f"CEF{i:03d}" for i in range(n_funds)]
    price = pd.DataFrame(price_tr, index=weeks, columns=funds)
    price.attrs["nav"] = pd.DataFrame(nav, index=weeks, columns=funds)
    price.attrs["discount"] = pd.DataFrame(d, index=weeks, columns=funds)
    price.attrs["mode"] = mode
    price.attrs["reversion_phi"] = float(phi)
    return price


def make_quality_panel(
    n_firms: int = 200,
    n_periods: int = 180,
    mode: str = "planted_quality",
    seed: int = 7,
    premium: float = 0.02,
) -> pd.DataFrame:
    """Synthetic fundamentals world for the H1 (quality) falsification gate.

    Returns a (period x firm) price panel; the GROSS-PROFITABILITY panel (GP/A)
    rides in ``attrs["gp_a"]``. Each firm has a persistent latent quality ``q``
    that drives its observable GP/A. Two paired modes (identical draws except the
    return link):

    - ``planted_quality``: high-quality firms earn a small return ``premium`` per
      period, so a long-high-GP/A / short-low-GP/A book harvests it. The H1
      machinery must RECOVER this.
    - ``null_quality``: ``premium = 0`` — GP/A is real but unrelated to returns;
      the book must earn ~nothing (finding quality alpha here = harness broken).

    SYNTHETIC, harness-validation only, labeled as such (law #7).
    """
    if mode not in {"planted_quality", "null_quality"}:
        raise ValueError(f"mode must be 'planted_quality' or 'null_quality', got {mode!r}")
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(n_firms)                              # persistent quality
    gp_noise = rng.standard_normal((n_periods, n_firms))
    gp_a = 0.15 + 0.05 * q[None, :] + 0.01 * gp_noise            # observable GP/A ~0.1–0.2

    betas = rng.uniform(0.6, 1.2, n_firms)
    mkt = rng.standard_normal(n_periods) * 0.04
    idio = rng.standard_normal((n_periods, n_firms)) * 0.06
    qz = (q - q.mean()) / (q.std() + 1e-12)
    prem = premium * qz[None, :] if mode == "planted_quality" else 0.0
    rets = betas[None, :] * mkt[:, None] + idio + prem

    periods = pd.bdate_range("2010-01-31", periods=n_periods, freq="BME")
    firms = [f"FIRM{i:03d}" for i in range(n_firms)]
    price = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=periods, columns=firms)
    price.attrs["gp_a"] = pd.DataFrame(gp_a, index=periods, columns=firms)
    price.attrs["mode"] = mode
    return price


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


def inject_post_event_drift(
    prices: pd.DataFrame,
    events: list[tuple],
    drift: float = 0.10,
    horizon: int = 60,
) -> pd.DataFrame:
    """SCENARIO TOOL, not data: add a known cumulative ``drift`` to each
    named ticker over the ``horizon`` trading days AFTER its event date,
    persisting afterward. The falsification ground truth for the H8 event
    study (quantlab.events): a correct event harness must RECOVER this
    planted post-event drift, and find nothing when drift=0.

    Mechanics: a geometric ramp multiplies the ticker's price so it gains
    exactly ``drift`` over the window vs the no-event counterfactual, then
    holds the level shift. Lives in synthetic.py per law #7 (fabricated
    values only here, always labeled). ``events`` is a list of
    (event_date, ticker) pairs.
    """
    out = prices.copy()
    idx = out.index
    injected = 0
    for d, t in events:
        if t not in out.columns:
            continue
        pos = int(idx.searchsorted(pd.Timestamp(d), side="right"))  # first day > d
        if pos >= len(idx):
            continue
        bump = (1.0 + drift) ** (1.0 / horizon)
        mult = np.ones(len(idx))
        end = min(pos + horizon, len(idx))
        ramp = bump ** np.arange(1, end - pos + 1)
        mult[pos:end] = ramp
        mult[end:] = ramp[-1] if end > pos else 1.0
        out[t] = out[t].to_numpy() * mult
        injected += 1
    out.attrs = dict(prices.attrs)
    out.attrs["event_drift_injected"] = injected
    out.attrs["event_drift"] = float(drift)
    return out
