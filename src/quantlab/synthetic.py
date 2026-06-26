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


_VALUE_SEED_OFFSET = 5150   # separate-rng offset for ALL value-world randomness;
                            # distinct from make_panel's seed+777. Never reuse the
                            # main seed or the value draws correlate with q/idio.
_VALUE_PREMIUM = 0.05       # mean of the HML factor return series val_f (the value
                            # premium that makes World A's raw SR positive). MUST be
                            # nonzero: a zero-mean factor gives E[ret]=loading*0=0
                            # and the discrimination evaporates.
_VALUE_VOL = 0.08           # per-period std of val_f; > idio (0.06) so a 36/18
                            # rolling HML beta is identifiable and neutralization bites.
_WORLD_B_PREMIUM = 0.004    # World B's genuine value-ORTHOGONAL quality alpha,
                            # CALIBRATED so World B raw SR overlaps World A's (~2.4),
                            # NOT premium=0.02 which gave SR ~12 and made the worlds
                            # separable by raw SR alone (the discrimination must come
                            # from neutralization, not headroom). Hardcoded — NOT a
                            # tunable knob (these are falsification worlds).
_QUALITY_MODES = {"planted_quality", "null_quality",
                  "quality_is_value", "quality_orthogonal"}


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

    Two additional modes turn the H1 raw-vs-HML-neutral decision into a known-
    answer test (the discrimination must come from neutralization, not a
    Sharpe-level gap — see ``_WORLD_B_PREMIUM``):

    - ``quality_is_value``: the firm's value loading IS its quality z-score
      (collinear), and the predictable return arrives via the value factor — so
      a raw quality book has positive Sharpe but an HML-neutralized book
      COLLAPSES (the "alpha" was the value tilt).
    - ``quality_orthogonal``: quality predicts idiosyncratic return orthogonal
      to value — so both raw and HML-neutralized books KEEP their Sharpe.

    SYNTHETIC, harness-validation only, labeled as such (law #7).
    """
    if mode not in _QUALITY_MODES:
        raise ValueError(
            "mode must be one of 'planted_quality', 'null_quality', "
            f"'quality_is_value', 'quality_orthogonal', got {mode!r}"
        )
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(n_firms)                              # persistent quality
    gp_noise = rng.standard_normal((n_periods, n_firms))
    gp_a = 0.15 + 0.05 * q[None, :] + 0.01 * gp_noise            # observable GP/A ~0.1–0.2

    betas = rng.uniform(0.6, 1.2, n_firms)
    mkt = rng.standard_normal(n_periods) * 0.04
    idio = rng.standard_normal((n_periods, n_firms)) * 0.06
    qz = (q - q.mean()) / (q.std() + 1e-12)

    value_loading = None
    value_factor = None
    if mode in ("planted_quality", "null_quality"):
        prem = premium * qz[None, :] if mode == "planted_quality" else 0.0
        rets = betas[None, :] * mkt[:, None] + idio + prem
    else:
        # SEPARATE rng, created AFTER all 5 main draws so the old modes' byte
        # stream is untouched (val_rng never instantiated for them).
        val_rng = np.random.default_rng(seed + _VALUE_SEED_OFFSET)
        val_f = _VALUE_PREMIUM + val_rng.standard_normal(n_periods) * _VALUE_VOL
        raw = val_rng.standard_normal(n_firms)
        if mode == "quality_is_value":
            # World A: value loading IS the quality z-score (collinear). The
            # entire predictable return arrives through the value factor's
            # positive mean times the loading: E[ret] = qz * _VALUE_PREMIUM.
            value_loading = qz.copy()
            rets = (betas[None, :] * mkt[:, None] + idio
                    + value_loading[None, :] * val_f[:, None])
        else:  # quality_orthogonal — World B
            # Gram-Schmidt residualize the value loading AGAINST qz so
            # corr(value_loading, qz) == 0, then re-zscore to match scale.
            vl0 = raw - (raw @ qz / (qz @ qz)) * qz
            value_loading = (vl0 - vl0.mean()) / (vl0.std() + 1e-12)
            prem = _WORLD_B_PREMIUM * qz       # genuine value-ORTHOGONAL alpha
            rets = (betas[None, :] * mkt[:, None] + idio + prem[None, :]
                    + value_loading[None, :] * val_f[:, None])
        value_factor = val_f

    periods = pd.bdate_range("2010-01-31", periods=n_periods, freq="BME")
    firms = [f"FIRM{i:03d}" for i in range(n_firms)]
    price = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=periods, columns=firms)
    price.attrs["gp_a"] = pd.DataFrame(gp_a, index=periods, columns=firms)
    price.attrs["mode"] = mode
    if value_loading is not None:                                # NEW modes ONLY
        price.attrs["value_loading"] = pd.Series(value_loading, index=firms)
        price.attrs["book_to_market"] = pd.Series(value_loading, index=firms)  # alias
        price.attrs["value_factor"] = pd.Series(value_factor, index=periods)
        price.attrs["betas"] = pd.Series(betas, index=firms)
        # deterministic sector map drawn from val_rng AFTER the two value draws
        # (stream stays clean); defensive ground truth for a future
        # market+sector+HML neutralization test.
        sect = val_rng.integers(0, 6, n_firms)
        price.attrs["sector"] = {firms[i]: f"S{int(sect[i])}" for i in range(n_firms)}
    return price


_INSIDER_MODES = {"planted_opportunistic", "null_opportunistic"}
_INSIDER_SEED_OFFSET = 9090   # separate-rng offset for ALL insider-world draws;
                              # distinct from make_panel's seed+777 and the value
                              # world's +5150. A dedicated rng means this function
                              # NEVER perturbs any existing make_* draw stream.


def make_insider_panel(
    mode: str = "planted_opportunistic",
    seed: int = 7,
    n_firms: int = 120,
    n_periods: int = 120,
    window_days: int = 90,
    n_routine: int = 40,
    n_opportunistic: int = 60,
    cluster_premium: float = 0.06,
) -> pd.DataFrame:
    """Synthetic insider-cluster-buy world for the H10 falsification gate.

    Returns a (period x firm) price panel; the synthetic Form-4 PURCHASE event
    table rides in ``attrs["purchases"]`` (long-form, indexed by ``filed_date``,
    columns ``owner_name, role, shares, value, transaction_date, ticker``) — the
    exact shape ``insider.cluster_buy_signal`` consumes. ``attrs["window_days"]``
    carries the signal window the gate should use.

    The H10-specific subtlety this world encodes: the NULL is not "no insider
    buying" — insiders plainly buy — it is "buying does not PREDICT returns". And
    the planted signal must live ONLY in OPPORTUNISTIC clusters, never routine
    ones, so the harness's routine/opportunistic split is exercised. Two paired
    modes (identical draws except the return link):

    - ``planted_opportunistic``: on scattered event months a CLUSTER of distinct
      OPPORTUNISTIC insiders buys a firm, and that firm earns a positive forward
      idiosyncratic return ``cluster_premium`` the NEXT period. ROUTINE insiders
      also buy (same calendar month every year) but their buys are return-neutral
      by construction — a correct harness must classify them out and still recover
      the opportunistic premium. The H10 machinery must RECOVER this.
    - ``null_opportunistic``: the SAME opportunistic + routine buy events occur,
      but they are INDEPENDENT of forward returns (no premium). The machinery must
      find NOTHING here (finding cluster-buy alpha in this world = the harness is
      broken or is leaking — the exact failure this world guards against).

    Routine insiders are planted by giving each one a fixed "home" firm and a fixed
    calendar month, and buying there every year for the whole sample (the >=3-
    consecutive-year same-month pattern ``classify_routine_opportunistic`` brands
    routine). Opportunistic insiders buy in sporadic, non-repeating months so they
    never establish that pattern.

    SYNTHETIC, harness-validation only, labeled as such (law #7). Drawn entirely
    from a DEDICATED rng (seed + ``_INSIDER_SEED_OFFSET``) so no existing make_*
    draw sequence is touched.
    """
    if mode not in _INSIDER_MODES:
        raise ValueError(
            "mode must be 'planted_opportunistic' or 'null_opportunistic', "
            f"got {mode!r}"
        )
    rng = np.random.default_rng(seed + _INSIDER_SEED_OFFSET)

    firms = [f"INS{i:03d}" for i in range(n_firms)]
    periods = pd.bdate_range("2008-01-31", periods=n_periods, freq="BME")
    is_planted = mode == "planted_opportunistic"

    # Base return components (market + idiosyncratic). The planted premium is
    # injected on top for firms that received an OPPORTUNISTIC cluster the prior
    # period; routine buys never move returns.
    betas = rng.uniform(0.6, 1.2, n_firms)
    mkt = rng.standard_normal(n_periods) * 0.04
    idio = rng.standard_normal((n_periods, n_firms)) * 0.06
    rets = betas[None, :] * mkt[:, None] + idio

    purchase_rows: list[dict] = []

    # --- ROUTINE insiders: a fixed home firm + fixed calendar month, every year.
    # These establish the >=3-consecutive-year same-month pattern and must be
    # classified OUT; their buys never carry a premium in EITHER mode.
    years = sorted({d.year for d in periods})
    for k in range(n_routine):
        home = firms[int(rng.integers(0, n_firms))]
        month = int(rng.integers(1, 13))
        owner = f"ROUTINE_{k:03d}"
        for y in years:
            # place the routine buy on a period in (y, month) if the sample covers
            # it; the transaction_date is the calendar date, filed ~ same day.
            day = pd.Timestamp(year=y, month=month, day=15)
            if periods[0] <= day <= periods[-1]:
                purchase_rows.append({
                    "filed_date": day,
                    "owner_name": owner,
                    "role": "officer",
                    "shares": float(rng.integers(500, 5000)),
                    "value": np.nan,
                    "transaction_date": day,
                    "ticker": home,
                })

    # --- OPPORTUNISTIC clusters: on scattered event periods, several DISTINCT
    # opportunistic insiders buy the SAME firm. In the planted world that firm
    # earns the premium the NEXT period; in the null world it does not.
    opp_pool = [f"OPP_{j:04d}" for j in range(n_opportunistic * 8)]
    # SEVERAL cluster events per period on distinct firms (keeps every rebalance's
    # long quintile populated by genuinely-clustered names, so per-seed variance is
    # low and the gate's known answer is clean). Skip the last period so a forward
    # premium always has a landing period.
    n_clusters_per_period = max(2, n_firms // 20)
    for t in range(n_periods - 1):
        chosen = rng.choice(n_firms, size=n_clusters_per_period, replace=False)
        for firm_idx in chosen:
            firm_idx = int(firm_idx)
            firm = firms[firm_idx]
            n_buyers = int(rng.integers(3, 7))  # a CLUSTER: several distinct insiders
            buyers = rng.choice(opp_pool, size=n_buyers, replace=False)
            d = periods[t]
            for b in buyers:
                # jitter the filed date a few days around the period end (still
                # <= t at the next rebalance) so distinct-owner counting is
                # exercised.
                jitter = int(rng.integers(0, 20))
                fd = d - pd.Timedelta(days=jitter)
                purchase_rows.append({
                    "filed_date": fd,
                    "owner_name": str(b),
                    "role": "director",
                    "shares": float(rng.integers(1000, 10000)),
                    "value": np.nan,
                    "transaction_date": fd,
                    "ticker": firm,
                })
            if is_planted:
                # planted: the cluster precedes positive forward idiosyncratic
                # returns on THAT firm while the buys stay inside the signal's
                # trailing window. The signal flags the firm for ~window_days after
                # the cluster, so the book holds it across those rebalances; plant
                # the premium on exactly those forward returns (t+1 .. t+hold) so
                # the HELD position earns it — alignment, not extra magnitude.
                # ``hold`` ≈ window in periods.
                hold = max(1, round(window_days / 30))
                for h in range(1, hold + 1):
                    if t + h < n_periods:
                        rets[t + h, firm_idx] += cluster_premium

    price = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rets, axis=0)), index=periods, columns=firms)
    purchases = (pd.DataFrame(purchase_rows)
                 .set_index("filed_date").sort_index())
    purchases.index.name = "filed_date"
    price.attrs["purchases"] = purchases
    price.attrs["window_days"] = int(window_days)
    price.attrs["mode"] = mode
    price.attrs["betas"] = pd.Series(betas, index=firms)
    return price


_PEAD_MODES = {"planted_pead", "null_pead"}
_PEAD_SEED_OFFSET = 7171      # separate-rng offset for ALL PEAD-world draws;
                             # distinct from make_panel's +777, the value world's
                             # +5150 and the insider world's +9090. A dedicated rng
                             # means this function NEVER perturbs any existing make_*
                             # draw stream (pinned byte-identical by the
                             # falsification gate).


def make_pead_panel(
    mode: str = "planted_pead",
    seed: int = 7,
    n_firms: int = 80,
    n_quarters: int = 24,
    drift: float = 0.08,
    hold: int = 60,
    enter_lag: int = 2,
    sue_threshold: float = 1.0,
) -> pd.DataFrame:
    """Synthetic post-earnings-announcement-drift world for the H13 gate.

    Returns a (day x firm) total-return PRICE panel; the synthetic earnings-
    surprise EVENT table rides in ``attrs["events"]`` (long-form, columns
    ``ticker, ann_date, period, actual_eps, est_eps, std_est, sue`` — the exact
    shape ``pead.parse_pead_csv`` / ``pead.compute_sue`` consume after a CSV
    round-trip). ``attrs["mode"]`` / ``attrs["drift"]`` carry the ground truth.

    The PEAD-specific subtlety this world encodes: the NULL is not "no earnings
    surprises" — surprises plainly exist — it is "surprises do not PREDICT post-
    announcement drift". And the planted drift must live ONLY in high-|SUE|
    events and ONLY AFTER the announcement (entry T+``enter_lag``), so the
    harness's T+2 PIT entry and drift-vs-reaction control are exercised. Two
    paired modes (identical draws — same firms, same announcements, same SUEs —
    except the return link):

    - ``planted_pead``: for each event with |SUE| >= ``sue_threshold``, a
      cumulative ``drift`` (signed by the surprise) is injected over the ``hold``
      trading days that begin at announcement + ``enter_lag`` (so the drift is
      strictly POST-T+2 — a harness that enters on/before the announcement bar
      captures none of it, and one that enters at T+2 captures it all). The H13
      machinery must RECOVER this. The drift PERSISTS (geometric ramp then level
      shift), so re-entering at T+5/T+10 still captures >=50% (the drift-vs-
      reaction control's PASS side).
    - ``null_pead``: the SAME events and SUEs, but NO post-event drift — returns
      are pure market + idiosyncratic. The machinery must find NOTHING here
      (finding PEAD in this world = the harness is broken / leaking — the exact
      failure this world guards against).

    PIT-safety of the construction: the drift is placed ONLY on bars strictly
    after announcement + ``enter_lag``; the announcement bar and the ``enter_lag``
    skip bars carry ZERO planted drift, so the synthetic world itself contains no
    same-bar leakage for the harness to accidentally exploit.

    SYNTHETIC, harness-validation only, labeled as such (law #7). Drawn entirely
    from a DEDICATED rng (seed + ``_PEAD_SEED_OFFSET``) so no existing make_*
    draw sequence is touched.
    """
    if mode not in _PEAD_MODES:
        raise ValueError(
            f"mode must be 'planted_pead' or 'null_pead', got {mode!r}"
        )
    rng = np.random.default_rng(seed + _PEAD_SEED_OFFSET)
    is_planted = mode == "planted_pead"

    # Daily price panel long enough to span every event's hold window. ~63
    # trading days per quarter; pad a year of warm-up + a hold tail.
    q_days = 63
    n_days = 252 + n_quarters * q_days + hold + enter_lag + 5
    firms = [f"PEAD{i:03d}" for i in range(n_firms)]
    dates = pd.bdate_range("2014-01-02", periods=n_days)

    betas = rng.uniform(0.6, 1.4, n_firms)
    idio_vol = rng.uniform(0.010, 0.020, n_firms)
    mkt = rng.standard_normal(n_days) * 0.009 + 0.0002
    rets = (betas[None, :] * mkt[:, None]
            + rng.standard_normal((n_days, n_firms)) * idio_vol[None, :])

    # Announcement calendar: one announcement per firm per quarter, on a per-firm
    # jittered day near each quarter end (real earnings cluster but are staggered).
    first_ann = 252  # after the warm-up year
    event_rows: list[dict] = []
    for fi, firm in enumerate(firms):
        jitter = int(rng.integers(0, 15))
        for q in range(n_quarters):
            pos = first_ann + q * q_days + jitter
            if pos + enter_lag + hold >= n_days:
                continue
            ann_date = dates[pos]
            # A surprise: standardized SUE drawn ~N(0,1.2) so a healthy tail
            # exceeds the |SUE|>=1 planted threshold; build est/actual/std to be
            # CONSISTENT with that SUE so the CSV round-trip recovers it.
            sue = float(rng.standard_normal() * 1.2)
            est_eps = float(rng.uniform(0.5, 3.0))
            std_est = float(rng.uniform(0.05, 0.20))
            actual_eps = est_eps + sue * std_est        # => (actual-est)/std = sue
            year = ann_date.year
            quarter = (ann_date.month - 1) // 3 + 1
            event_rows.append({
                "ticker": firm, "ann_date": ann_date,
                "period": f"{year}Q{quarter}",
                "actual_eps": round(actual_eps, 4), "est_eps": round(est_eps, 4),
                "std_est": round(std_est, 4), "sue": sue,
                "_pos": pos, "_fi": fi,
            })

    # Plant the drift (planted mode only) on bars STRICTLY after T+enter_lag.
    if is_planted:
        bump_base = (1.0 + drift)
        for ev in event_rows:
            if abs(ev["sue"]) < sue_threshold:
                continue
            sign = 1.0 if ev["sue"] > 0 else -1.0
            entry = ev["_pos"] + enter_lag           # first BAR we may hold from
            end = min(entry + hold, n_days)
            steps = end - entry
            if steps <= 0:
                continue
            # signed cumulative drift over the hold window: geometric per-bar
            # increment so the position gains ~sign*drift across the window and
            # then holds the level shift (persistence -> T+5/T+10 still captures).
            per_bar = bump_base ** (sign / hold) - 1.0
            rets[entry + 1: end + 1, ev["_fi"]] += per_bar

    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=firms)

    events = (pd.DataFrame(event_rows)
              .drop(columns=["_pos", "_fi"])
              .reset_index(drop=True))
    events["ann_date"] = pd.to_datetime(events["ann_date"])
    prices.attrs["events"] = events
    prices.attrs["mode"] = mode
    prices.attrs["drift"] = float(drift) if is_planted else 0.0
    prices.attrs["hold"] = int(hold)
    prices.attrs["enter_lag"] = int(enter_lag)
    prices.attrs["sue_threshold"] = float(sue_threshold)
    prices.attrs["betas"] = pd.Series(betas, index=firms)
    return prices


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
