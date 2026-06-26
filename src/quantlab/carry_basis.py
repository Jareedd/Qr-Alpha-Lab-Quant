"""Cash-and-carry economics for the delta-neutral crypto carry feasibility audit.

PURE functions only (no network, no I/O). This module is the credibility core of
the audit: the realized net carry of a delta-neutral cash-and-carry book and the
boundary (convergence) measurement that decides whether a bucket is PRICED. The
single biggest risk flagged in CLAUDE.md is a SIGN ERROR that manufactures a fake
edge, so every sign here is pinned by a known-answer test in test_carry_basis.py.

THE BOOK (per symbol s, daily-rebalanced, delta-neutral):
    long  $1 of SPOT  s
    short $1 of PERP  s
    collect funding when funding is positive (longs pay shorts on Binance).

DAILY DATA (all from daily klines + daily funding):
    S_t = spot close, P_t = perp close, f_t = that UTC day's TOTAL funding
          (sum of the day's 8h settlements; Binance: f_t > 0 => longs pay shorts).
    r_spot_t = S_t / S_{t-1} - 1        (long spot leg's price return)
    r_perp_t = P_t / P_{t-1} - 1        (perp price return)

LEG RETURNS:
    long  spot leg daily return        =  r_spot
    short perp leg daily TOTAL return  = -r_perp + f_t
        Rationale (PINNED): a SHORT loses the perp's price gain (-r_perp) and
        RECEIVES funding when f_t > 0 (positive funding = longs pay shorts, and
        we are short, i.e. on the receiving side). f_t enters with a PLUS sign
        for the short.

GROSS DAILY CARRY RETURN (sum of the two legs):
    g_t = r_spot - r_perp + f_t
        Decomposes as g_t = (r_spot - r_perp) + f_t. The first term is the
        negative of the daily basis change (convergence drag/credit on the
        spot-vs-perp spread); the second is the funding leg (the income).

BASIS (perp premium):
    B_t = (P_t - S_t) / S_t.   dB_t = B_t - B_{t-1}.
        When the perp trades at a premium that DECAYS toward spot (convergence),
        the short-perp/long-spot book earns; when the premium WIDENS against the
        short it is a mark-to-market loss. (r_spot - r_perp) captures this.

COSTS (LABELED ASSUMPTIONS -- we have NO order books; everything is parametric):
    - round-trip two-leg cost per holding EPISODE (entry+exit, both legs):
      majors ~5 bps, mid ~20 bps, tail ~45 bps. Charged ONCE per episode,
      allocated to the episode's FIRST in-position day.
    - daily rebalance cost ~ rebalance_bps * |r_spot - r_perp| (cost of trading
      the notional drift between the two legs to restore delta-neutrality);
      rebalance_bps ~ half the bucket's per-side cost. Charged every held day.
    All costs are SUBTRACTED from the gross return.

NET DAILY CARRY: n_t = g_t - allocated_cost_t.
REALIZED NET CARRY (per symbol/episode): cumulative sum of n_t; annualized via
365 * mean(daily); Sharpe via mean/std * sqrt(365).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Cost parameters -- LABELED ASSUMPTIONS. We have no order books; these are
# round-trip (entry+exit, BOTH legs) bps per holding episode, by liquidity
# bucket, plus a per-day rebalance bps. Exposed as a dataclass so every result
# carries the exact assumption that produced it.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CostParams:
    roundtrip_bps: float          # entry+exit, both legs, charged ONCE per episode
    rebalance_bps: float          # per-day, applied to |r_spot - r_perp| drift

    @property
    def roundtrip(self) -> float:
        return self.roundtrip_bps * 1e-4

    @property
    def rebalance(self) -> float:
        return self.rebalance_bps * 1e-4


# Per-bucket defaults (round-trip bps; rebalance ~= half the per-SIDE cost,
# i.e. roundtrip/2 per side -> rebalance ~ roundtrip/4 as a per-day fraction of
# drift). These are the numbers the feasibility script reports against.
BUCKET_COSTS: dict[str, CostParams] = {
    "major": CostParams(roundtrip_bps=5.0, rebalance_bps=1.25),
    "mid":   CostParams(roundtrip_bps=20.0, rebalance_bps=5.0),
    "tail":  CostParams(roundtrip_bps=45.0, rebalance_bps=11.25),
}


def _align(*series: pd.Series) -> tuple[pd.Series, ...]:
    """Inner-align several daily series on their common dates (sorted)."""
    df = pd.concat(series, axis=1, join="inner").dropna()
    df = df.sort_index()
    return tuple(df.iloc[:, i] for i in range(df.shape[1]))


def basis(spot: pd.Series, perp: pd.Series) -> pd.Series:
    """Perp premium B_t = (P_t - S_t) / S_t on the common dates."""
    s, p = _align(spot, perp)
    return (p - s) / s


def gross_carry_returns(
    spot: pd.Series, perp: pd.Series, funding: pd.Series
) -> pd.DataFrame:
    """Daily gross cash-and-carry decomposition for one symbol.

    Returns a frame indexed by date with columns:
        r_spot, r_perp, funding, dbasis, gross
    where gross = r_spot - r_perp + funding (the long-spot + short-perp book).
    The first row (no prior close) is dropped.
    """
    s, p, f = _align(spot, perp, funding)
    r_spot = s.pct_change()
    r_perp = p.pct_change()
    b = (p - s) / s
    out = pd.DataFrame({
        "r_spot": r_spot,
        "r_perp": r_perp,
        "funding": f,
        "dbasis": b.diff(),
        # SHORT perp receives funding (+f) and loses perp price move (-r_perp);
        # long spot earns r_spot. This is THE sign that must never flip.
        "gross": r_spot - r_perp + f,
    })
    return out.iloc[1:]  # drop the undefined first-difference row


def cash_and_carry_returns(
    spot: pd.Series,
    perp: pd.Series,
    funding: pd.Series,
    costs: CostParams,
    held: pd.Series | None = None,
) -> pd.DataFrame:
    """Realized DAILY NET carry for an ALWAYS-ON (or masked) cash-and-carry book.

    Parameters
    ----------
    spot, perp, funding : daily series for one symbol (Binance sign convention).
    costs : CostParams (round-trip per episode + per-day rebalance).
    held : optional boolean Series; True on days the book is IN the position.
        Defaults to always-on (held every available day). Used by the
        funding-gated variant to restrict P&L to in-position days.

    Returns a frame with the gross decomposition plus:
        rebalance_cost : per-day rebalance drag (only on held days)
        episode_cost   : the round-trip charge, allocated to each episode's
                         FIRST held day (0 elsewhere)
        net            : gross - rebalance_cost - episode_cost on held days,
                         0.0 on flat days (out of the market => no P&L, no cost).
    """
    g = gross_carry_returns(spot, perp, funding)
    if held is None:
        held_mask = pd.Series(True, index=g.index)
    else:
        held_mask = held.reindex(g.index).fillna(False).astype(bool)

    # Per-day rebalance cost: pay rebalance_bps on the notional drift between
    # legs (|r_spot - r_perp|), only on held days.
    drift = (g["r_spot"] - g["r_perp"]).abs()
    rebalance_cost = costs.rebalance * drift * held_mask.astype(float)

    # Episode entry/exit cost: charge the full round-trip ONCE per contiguous
    # held block, on its first held day. A "rising edge" of held_mask marks a
    # new episode.
    prev = held_mask.shift(1, fill_value=False)
    episode_start = held_mask & (~prev)
    episode_cost = costs.roundtrip * episode_start.astype(float)

    gross = g["gross"].where(held_mask, 0.0)
    net = gross - rebalance_cost - episode_cost

    out = g.copy()
    out["held"] = held_mask
    out["gross"] = gross  # gross masked to held days (0 when flat)
    out["rebalance_cost"] = rebalance_cost
    out["episode_cost"] = episode_cost
    out["net"] = net
    return out


def rolling_funding(funding: pd.Series, window: int = 3) -> pd.Series:
    """Trailing ``window``-day rolling SUM of daily funding (the 72h proxy when
    window=3). Used by the funding-gated hurdle. min_periods=window so we never
    gate on a partial window (no look-behind leakage into the first days)."""
    return funding.sort_index().rolling(window, min_periods=window).sum()


def funding_gated_held(
    spot: pd.Series,
    perp: pd.Series,
    funding: pd.Series,
    costs: CostParams,
    hurdle_mult: float = 3.0,
    window: int = 3,
) -> pd.Series:
    """Boolean held-mask for the FUNDING-GATED variant: be IN only when the
    trailing ``window``-day rolling funding exceeds ``hurdle_mult`` x the
    round-trip cost; flat otherwise.

    The gate uses funding KNOWN AS OF day t (trailing window ending at t), so
    entering at t to earn t+1.. would be PIT-safe; here, at daily resolution and
    for a feasibility estimate, we hold on day t when its trailing window clears
    the hurdle. This is intentionally the simple, defensible rule -- the audit
    reports it as such, no intraday timing claimed.
    """
    s, p, f = _align(spot, perp, funding)
    rf = rolling_funding(f, window)
    hurdle = hurdle_mult * costs.roundtrip
    held = (rf > hurdle).fillna(False)
    # align to the gross-return index (which drops the first day)
    return held


def funding_gated_episodes(
    spot: pd.Series,
    perp: pd.Series,
    funding: pd.Series,
    costs: CostParams,
    hurdle_mult: float = 3.0,
    window: int = 3,
) -> pd.DataFrame:
    """Realized DAILY NET carry for the funding-gated variant (in only above the
    3x hurdle; each contiguous in-block is an episode paying open+close)."""
    held = funding_gated_held(spot, perp, funding, costs, hurdle_mult, window)
    return cash_and_carry_returns(spot, perp, funding, costs, held=held)


def episode_count(held: pd.Series) -> int:
    """Number of contiguous in-position episodes in a boolean held-mask."""
    held = held.astype(bool)
    prev = held.shift(1, fill_value=False)
    return int((held & (~prev)).sum())


def convergence_neutralization(
    funding: pd.Series, basis_series: pd.Series, horizon: int = 1
) -> dict:
    """Measure how much of day-t funding the basis change NEUTRALIZES -- the
    boundary equation, measured directly (the Trial #10 'is it priced?' kill).

    Economic logic: if the perp premium B is fair, the funding a short collects
    should be offset by an adverse basis move. The carry book's basis P&L is
    (r_spot - r_perp) ~= -dB. So the funding NEUTRALIZED by convergence over the
    contemporaneous+next ``horizon`` days is the part of f_t cancelled by the
    summed basis change dB it 'pays for':

        neutralized_t = -sum_{k=0..horizon} dB_{t+k}    (the basis P&L 'against'
                         the funding, expressed as a fraction of f_t)
        frac_t        = neutralized_t / f_t

    We report the MEDIAN fraction over days with meaningfully nonzero funding
    (|f_t| above a small floor, to avoid divide-by-noise), plus an OLS slope of
    the cumulative dB on f as a robustness cross-check.

    Returns a dict: median_frac_neutralized, ols_slope, ols_r2, n.
    A bucket whose median_frac >= 0.80 is PRICED -> DISQUALIFY.
    """
    f, b = _align(funding, basis_series)
    db = b.diff()
    # cumulative basis change over t..t+horizon (the convergence the funding day
    # is associated with). horizon=1 => dB_t + dB_{t+1}.
    cum_db = sum(db.shift(-k) for k in range(0, horizon + 1))
    pair = pd.concat([f, cum_db], axis=1, keys=["f", "cum_db"]).dropna()
    if pair.empty:
        return {"median_frac_neutralized": float("nan"), "ols_slope": float("nan"),
                "ols_r2": float("nan"), "n": 0}

    # The funding the short collects is f; the basis P&L the short earns is
    # (r_spot - r_perp) ~= -cum_db over the window. The fraction of funding
    # NEUTRALIZED by an ADVERSE basis move is (+cum_db) / f: positive funding
    # accompanied by a widening premium (cum_db>0) eats the funding.
    floor = max(1e-6, pair["f"].abs().median() * 0.10)
    sized = pair[pair["f"].abs() > floor]
    if sized.empty:
        frac = float("nan")
    else:
        frac = float((sized["cum_db"] / sized["f"]).median())

    # OLS cross-check: regress cum_db on f. slope ~ fraction of a unit of
    # funding offset by basis convergence (sign: +slope => basis moves WITH and
    # against the short as funding rises => priced).
    x = pair["f"].to_numpy()
    y = pair["cum_db"].to_numpy()
    if len(x) >= 3 and np.std(x) > 0:
        A = np.vstack([x, np.ones_like(x)]).T
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        yhat = A @ np.linalg.lstsq(A, y, rcond=None)[0]
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        slope, r2 = float("nan"), float("nan")

    return {"median_frac_neutralized": frac, "ols_slope": float(slope),
            "ols_r2": float(r2), "n": int(len(pair))}


def hurdle_availability(
    funding: pd.Series, costs: CostParams, hurdle_mult: float = 3.0,
    window: int = 3,
) -> float:
    """Kill criterion 1: fraction of days where the trailing ``window``-day
    rolling funding exceeds ``hurdle_mult`` x round-trip cost (is there enough
    above-hurdle opportunity to harvest?)."""
    rf = rolling_funding(funding, window)
    hurdle = hurdle_mult * costs.roundtrip
    valid = rf.dropna()
    if valid.empty:
        return float("nan")
    return float((valid > hurdle).mean())


def summarize_returns(net: pd.Series, periods: int = 365) -> dict:
    """Annualized stats for a daily net-return series: mean, total (cumulative),
    annualized return (periods x mean), volatility, Sharpe (mean/std*sqrt)."""
    net = net.dropna()
    n = len(net)
    if n == 0:
        return {"n_days": 0, "mean_daily": float("nan"), "total": float("nan"),
                "ann_return": float("nan"), "ann_vol": float("nan"),
                "sharpe": float("nan")}
    mean = float(net.mean())
    std = float(net.std(ddof=1)) if n > 1 else float("nan")
    # A (near-)constant series has no risk: report Sharpe as nan, not a value
    # exploded by floating-point residual std. Tolerance is relative to the
    # mean magnitude so a genuinely tiny-but-real vol is still measured.
    risk_floor = 1e-12 * (abs(mean) + 1.0)
    has_risk = (std == std) and (std > risk_floor)
    sharpe = (mean / std * np.sqrt(periods)) if has_risk else float("nan")
    return {
        "n_days": n,
        "mean_daily": mean,
        "total": float(net.sum()),
        "ann_return": periods * mean,
        "ann_vol": (std * np.sqrt(periods)) if std == std else float("nan"),
        "sharpe": float(sharpe),
    }


# ---------------------------------------------------------------------------
# Tiny synthetic helper (CLEARLY LABELED -- for tests/sanity only, never in the
# audit's data outputs). Builds a known-answer cash-and-carry world.
# ---------------------------------------------------------------------------
def make_synthetic_carry(
    n_days: int = 40,
    daily_funding: float = 0.0005,
    basis_drift: float = 0.0,
    spot_drift: float = 0.0,
    start: str = "2022-01-01",
    seed: int | None = None,
) -> dict[str, pd.Series]:
    """SYNTHETIC (labeled) cash-and-carry world for sign-convention tests.

    Constructs spot/perp/funding so the carry math has a KNOWN answer:
      - spot follows a deterministic drift (spot_drift per day),
      - perp = spot * (1 + basis_t), where basis grows by ``basis_drift``/day,
      - funding is constant ``daily_funding`` per day.
    With basis_drift=0 and spot_drift=0: r_spot=r_perp=0 and gross == funding,
    so a positive funding => positive gross (the headline sign pin).
    Returns {"spot","perp","funding"} daily Series. NOT market data.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    s0 = 100.0
    spot_path = s0 * np.cumprod(1.0 + np.full(n_days, spot_drift))
    basis_path = basis_drift * np.arange(n_days)  # B_t grows linearly
    perp_path = spot_path * (1.0 + basis_path)
    spot = pd.Series(spot_path, index=idx, name="spot")
    perp = pd.Series(perp_path, index=idx, name="perp")
    funding = pd.Series(np.full(n_days, daily_funding), index=idx, name="funding")
    _ = rng  # reserved for noisy variants; deterministic by default
    return {"spot": spot, "perp": perp, "funding": funding}
