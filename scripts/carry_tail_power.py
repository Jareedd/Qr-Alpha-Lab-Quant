"""C1 (long-tail perp carry) — fee-first POWER ANALYSIS. Zero trials.

Decides, BEFORE registering/spending a trial, whether the long-tail perp
universe (ranks below the majors) carries a gross funding spread large enough
to survive realistic tail costs, and whether its sample is deep enough to clear
the DSR hurdle (which scales ~1/sqrt(n_obs)). This is descriptive only: it
computes the cross-sectional funding DISPERSION (the gross carry available) and
the UNIVERSE DEPTH over time. It does NOT compute forward returns, IC, or P&L
-- that is the trial. Mirrors H8's event-count power gate: kill cheaply if the
effect size / power isn't there.

Run:  PYTHONPATH=src .venv/Scripts/python.exe scripts/carry_tail_power.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quantlab import perp_data as pdat  # noqa: E402

START, END = "2019-09-01", "2026-06-01"
RANK_LO, RANK_HI = 31, 150   # the "tail": below the ~top-30 majors, next ~120
MIN_NAMES = 20               # need enough names to form quartiles
SIG_LOOKBACK = 7
ADV_LOOKBACK = 30
QUANTILE = 0.25
REBALANCE = 7                # weekly, as in H2
FUNDING_PERIODS_PER_YEAR = 365  # daily funding (summed 8h settlements)


def tail_universe(dollar_volume: pd.DataFrame) -> pd.DataFrame:
    """Boolean (date x symbol): ADV rank in [RANK_LO, RANK_HI] among names
    trading at t. Past-only rolling ADV. This EXCLUDES the majors (the H2
    universe) and selects the liquid tail beneath them."""
    adv = dollar_volume.rolling(ADV_LOOKBACK,
                                min_periods=max(5, ADV_LOOKBACK // 2)).mean()
    mask = pd.DataFrame(False, index=adv.index, columns=adv.columns)
    arr = adv.to_numpy()
    for i in range(len(adv.index)):
        row = pd.Series(arr[i], index=adv.columns).dropna()
        if len(row) < RANK_LO + MIN_NAMES:
            continue
        ranked = row.sort_values(ascending=False)
        band = ranked.iloc[RANK_LO - 1:RANK_HI].index  # ranks RANK_LO..RANK_HI
        mask.iloc[i, mask.columns.get_indexer(band)] = True
    return mask


def main() -> None:
    base = os.path.join(pdat.CACHE, f"panels_{START}_{END}")
    panels = {n: pd.read_parquet(f"{base}__{n}.parquet")
              for n in ("price", "dollar_volume", "funding")}
    vol, funding = panels["dollar_volume"], panels["funding"]
    print(f"[data] {vol.shape[1]} symbols x {vol.shape[0]} days "
          f"({vol.index.min().date()} -> {vol.index.max().date()})")

    uni = tail_universe(vol)
    depth = uni.sum(axis=1)
    active = depth[depth >= MIN_NAMES]
    print("\n=== UNIVERSE DEPTH (tail = ADV ranks "
          f"{RANK_LO}-{RANK_HI}, need >={MIN_NAMES} to rank) ===")
    print(f"  first date with >= {MIN_NAMES} tail names: "
          f"{active.index.min().date() if len(active) else 'NEVER'}")
    print(f"  usable days (>= {MIN_NAMES} names): {len(active)} "
          f"of {len(depth)}  ->  ~{len(active)/252:.1f} yrs")
    print(f"  median tail names/day (active span): {int(active.median()) if len(active) else 0}")

    # DSR hurdle implied by this sample length (benign skew, N=10), from
    # scripts/graduation_hurdle.py: req net SR ~ 1.5745*sqrt(252/n_obs)/... ;
    # use the precomputed anchors instead of re-deriving.
    n_obs = len(active)
    # interpolate the hurdle from the known table (symmetric-skew column, N=10)
    anchors = [(504, 2.28), (1008, 1.61), (1512, 1.32), (2342, 1.06), (3378, 0.88)]
    if n_obs <= anchors[0][0]:
        hurdle = anchors[0][1]
    elif n_obs >= anchors[-1][0]:
        hurdle = anchors[-1][1]
    else:
        hurdle = np.interp(n_obs, [a[0] for a in anchors], [a[1] for a in anchors])
    print(f"  => implied DSR>=0.95 hurdle at N=10 (benign skew): "
          f"net SR ~ {hurdle:.2f}; tail carry's skew is WORSE than H2's "
          "-1.87, so add ~3-5%.")

    # --- Gross funding spread available (the carry effect size) ---
    sig = funding.rolling(SIG_LOOKBACK, min_periods=SIG_LOOKBACK).mean().where(uni)
    spreads = []
    for d in sig.index:
        row = sig.loc[d].dropna()
        if len(row) < MIN_NAMES:
            continue
        n_side = int(len(row) * QUANTILE)
        if n_side < 2:
            continue
        top = row.nlargest(n_side).mean()      # high funding (we SHORT, collect)
        bot = row.nsmallest(n_side).mean()     # low/neg funding (we LONG, collect)
        spreads.append(top - bot)              # daily carry harvested per unit
    spr = pd.Series(spreads)
    daily_spread = spr.mean()
    ann_spread = daily_spread * FUNDING_PERIODS_PER_YEAR
    print("\n=== GROSS FUNDING SPREAD (top-qtile minus bottom-qtile signal) ===")
    print(f"  mean daily spread: {daily_spread*1e4:.2f} bps/day")
    print(f"  annualized gross carry: {ann_spread*100:.1f}%/yr "
          f"(median {spr.median()*FUNDING_PERIODS_PER_YEAR*100:.1f}%, "
          f"p25 {spr.quantile(.25)*FUNDING_PERIODS_PER_YEAR*100:.1f}%)")

    # --- Cost wall (tail spreads are wide; pre-declare candidates) ---
    # H2 measured turnover ~35x/yr at weekly top-30. Tail cadence similar.
    ann_turnover = 35.0
    print("\n=== COST WALL (annual, at ~35x/yr turnover like H2) ===")
    for cps in (7.0, 10.0, 15.0, 20.0):
        cost = ann_turnover * cps / 1e4
        ratio = ann_spread / cost if cost else float("inf")
        print(f"  {cps:>4.0f} bps/side -> {cost*100:>5.2f}%/yr cost  | "
              f"gross/cost = {ratio:>4.1f}x")

    print("\n=== POWER VERDICT (descriptive; the trial is the real test) ===")
    cost15 = ann_turnover * 15.0 / 1e4
    net_carry_15 = ann_spread - cost15
    print(f"  gross carry {ann_spread*100:.1f}%/yr vs 15bps cost "
          f"{cost15*100:.1f}%/yr -> net gross-of-price-drift "
          f"{net_carry_15*100:.1f}%/yr")
    print("  NOTE: this is funding only; realized P&L also eats the price "
          "drift that offsets funding and the crash skew. A positive number "
          "here is necessary, NOT sufficient -- it justifies registering and "
          "running the trial, not claiming an edge.")


if __name__ == "__main__":
    main()
