"""H10 Stage-1.5 coverage / POWER probe — is trial #13 even powered on free data?

ZERO trials, N unchanged. This is a DATA AUDIT, not a graded run: it computes NO
forward returns and reaches NO verdict on the hypothesis. It answers ONE question
before we commit to the hour-plus full-universe Form 4 fetch a graded run needs:

    On the survivorship-safe PIT S&P 500 universe, are opportunistic insider
    CLUSTER buys (>= k=2 distinct opportunistic buyers in a trailing 90d window)
    dense enough that the frozen POWER GATE (n_obs >= 60 months AND median per-date
    long-basket >= 5 names) could pass?

Method (cheap by construction): fetch real Form 4 buys/sells for a SEEDED RANDOM
SAMPLE of the universe (default 30 names), build the EXACT frozen cluster mask via
``insider.net_cluster_buy_signal`` (the same code the graded harness uses), count
cluster-eligible names per month IN THE SAMPLE, and LINEARLY EXTRAPOLATE to the
full ~500-name universe. The top-decile long basket on the full universe is
estimated as ``0.10 * (full-universe cluster-eligible per month)``.

HONESTY CAVEATS, declared up front and printed with the result:
  * A random sample is an UNBIASED density estimate, but the documented edge
    concentrates in SMALL-caps — within the (all-large-cap) S&P 500 the smaller
    members skew the true density slightly ABOVE a uniform extrapolation, so a
    "fails the floor by a wide margin" read is robust while a "just clears it" read
    is not. The probe reports the margin, not a binary.
  * Linear extrapolation assumes the sample's per-month eligibility rate is
    representative; sampling error is ~1/sqrt(n_sample) and is stated.
  * This predicts the POWER GATE only. It says NOTHING about whether the signal
    has alpha — that is the graded trial's job, post-sign-off.

Reuses the graded harness's survivorship-safe assembly (``run_h10_trial``); it does
not re-implement CIK resolution, so the probe and the trial measure the same thing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))            # import the harness module

import numpy as np
import pandas as pd

from quantlab import insider
from quantlab.insider_data import InsiderSource
from quantlab.sec_xwalk_source import SurvivorshipSafeSECSource
import run_h10_trial as h10                              # frozen constants + assembly

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results",
                       "h10_coverage_probe.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=30,
                    help="number of universe names to sample (cheap-probe size)")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    source = SurvivorshipSafeSECSource()
    isrc = InsiderSource()
    members = source.universe()
    n_universe = len(members)
    rng = np.random.default_rng(args.seed)
    k = min(args.sample, n_universe)
    sample = sorted(rng.choice(np.array(members, dtype=object), size=k, replace=False))
    print(f"[universe] {n_universe} PIT members; probing a seeded random sample of "
          f"{k} (seed {args.seed}).")
    print(f"[sample] {', '.join(sample)}")

    # --- fetch real Form 4 buys/sells for the sample (rate-limited, cached) ----- #
    buy_frames, sell_frames, n_with_buys, n_failed = [], [], 0, 0
    for i, tkr in enumerate(sample, 1):
        cik = source._cik_for(tkr)
        if cik is None:
            print(f"  [{i:>2}/{k}] {tkr:<6} -> no CIK (unresolved); skip")
            continue
        # Resilient: a name whose fetch still errors after _get's retries is SKIPPED
        # (logged), never aborts the probe. The cache makes a later re-run fill it in.
        try:
            buys = isrc.purchases(cik)
            sells = isrc.sells(cik)
        except Exception as exc:                              # noqa: BLE001
            n_failed += 1
            print(f"  [{i:>2}/{k}] {tkr:<6} CIK {cik}: FETCH FAILED ({type(exc).__name__}); "
                  "skipped (re-run resumes from cache)")
            continue
        if not buys.empty:
            buys = buys.copy(); buys["ticker"] = tkr; buy_frames.append(buys)
            n_with_buys += 1
        if not sells.empty:
            sells = sells.copy(); sells["ticker"] = tkr; sell_frames.append(sells)
        print(f"  [{i:>2}/{k}] {tkr:<6} CIK {cik}: {len(buys):>3} open-mkt buys, "
              f"{len(sells):>3} sells")

    cols = ["owner_name", "role", "shares", "value", "transaction_date",
            "ticker", "accession"]
    empty = pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="filed_date"))
    purchases = pd.concat(buy_frames).sort_index() if buy_frames else empty.copy()
    sells = pd.concat(sell_frames).sort_index() if sell_frames else empty.copy()
    purchases.index.name = sells.index.name = "filed_date"
    print(f"[fetch] {n_with_buys}/{k} sample names have >=1 open-market purchase; "
          f"{len(purchases)} total buys, {len(sells)} total sells"
          f"{f'; {n_failed} names skipped after retries' if n_failed else ''}.")

    if purchases.empty:
        print("\n[POWER PREDICTION] ZERO open-market purchases in the sample -> the "
              "cluster signal is empty. The full-universe POWER GATE would ABORT "
              "(no trial spent). Trial #13 is not worth running on free large-cap "
              "data.")
        _dump(args, n_universe, k, sample, n_with_buys, 0, 0, None, verdict="empty")
        return

    # --- frozen cluster mask over a MONTHLY grid (the harness's exact signal) --- #
    asof = pd.date_range(source.start, source.end, freq=h10.REBALANCE_FREQ)
    _, mask = insider.net_cluster_buy_signal(
        purchases, sells, asof, tickers=list(sample),
        window_days=h10.WINDOW_DAYS, sector_map=None, classify="opportunistic")
    eligible = (mask >= h10.CLUSTER_K).sum(axis=1)       # cluster-eligible names / month
    active = eligible[eligible > 0]
    median_elig_sample = float(active.median()) if not active.empty else 0.0
    p90_elig_sample = float(active.quantile(0.90)) if not active.empty else 0.0
    max_elig_sample = float(eligible.max())
    months_nonzero = int((eligible > 0).sum())
    cell_rate = float((mask >= h10.CLUSTER_K).to_numpy().mean())  # frac of (mo x name)

    scale = n_universe / k
    # Extrapolate the per-month eligible COUNT to the full universe (median over
    # months that have any cluster, the regime a graded book would actually trade).
    est_full_median_elig = median_elig_sample * scale
    est_full_basket = est_full_median_elig * h10.QUANTILE   # top-decile long basket
    # n_obs proxy: months with a non-empty FULL-universe basket ~ months the sample
    # showed any cluster (a name in the sample clustering implies the universe does);
    # this is an UNDER-count (the sample misses clusters in unsampled names), so it
    # is a conservative LOWER bound on n_obs.
    est_n_obs_lb = months_nonzero

    print("\n=== H10 POWER PROBE (sample -> full-universe extrapolation) ===")
    print(f"  sample cluster-eligible names/month: median {median_elig_sample:.2f}, "
          f"p90 {p90_elig_sample:.2f}, max {max_elig_sample:.0f} "
          f"(over {months_nonzero} months with any cluster)")
    print(f"  (mo x name) cluster-eligible cell rate: {cell_rate*100:.3f}%")
    print(f"  extrapolate x{scale:.1f} -> full-universe median eligible/month "
          f"~{est_full_median_elig:.1f} -> est. top-decile long basket "
          f"~{est_full_basket:.1f} names")
    print(f"  est. n_obs (months w/ non-empty basket), LOWER bound: ~{est_n_obs_lb}")

    # --- POWER-GATE prediction against the FROZEN floors ----------------------- #
    basket_ok = est_full_basket >= h10.MIN_BASKET
    nobs_ok = est_n_obs_lb >= h10.MIN_N_OBS
    would_pass = basket_ok and nobs_ok
    print(f"\n  frozen floors: median basket >= {h10.MIN_BASKET}, "
          f"n_obs >= {h10.MIN_N_OBS}")
    print(f"  basket floor: est ~{est_full_basket:.1f} vs {h10.MIN_BASKET}  -> "
          f"{'PLAUSIBLE' if basket_ok else 'FAILS'}")
    print(f"  n_obs floor : est >=~{est_n_obs_lb} vs {h10.MIN_N_OBS}  -> "
          f"{'PLAUSIBLE' if nobs_ok else 'FAILS (lower bound)'}")
    print(f"\n  >>> POWER PREDICTION: a full trial #13 would "
          f"{'PLAUSIBLY CLEAR' if would_pass else 'LIKELY ABORT at'} the power gate.")
    print("      (Random-sample estimate; small-caps skew density UP, so a FAILS "
          "margin is robust and a PLAUSIBLE margin is optimistic. Predicts power "
          "only, not alpha. No trial spent; N unchanged.)")

    _dump(args, n_universe, k, sample, n_with_buys, len(purchases), len(sells),
          {"median_elig_sample": median_elig_sample,
           "p90_elig_sample": p90_elig_sample, "max_elig_sample": max_elig_sample,
           "months_nonzero": months_nonzero, "cell_rate": cell_rate,
           "scale": scale, "est_full_median_elig": est_full_median_elig,
           "est_full_basket": est_full_basket, "est_n_obs_lb": est_n_obs_lb,
           "basket_ok": basket_ok, "nobs_ok": nobs_ok, "would_pass": would_pass},
          verdict=("plausible" if would_pass else "likely_abort"))


def _dump(args, n_universe, k, sample, n_with_buys, n_buys, n_sells, stats, verdict):
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    payload = {
        "probe": "h10_coverage_power", "trial_spent": False,
        "seed": args.seed, "n_universe": n_universe, "n_sample": k,
        "sample": list(sample), "n_sample_with_buys": n_with_buys,
        "n_buys": n_buys, "n_sells": n_sells, "verdict": verdict,
        "frozen_floors": {"min_basket": h10.MIN_BASKET, "min_n_obs": h10.MIN_N_OBS,
                          "quantile": h10.QUANTILE, "cluster_k": h10.CLUSTER_K,
                          "window_days": h10.WINDOW_DAYS},
        "stats": stats,
    }
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[written] {os.path.relpath(RESULTS)}")


if __name__ == "__main__":
    main()
