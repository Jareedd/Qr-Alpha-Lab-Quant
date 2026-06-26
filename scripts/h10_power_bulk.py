"""H10 trial-#13 POWER GATE — computed on the FULL PIT S&P 500 universe from the
SEC bulk insider data set (``BulkInsiderSource``). ZERO trials, N unchanged.

This replaces the slow per-company crawl probe (a 40-name SAMPLE extrapolated) with
the EXACT full-universe answer: it reads SEC's bulk quarterly Form 4 data (~64 ZIPs
instead of ~200k requests), builds the FROZEN opportunistic-cluster mask via the
same ``insider.net_cluster_buy_signal`` the graded harness uses, and reports whether
the frozen POWER GATE would pass:

    (a) n_obs  >= 60 months with a non-empty long basket, AND
    (b) median per-date long-basket size >= 5 names
    (the long basket = top DECILE of names with >= k=2 distinct opportunistic
     buyers in a trailing 90d window).

It computes NO forward returns and reaches NO verdict on the hypothesis — it only
predicts power, so a graded trial is not run and N is not touched. If the gate would
pass, trial #13 is worth running (and runs in minutes on this bulk source); if not,
H10 is logged as "underpowered on free large-cap data" (the trial-#10 fee-first
precedent) and we move on.

Faithfulness: the bulk source is cross-checked byte-for-byte against the raw-XML
crawl (``scripts/insider_bulk_crosscheck.py`` + an independent re-verification), so
the buys counted here are the same Form 4 open-market purchases the crawl would
surface. The cross-section is keyed on the issuer trading symbol carried in the bulk
SUBMISSION table; rows with no issuer symbol are dropped and counted (reported).
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))            # import the harness module

import numpy as np
import pandas as pd

from quantlab import insider, metrics
from quantlab.sec_xwalk_source import SurvivorshipSafeSECSource
from quantlab.insider_bulk import BulkInsiderSource
import run_h10_trial as h10                              # frozen constants

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results",
                       "h10_power_bulk.json")


def main() -> None:
    source = SurvivorshipSafeSECSource()
    members = source.universe()
    print(f"[universe] {len(members)} PIT members; resolving CIKs "
          "(survivorship-safe)...")
    cik_by_member, ciks = {}, set()
    for t in members:
        c = source._cik_for(t)
        if c:
            cik_by_member[t] = c
            ciks.add(c)
    print(f"[cik] resolved {len(cik_by_member)}/{len(members)} members -> "
          f"{len(ciks)} distinct issuer CIKs.")

    bulk = BulkInsiderSource(start=source.start, end=source.end)
    print(f"[bulk] reading open-market BUYS across {source.start}.."
          f"{source.end} (downloads missing quarters once, then parses)...")
    buys = bulk.transactions(sorted(ciks), kind="P")     # one pass over all quarters
    n_total = len(buys)
    n_null_tkr = int(buys["ticker"].isna().sum())
    buys = buys[buys["ticker"].notna()].copy()
    print(f"[bulk] {n_total} open-market buys for the universe CIKs "
          f"({n_null_tkr} dropped for missing issuer symbol).")

    # FROZEN monthly cluster mask (the harness's exact signal; sells are NOT needed
    # for the POWER gate — the cluster gate keys on n_opp_BUYERS only).
    asof = pd.date_range(source.start, source.end, freq=h10.REBALANCE_FREQ)
    tickers = sorted(buys["ticker"].dropna().unique().tolist())
    print(f"[signal] building opportunistic-cluster mask over {len(asof)} months x "
          f"{len(tickers)} firms...")
    _, mask = insider.net_cluster_buy_signal(
        buys, None, asof, tickers=tickers,
        window_days=h10.WINDOW_DAYS, sector_map=None, classify="opportunistic")

    eligible = (mask >= h10.CLUSTER_K).sum(axis=1)        # cluster-eligible / month

    def basket_size(n_elig: float) -> int:
        # mirrors long_vs_ew_weights: need >=2 eligible, long = top-decile (floor 1).
        if n_elig < 2:
            return 0
        return max(1, int(n_elig * h10.QUANTILE))

    baskets = eligible.apply(basket_size)
    nonempty = baskets[baskets > 0]
    n_obs = int((baskets > 0).sum())
    median_basket = float(nonempty.median()) if len(nonempty) else 0.0
    median_elig = float(eligible[eligible > 0].median()) if (eligible > 0).any() else 0.0
    p90_elig = float(eligible[eligible > 0].quantile(0.9)) if (eligible > 0).any() else 0.0
    max_elig = float(eligible.max())

    print("\n=== H10 FULL-UNIVERSE POWER (bulk Form 4) ===")
    print(f"  cluster-eligible firms/month: median {median_elig:.1f}, p90 {p90_elig:.1f}, "
          f"max {max_elig:.0f}  (over {n_obs} months with a tradeable basket)")
    print(f"  implied top-decile long basket/month: median {median_basket:.1f}")

    # MDE at the realized n_obs (N=13), annualized — for context next to the floors.
    if n_obs > 0:
        var_sr = 1.0 / n_obs
        sr_star = metrics.expected_max_sharpe(h10.N_TRIALS_DEFAULT, var_sr, n_obs)
        mde_ann = sr_star * np.sqrt(h10.PERIODS_PER_YEAR)
        print(f"  MDE @ N={h10.N_TRIALS_DEFAULT}, n_obs={n_obs}: net annual SR hurdle "
              f"~{mde_ann:.2f}")
    else:
        mde_ann = float("nan")

    basket_ok = median_basket >= h10.MIN_BASKET
    nobs_ok = n_obs >= h10.MIN_N_OBS
    would_pass = basket_ok and nobs_ok
    print(f"\n  frozen floors: median basket >= {h10.MIN_BASKET}, n_obs >= {h10.MIN_N_OBS}")
    print(f"  basket floor: {median_basket:.1f} vs {h10.MIN_BASKET}  -> "
          f"{'PASS' if basket_ok else 'FAIL'}")
    print(f"  n_obs floor : {n_obs} vs {h10.MIN_N_OBS}  -> "
          f"{'PASS' if nobs_ok else 'FAIL'}")
    print(f"\n  >>> POWER GATE would {'PASS' if would_pass else 'ABORT'} -> "
          f"{'RUN trial #13' if would_pass else 'log H10 underpowered-on-free-data'}.")
    print("      (Full universe, exact — not a sample extrapolation. Predicts power "
          "only, not alpha. No trial spent; N unchanged.)")

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump({
            "probe": "h10_power_full_universe_bulk", "trial_spent": False,
            "n_members": len(members), "n_members_with_cik": len(cik_by_member),
            "n_ciks": len(ciks), "n_buys": n_total, "n_buys_dropped_no_ticker": n_null_tkr,
            "n_firms_with_buys": len(tickers), "n_months": len(asof),
            "median_eligible_per_month": median_elig, "p90_eligible": p90_elig,
            "max_eligible": max_elig, "median_basket": median_basket,
            "n_obs_nonempty_basket": n_obs, "mde_ann": mde_ann,
            "floors": {"min_basket": h10.MIN_BASKET, "min_n_obs": h10.MIN_N_OBS},
            "basket_ok": basket_ok, "nobs_ok": nobs_ok, "would_pass": would_pass,
            "verdict": "run_trial_13" if would_pass else "underpowered_free_data",
        }, f, indent=2)
    print(f"\n[written] {os.path.relpath(RESULTS)}")


if __name__ == "__main__":
    main()
