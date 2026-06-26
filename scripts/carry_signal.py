"""Crypto cash-and-carry DEPLOY signal — the "dormant-but-armed" decision-support
tool for the real-money carry vehicle. NOT an executor and NOT a trial.

The Stage-1 audit (research_log 2026-06-25; `results/carry_basis_feasibility.json`)
established that delta-neutral cash-and-carry is a REAL funding harvest whose basis
P&L nets to ~0 — but whose premium over the risk-free rate has decayed to ~0 in the
current regime, with a headline Sharpe that is blind to the tail (funding-flip /
liquidation / exchange-failure / de-peg). The honest conclusion was: keep the
vehicle DORMANT and only deploy when funding pays a real premium OVER cash.

This script is that gate, run on CURRENT data. For a liquid USDT-perp universe it
pulls recent funding (no spot needed — the deploy decision is funding-vs-hurdle,
and the audit already showed basis ≈ 0 for the real harvest), and via
`carry_basis.deploy_signal` flags each symbol DEPLOY only if its trailing-window
annualized NET carry clears BOTH the 3x round-trip cost gate AND the risk-free rate
by a tail buffer. It prints a deploy plan with SMALL, hard-capped, equal-weight
sizing — deliberately NOT Sharpe/Kelly-sized, because that Sharpe is the number the
tail makes a lie.

*** DECISION-SUPPORT ONLY. It places no orders. The operator executes manually,
delta-neutral (long spot / short perp, equal notional), and owns the margin,
liquidation, and exchange/counterparty risk. ***

In the 2025–26 regime this will almost certainly print FLAT — which is the correct
dormant state. It exists to wake up when a high-funding regime returns.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from quantlab import carry_basis as cb
from quantlab import perp_data

# Liquid USDT-perp universe (funding-keyed; perp tickers). Majors get the cheap
# cost bucket, the rest the mid bucket — matching the audit's cost assumptions.
MAJORS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT"]
MID = ["ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT", "NEARUSDT",
       "APTUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT", "FILUSDT", "ATOMUSDT",
       "UNIUSDT", "AAVEUSDT", "INJUSDT", "TIAUSDT", "SEIUSDT", "WLDUSDT"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=30,
                    help="trailing days of funding for the carry estimate")
    ap.add_argument("--risk-free", type=float, default=0.05,
                    help="annual risk-free rate (T-bill) the carry must beat")
    ap.add_argument("--tail-buffer", type=float, default=0.10,
                    help="required annual NET carry premium OVER risk-free to "
                         "justify the tail risk (deploy hurdle = rf + buffer)")
    ap.add_argument("--max-gross", type=float, default=0.50,
                    help="cap on total carry book as a fraction of carry capital")
    ap.add_argument("--per-name-cap", type=float, default=0.10,
                    help="cap on any single delta-neutral position")
    ap.add_argument("--start", default=None,
                    help="funding history start (default: ~6 months back)")
    args = ap.parse_args()

    start = args.start or (pd.Timestamp.today() - pd.Timedelta(days=200)
                           ).strftime("%Y-%m-%d")
    universe = MAJORS + MID
    costs_by_symbol = {s: cb.BUCKET_COSTS["major" if s in MAJORS else "mid"]
                       for s in universe}

    print(f"[carry-signal] funding since {start}; hurdle = risk_free "
          f"{args.risk_free:.0%} + tail buffer {args.tail_buffer:.0%} = "
          f"{args.risk_free + args.tail_buffer:.0%} net annual carry; "
          f"trailing window {args.window}d.")
    funding_by_symbol = {}
    for s in universe:
        try:
            f = perp_data.load_funding(s, start)
        except Exception as exc:  # noqa: BLE001 — one bad symbol must not abort
            print(f"  {s:<10} funding fetch failed ({type(exc).__name__}); skip")
            continue
        if f is not None and len(f):
            funding_by_symbol[s] = f

    sig = cb.deploy_signal(funding_by_symbol, costs_by_symbol,
                           risk_free=args.risk_free, tail_buffer=args.tail_buffer,
                           window=args.window)

    rows = sorted(sig.items(),
                  key=lambda kv: (kv[1].get("net_ann") or -1e9), reverse=True)
    print(f"\n{'symbol':<10}{'gross%':>9}{'net%':>9}{'excess_RF%':>12}"
          f"{'cost_gate':>11}{'DEPLOY':>8}")
    for s, r in rows:
        if r.get("reason") == "insufficient_history":
            print(f"{s:<10}{'—':>9}{'—':>9}{'—':>12}{'—':>11}{'(no hist)':>10}")
            continue
        print(f"{s:<10}{r['gross_ann']*100:>9.1f}{r['net_ann']*100:>9.1f}"
              f"{r['excess_over_rf']*100:>12.1f}"
              f"{('yes' if r['cost_gate'] else 'no'):>11}"
              f"{('YES' if r['deploy'] else '·'):>8}")

    deploy = [s for s, r in sig.items() if r.get("deploy")]
    print("\n" + "=" * 60)
    if not deploy:
        print("VERDICT: FLAT — nothing clears the hurdle. The carry vehicle stays "
              "DORMANT (correct for a low-funding regime). Re-run when funding "
              "spikes; it is armed for the next high-funding regime.")
    else:
        n = len(deploy)
        per = min(args.max_gross / n, args.per_name_cap)
        print(f"VERDICT: DEPLOY {n} name(s), delta-neutral (long spot / short "
              f"perp, equal notional), EACH sized at {per:.0%} of carry capital "
              f"(total gross {per*n:.0%}, capped):")
        for s in deploy:
            print(f"   {s}: net carry ~{sig[s]['net_ann']*100:.1f}%/yr "
                  f"(excess over RF ~{sig[s]['excess_over_rf']*100:.1f}%)")
    print("=" * 60)
    print("CAVEATS (honest): decision-support only — places NO orders. Sizing is "
          "small/equal-weight/capped and deliberately NOT Sharpe-Kelly (the carry "
          "Sharpe is tail-blind). You bear margin, liquidation, exchange/"
          "counterparty, and stablecoin-de-peg risk; the smooth carry can reverse "
          "violently. Funding is single-venue (Binance); slippage is not modeled.")


if __name__ == "__main__":
    main()
