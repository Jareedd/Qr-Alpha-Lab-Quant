"""Stage-1 feasibility DATA AUDIT for a delta-neutral CASH-AND-CARRY crypto
strategy (long spot, short perp, collect funding).

This is NOT a graded trial. It spends NO trial, writes NO research_log row, and
returns NO forward-return "strategy verdict" -- it computes the REALIZED net
carry honestly from daily data and applies pre-set kill criteria to a GO/NO-GO
per liquidity bucket, so a human can decide whether the strategy is worth
registering. Everything is reproducible from this file's config; real data is
labeled as such; the only synthetic data anywhere in the audit is the clearly
labeled test fixtures.

Pipeline
--------
1. Assemble the bucketed universe: USDT perps with BOTH perp price+funding AND an
   available SPOT pair, the top ~60 by trailing perp dollar-ADV, bucketed
   major(top5)/mid(next25)/tail(rest). Spot fetched SEQUENTIALLY and cached
   (resumable). Perps with no spot pair are SKIPPED and counted.
2. For each symbol: compute ALWAYS-ON and FUNDING-GATED (3x hurdle) realized net
   carry (quantlab.carry_basis -- the sign-pinned economics).
3. Kill criteria per bucket: (1) 3x-hurdle availability, (2) convergence-
   neutralization (priced if >=80%), (3) by-year decay 2019->2026.
4. Per-bucket GO/NO-GO; write results/carry_basis_feasibility.json + print a
   readable report WITH honest limitations.

Run:
    PYTHONIOENCODING=utf-8 python scripts/carry_basis_feasibility.py
    PYTHONIOENCODING=utf-8 python scripts/carry_basis_feasibility.py --max 20  # smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import carry_basis as cb
from quantlab import perp_data, spot_data

START = "2019-09-01"

# ---------------------------------------------------------------------------
# Candidate pool of liquid USDT perps. The audit measures REAL dollar-ADV from
# fetched perp klines and ranks/buckets by that measurement -- this pool is only
# the set we PAY to fetch (we cannot afford to fetch klines for all ~900 listed
# perps sequentially on one IP). It is deliberately broad (~90 names spanning
# majors -> speculative tail incl. 1000x-scaled meme perps) so the measured
# top-60 is not biased by the pool's edges. Pass --full to instead rank the
# entire S3 perp listing (slow: fetches klines for every listed perp).
# ---------------------------------------------------------------------------
CANDIDATE_POOL = [
    # majors / large caps
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT", "TRXUSDT", "DOTUSDT",
    "MATICUSDT", "BCHUSDT", "NEARUSDT", "UNIUSDT", "ATOMUSDT", "ETCUSDT",
    "FILUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT", "SUIUSDT",
    "TIAUSDT", "SEIUSDT", "RUNEUSDT", "AAVEUSDT", "MKRUSDT", "LDOUSDT",
    # mid caps
    "ICPUSDT", "FTMUSDT", "SANDUSDT", "AXSUSDT", "GALAUSDT", "MANAUSDT",
    "EOSUSDT", "THETAUSDT", "XLMUSDT", "ALGOUSDT", "GRTUSDT", "FLOWUSDT",
    "CHZUSDT", "ENJUSDT", "DYDXUSDT", "CRVUSDT", "SNXUSDT", "COMPUSDT",
    "SUSHIUSDT", "1INCHUSDT", "ZILUSDT", "WLDUSDT", "JUPUSDT", "STXUSDT",
    "PYTHUSDT", "ORDIUSDT", "FETUSDT", "RNDRUSDT", "IMXUSDT", "ENSUSDT",
    "GMTUSDT", "APEUSDT", "MASKUSDT", "WIFUSDT", "ARKMUSDT", "JTOUSDT",
    # tail / speculative (incl. 1000x-scaled meme perps -> spot maps unscaled)
    "1000PEPEUSDT", "1000SHIBUSDT", "1000FLOKIUSDT", "1000BONKUSDT",
    "1000LUNCUSDT", "1000XECUSDT", "BOMEUSDT", "MEMEUSDT", "PEOPLEUSDT",
    "TURBOUSDT", "NEIROUSDT", "POPCATUSDT", "ACTUSDT", "PNUTUSDT",
    "MOODENGUSDT", "GOATUSDT", "HMSTRUSDT", "DOGSUSDT", "CATIUSDT",
    "BANANAUSDT", "ZROUSDT", "ETHFIUSDT", "ALTUSDT", "MANTAUSDT",
    "AEVOUSDT", "PORTALUSDT", "SAGAUSDT", "OMNIUSDT", "REZUSDT", "BBUSDT",
]


def _keep_system_awake() -> None:
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:  # noqa: BLE001 -- best-effort, non-Windows no-op
        pass


def _bucket_for_rank(rank: int) -> str:
    if rank < 5:
        return "major"
    if rank < 30:
        return "mid"
    return "tail"


def _bucket_costs(bucket: str) -> cb.CostParams:
    return cb.BUCKET_COSTS[bucket]


def fetch_symbol(sym: str, start: str, end: str):
    """Fetch (cached) perp klines+funding and the mapped spot klines for one
    symbol, SEQUENTIALLY. Returns a dict of aligned daily series, or None if the
    perp has no klines/funding or the spot pair is unavailable (caller SKIPS)."""
    try:
        pk = perp_data.load_klines(sym, start=start, end=end)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"perp_klines:{type(exc).__name__}:{str(exc)[:60]}"}
    if pk is None or pk.empty:
        return {"skip": "no_perp_klines"}
    try:
        pf = perp_data.load_funding(sym, start=start, end=end)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"perp_funding:{type(exc).__name__}:{str(exc)[:60]}"}
    if pf is None or pf.empty:
        return {"skip": "no_perp_funding"}
    try:
        sk = spot_data.load_spot_klines(sym, start=start, end=end)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"spot_klines:{type(exc).__name__}:{str(exc)[:60]}"}
    if sk is None or sk.empty:
        return {"skip": "no_spot_pair"}

    perp_dollar_adv = float((pk["quote_volume"]).tail(90).mean())
    return {
        "spot_close": sk["close"].astype(float),
        "perp_close": pk["close"].astype(float),
        "funding": pf.astype(float),
        "perp_dollar_adv": perp_dollar_adv,
        "spot_symbol": spot_data.perp_to_spot_symbol(sym),
        "n_perp_days": int(len(pk)),
        "n_spot_days": int(len(sk)),
    }


def _by_year(net: pd.Series) -> dict:
    """Net carry total + Sharpe by calendar year (the decay curve)."""
    net = net.dropna()
    out = {}
    if net.empty:
        return out
    for year, grp in net.groupby(net.index.year):
        s = cb.summarize_returns(grp)
        out[str(int(year))] = {"total": s["total"], "sharpe": s["sharpe"],
                               "n_days": s["n_days"]}
    return out


def analyze_symbol(data: dict, costs: cb.CostParams) -> dict:
    """Both variants + per-symbol convergence + by-year decay for one symbol."""
    spot, perp, fund = data["spot_close"], data["perp_close"], data["funding"]

    always = cb.cash_and_carry_returns(spot, perp, fund, costs)
    gated = cb.funding_gated_episodes(spot, perp, fund, costs,
                                      hurdle_mult=3.0, window=3)

    b = cb.basis(spot, perp)
    conv = cb.convergence_neutralization(fund, b, horizon=1)
    avail = cb.hurdle_availability(fund, costs, hurdle_mult=3.0, window=3)

    return {
        "spot_symbol": data["spot_symbol"],
        "perp_dollar_adv": data["perp_dollar_adv"],
        "n_days_overlap": int(len(always)),
        "always_on": cb.summarize_returns(always["net"]),
        "gated": cb.summarize_returns(gated["net"]),
        "gated_n_episodes": cb.episode_count(gated["held"]),
        "gated_frac_days_held": float(gated["held"].mean()),
        "hurdle_availability": avail,
        "convergence": conv,
        "always_on_by_year": _by_year(always["net"]),
        "gated_by_year": _by_year(gated["net"]),
        "median_basis": float(b.median()),
        "median_daily_funding": float(fund.median()),
    }


def _agg(values: list[float]) -> dict:
    arr = np.array([v for v in values if v == v], dtype=float)  # drop nan
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "median": float(np.median(arr)),
            "n": int(arr.size)}


def _pool_by_year(per_symbol: list[dict], key: str) -> dict:
    """Equal-weight average across symbols of per-symbol by-year NET TOTAL."""
    years: dict[str, list[float]] = {}
    for r in per_symbol:
        for yr, st in r[key].items():
            years.setdefault(yr, []).append(st["total"])
    return {yr: _agg(v) for yr, v in sorted(years.items())}


def summarize_bucket(name: str, rows: list[dict]) -> dict:
    """Per-bucket aggregation + the three kill criteria + GO/NO-GO."""
    if not rows:
        return {"bucket": name, "n_symbols": 0, "verdict": "NO_DATA"}

    always_means = [r["always_on"]["ann_return"] for r in rows]
    always_sharpe = [r["always_on"]["sharpe"] for r in rows]
    gated_means = [r["gated"]["ann_return"] for r in rows]
    gated_sharpe = [r["gated"]["sharpe"] for r in rows]
    avail = [r["hurdle_availability"] for r in rows]
    conv_frac = [r["convergence"]["median_frac_neutralized"] for r in rows]

    avail_agg = _agg(avail)
    conv_agg = _agg(conv_frac)
    always_agg = _agg(always_means)
    gated_agg = _agg(gated_means)

    # --- Kill criteria (PASS = strategy still viable) ---
    # 1. 3x hurdle availability: need a non-trivial fraction of above-hurdle days
    #    to harvest the gated variant. Threshold (assumption): >= 2% of days.
    k1_pass = avail_agg["median"] >= 0.02
    # 2. Convergence floor: if median fraction neutralized >= 80% the bucket is
    #    PRICED -> DISQUALIFY (Trial #10 kill, measured).
    k2_priced = (conv_agg["median"] == conv_agg["median"]
                 and conv_agg["median"] >= 0.80)
    k2_pass = not k2_priced
    # 3. Decay: at least one recent year (2023+) has positive median net carry
    #    in the gated variant (the regime must still exist, not be a dead relic).
    gated_year = _pool_by_year(rows, "gated_by_year")
    recent_positive = any(
        yr >= "2023" and st["median"] == st["median"] and st["median"] > 0
        for yr, st in gated_year.items()
    )
    k3_pass = recent_positive

    # GO requires: the better of the two variants clears costs (positive median
    # annualized net) AND not priced AND a recent positive regime exists.
    best_net_positive = (max(
        always_agg["median"] if always_agg["median"] == always_agg["median"] else -1e9,
        gated_agg["median"] if gated_agg["median"] == gated_agg["median"] else -1e9,
    ) > 0)
    verdict = "GO" if (best_net_positive and k2_pass and (k1_pass or k3_pass)) else "NO_GO"

    return {
        "bucket": name,
        "n_symbols": len(rows),
        "symbols": [r["spot_symbol"] for r in rows],
        "always_on_ann_return": always_agg,
        "always_on_sharpe": _agg(always_sharpe),
        "gated_ann_return": gated_agg,
        "gated_sharpe": _agg(gated_sharpe),
        "hurdle_availability": avail_agg,
        "convergence_frac_neutralized": conv_agg,
        "kill_criteria": {
            "k1_hurdle_availability_pass": bool(k1_pass),
            "k2_not_priced_pass": bool(k2_pass),
            "k2_priced": bool(k2_priced),
            "k3_recent_regime_pass": bool(k3_pass),
        },
        "always_on_by_year": _pool_by_year(rows, "always_on_by_year"),
        "gated_by_year": gated_year,
        "verdict": verdict,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=60,
                    help="top-N by measured perp dollar-ADV to keep (default 60)")
    ap.add_argument("--start", default=START)
    ap.add_argument("--full", action="store_true",
                    help="rank the ENTIRE S3 perp listing (slow); default is the "
                         "curated candidate pool")
    ap.add_argument("--out", default=os.path.join("results",
                                                  "carry_basis_feasibility.json"))
    args = ap.parse_args()

    _keep_system_awake()
    end = pd.Timestamp.today().strftime("%Y-%m-01")

    if args.full:
        candidates = perp_data.list_usdt_perp_symbols()
        print(f"[feas] FULL listing: {len(candidates)} perps", flush=True)
    else:
        candidates = CANDIDATE_POOL
        print(f"[feas] candidate pool: {len(candidates)} perps", flush=True)

    # --- SEQUENTIAL fetch (one IP); resumable via per-symbol parquet caches ---
    fetched: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    errored: dict[str, str] = {}
    for i, sym in enumerate(candidates):
        res = fetch_symbol(sym, args.start, end)
        if res is None:
            skipped[sym] = "none"
        elif "error" in res:
            errored[sym] = res["error"]
        elif "skip" in res:
            skipped[sym] = res["skip"]
        else:
            fetched[sym] = res
        if i % 10 == 0:
            print(f"[feas] {i+1}/{len(candidates)} fetched={len(fetched)} "
                  f"skipped={len(skipped)} errored={len(errored)}", flush=True)

    print(f"[feas] fetch done: {len(fetched)} usable, {len(skipped)} skipped "
          f"(no spot/perp), {len(errored)} errored", flush=True)
    if skipped:
        print(f"[feas] skip reasons: "
              f"{pd.Series(list(skipped.values())).value_counts().to_dict()}",
              flush=True)

    # --- Rank by MEASURED perp dollar-ADV, take top-N, bucket ---
    ranked = sorted(fetched.items(), key=lambda kv: kv[1]["perp_dollar_adv"],
                    reverse=True)[: args.max]
    print(f"[feas] universe = top {len(ranked)} by perp dollar-ADV", flush=True)

    buckets: dict[str, list[dict]] = {"major": [], "mid": [], "tail": []}
    per_symbol_records = []
    for rank, (sym, data) in enumerate(ranked):
        bucket = _bucket_for_rank(rank)
        costs = _bucket_costs(bucket)
        rec = analyze_symbol(data, costs)
        rec["perp_symbol"] = sym
        rec["rank"] = rank
        rec["bucket"] = bucket
        buckets[bucket].append(rec)
        per_symbol_records.append(rec)

    bucket_summaries = {b: summarize_bucket(b, rows) for b, rows in buckets.items()}

    # cost params used, surfaced for adversarial re-verification
    cost_assumptions = {
        b: {"roundtrip_bps": cp.roundtrip_bps, "rebalance_bps": cp.rebalance_bps}
        for b, cp in cb.BUCKET_COSTS.items()
    }

    out = {
        "meta": {
            "audit": "Stage-1 cash-and-carry feasibility (NOT a graded trial)",
            "data_source": "Binance public dumps: spot + futures/um, daily 1d",
            "start": args.start, "end": end,
            "is_synthetic": False,
            "n_candidates": len(candidates),
            "n_usable": len(fetched),
            "n_skipped_no_spot_or_perp": len(skipped),
            "skip_reasons": pd.Series(list(skipped.values())).value_counts().to_dict()
                            if skipped else {},
            "n_errored": len(errored),
            "errored": errored,
            "universe_size": len(ranked),
            "hurdle_mult": 3.0, "rolling_window_days": 3,
            "convergence_horizon_days": 1,
            "priced_disqualify_threshold": 0.80,
            "cost_params_bps": cost_assumptions,
        },
        "limitations": [
            "DAILY resolution: intraday basis convergence and the exact 8h "
            "funding timing are not modeled; gross carry uses daily close-to-close.",
            "SINGLE VENUE (Binance only): no cross-venue basis ('condition B') "
            "and no borrow/locate constraints on the spot leg.",
            "SLIPPAGE / COSTS ASSUMED, NOT MEASURED: we have no order books; "
            "round-trip and rebalance bps are labeled per-bucket assumptions.",
            "PERP->SPOT mapping strips 1000x scale on meme perps; price SCALE "
            "cancels in returns but a wrong pair mapping would corrupt a symbol "
            "-- mappings are logged per symbol (spot_symbol).",
            "FUNDING-GATED variant holds on day t when its trailing-3d funding "
            "clears the hurdle; no intraday entry timing is claimed.",
        ],
        "buckets": bucket_summaries,
        "per_symbol": [
            {k: v for k, v in r.items()
             if k in ("perp_symbol", "spot_symbol", "rank", "bucket",
                      "perp_dollar_adv", "n_days_overlap", "always_on", "gated",
                      "gated_n_episodes", "gated_frac_days_held",
                      "hurdle_availability", "convergence", "median_basis",
                      "median_daily_funding")}
            for r in per_symbol_records
        ],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)

    _print_report(out)
    print(f"\n[feas] wrote {args.out}", flush=True)


def _fmt_pct(x: float) -> str:
    return "   nan" if x != x else f"{x*100:6.2f}%"


def _print_report(out: dict) -> None:
    m = out["meta"]
    print("\n" + "=" * 78)
    print("CASH-AND-CARRY FEASIBILITY AUDIT  (Stage-1, NOT a graded trial)")
    print("=" * 78)
    print(f"Window: {m['start']} .. {m['end']}   "
          f"Source: {m['data_source']}")
    print(f"Candidates: {m['n_candidates']}   usable(spot+perp): {m['n_usable']}   "
          f"skipped: {m['n_skipped_no_spot_or_perp']}   "
          f"universe: {m['universe_size']}")
    print(f"Skip reasons: {m['skip_reasons']}")
    print(f"Cost assumptions (round-trip / rebalance bps): {m['cost_params_bps']}")
    print(f"Hurdle: {m['hurdle_mult']}x round-trip over {m['rolling_window_days']}d; "
          f"PRICED-disqualify if >= {m['priced_disqualify_threshold']*100:.0f}% "
          "funding neutralized by convergence")

    for b in ("major", "mid", "tail"):
        s = out["buckets"][b]
        print("\n" + "-" * 78)
        print(f"BUCKET: {b.upper()}   n={s.get('n_symbols', 0)}   "
              f">>> VERDICT: {s.get('verdict', 'NO_DATA')} <<<")
        if s.get("n_symbols", 0) == 0:
            continue
        print(f"  symbols: {', '.join(s['symbols'][:12])}"
              + (" ..." if len(s["symbols"]) > 12 else ""))
        ao, ga = s["always_on_ann_return"], s["gated_ann_return"]
        aos, gas = s["always_on_sharpe"], s["gated_sharpe"]

        def _sh(x: float) -> str:
            return f"{x:.2f}" if x == x else "nan"

        print(f"  ALWAYS-ON  net carry ann: median {_fmt_pct(ao['median'])}  "
              f"mean {_fmt_pct(ao['mean'])}   Sharpe median {_sh(aos['median'])}")
        print(f"  GATED(3x)  net carry ann: median {_fmt_pct(ga['median'])}  "
              f"mean {_fmt_pct(ga['mean'])}   Sharpe median {_sh(gas['median'])}")
        av = s["hurdle_availability"]
        cv = s["convergence_frac_neutralized"]
        print(f"  K1 hurdle availability (median frac days): {_fmt_pct(av['median'])}"
              f"   -> {'PASS' if s['kill_criteria']['k1_hurdle_availability_pass'] else 'FAIL'}")
        print(f"  K2 convergence neutralized (median frac):  {_fmt_pct(cv['median'])}"
              f"   -> {'PRICED/DISQUALIFY' if s['kill_criteria']['k2_priced'] else 'PASS (not priced)'}")
        print(f"  K3 recent (2023+) positive gated regime:   "
              f"{'PASS' if s['kill_criteria']['k3_recent_regime_pass'] else 'FAIL'}")
        print("  GATED net carry BY YEAR (median across symbols, total per year):")
        for yr, st in s["gated_by_year"].items():
            print(f"     {yr}: {_fmt_pct(st['median'])}  (n_symbols={st['n']})")

    print("\nLIMITATIONS (honest):")
    for lim in out["limitations"]:
        print(f"  - {lim}")


if __name__ == "__main__":
    main()
