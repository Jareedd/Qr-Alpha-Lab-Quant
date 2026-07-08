"""Synthetic PIT analyst-estimate generator for H13 (PEAD) scaffolding.

Mimics the SHAPE of a Bloomberg/IBES point-in-time estimate pull so the
PEADAgent logic can be exercised end-to-end BEFORE the real, license-bounded
pull lands (see writeup/bloomberg_pead_pull.md). This is SYNTHETIC data and
is labeled as such in the output filename and a provenance column — it must
never be mistaken for, or logged as, a real trial (research law #7).

Design (so the agent's reaction is testable, planted-signal style):
- SUE (Standardized Unexpected Earnings) = (actual_eps - consensus_eps_est)
  / pre_earnings_volatility. Volatility is in EPS-dollar units, so SUE is a
  dimensionless z-score of the surprise.
- Three deterministic cohorts are injected so the expected signal is known:
    BEAT  -> SUE ~ +2.5  (should trigger BULLISH,  |SUE| > 1.5)
    MISS  -> SUE ~ -2.5  (should trigger BEARISH,  |SUE| > 1.5)
    INLINE-> SUE ~ +/-0.4 (NO signal, |SUE| < 1.5) — the null cohort
- as_of_date == announcement_date: the row becomes point-in-time knowable
  only when the actual EPS is released. The consensus is a pre-announcement
  quantity but is only JOINED into an actionable row at the announcement.

Run:  python data_mocks/generate_synthetic_pead.py
Out:  data_mocks/mock_pead_data.csv
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

SEED = 13  # deterministic: same CSV every run (no Math.random surprises)
N_PER_COHORT = {"BEAT": 14, "MISS": 14, "INLINE": 12}
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mock_pead_data.csv")


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    rows = []
    ticker_i = 0
    # Announcement dates spread across an earnings season (business days).
    announce_grid = pd.bdate_range("2025-01-06", periods=60)

    for cohort, n in N_PER_COHORT.items():
        for _ in range(n):
            ticker = f"SYN{ticker_i:03d}"
            ticker_i += 1

            # Pre-earnings vol in EPS dollars: wide range so inverse-vol
            # sizing has something to bite on (0.04 .. 0.30).
            vol = float(rng.uniform(0.04, 0.30))
            consensus = float(rng.uniform(0.50, 3.00))

            if cohort == "BEAT":
                sue = float(rng.uniform(2.0, 3.2))
            elif cohort == "MISS":
                sue = float(rng.uniform(-3.2, -2.0))
            else:  # INLINE — inside the +/-1.5 band, no signal
                sue = float(rng.uniform(-0.6, 0.6))

            actual = consensus + sue * vol  # invert SUE definition

            announce = announce_grid[rng.integers(0, len(announce_grid))]
            rows.append(
                {
                    "ticker": ticker,
                    # PIT timestamp: knowable only at the announcement.
                    "as_of_date": announce.date().isoformat(),
                    "announcement_date": announce.date().isoformat(),
                    "consensus_eps_est": round(consensus, 4),
                    "actual_eps": round(actual, 4),
                    "pre_earnings_volatility": round(vol, 4),
                    "cohort": cohort,          # ground-truth label (synthetic only)
                    "data_source": "SYNTHETIC",  # provenance, per law #7
                }
            )

    df = pd.DataFrame(rows).sort_values("announcement_date").reset_index(drop=True)
    return df


def main() -> None:
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    sue = (df["actual_eps"] - df["consensus_eps_est"]) / df["pre_earnings_volatility"]
    print(f"wrote {OUT_PATH}  ({len(df)} rows)")
    print(
        "cohort SUE means: "
        + ", ".join(
            f"{c}={sue[df.cohort == c].mean():+.2f}" for c in N_PER_COHORT
        )
    )
    print(
        f"expected signals: {(sue.abs() > 1.5).sum()} names "
        f"({(sue > 1.5).sum()} bullish / {(sue < -1.5).sum()} bearish), "
        f"{(sue.abs() <= 1.5).sum()} inline (no signal)"
    )


if __name__ == "__main__":
    main()
