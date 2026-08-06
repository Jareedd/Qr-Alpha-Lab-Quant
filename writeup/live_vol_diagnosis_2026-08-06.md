# Live vol diagnosis — why a "neutral" book runs 20% (2026-08-06)

**Question (research menu §5.0).** The live paper book is dollar-neutral,
sector-demeaned and beta-projected at construction, yet realized vol is
~20% annualized with episodic residual beta. Which of three mutually
exclusive mechanisms explains it: a construction leak (the intended book
carries beta), an execution leak (the broker book departs from the
intended book), or idiosyncratic/structural concentration?

**Method.** Read-only decomposition of the committed audit trail — 38
cycles of `results/live/weights_*.csv` and `summary_*.json`
(2026-06-10 → 2026-08-05). The INTENDED book is marked at public prices
with the monitor's own t+1 convention (`monitor.realized_book_returns`);
the BROKER stream is the committed Alpaca equity. Machinery:
`src/quantlab/live_diagnosis.py` (8 known-answer tests, incl. a pin that
the ex-ante-beta mirror reproduces `risk.beta_neutralize_weights`
exactly); runner: `scripts/live_beta_diagnosis.py`. Every number below is
in `results/live_beta_diagnosis.json`. **Zero trials spent; the frozen
live config untouched.**

## Verdict, one line each

1. **Execution leak: NO.** Cumulative broker-vs-intended gap −1.23% over
   37 cycles. The one mass order failure (2026-07-23: 49 of 95 orders
   rejected, "account is not allowed to short", the whole short-leg
   rebalance) left a **4.5 bp** mark the next day — daily delta
   rebalancing is self-healing, because one day's failed *deltas* are a
   small fraction of the standing book. Failed orders are logged and
   forgotten by design; that design survives this test.
2. **Beta leak: REAL, but secondary.** Construction neutralizes against
   the *equal-weight member-mean* market. Against **SPY** the book was
   never neutral, even ex-ante: w·β_SPY is **positive on 38 of 38
   cycles**, mean **+0.31**, p95 **+0.45**. Realized full-window SPY beta:
   intended **+0.53** (iid se 0.21), broker **+0.57** (se 0.20). But the
   market-explained variance share is only **15–19%** — hedging beta
   perfectly would cut vol from ~20.3% to ~18.7%, not to backtest levels.
3. **The main wound: the live book is not the backtested strategy.**
   The deployed config's backtest
   (`results/metrics_sp500_ridge_both_residlabel.json`) ran **5.6%** ann
   vol at **3.46×/yr** one-way turnover. The live layer rebuilds the
   decile book from fresh predictions **every cycle**: one-way turnover
   **42×/yr** (a third of the book changes per day; 278 distinct names
   held across 38 cycles), intended-book vol **20.3%** — 3.6× its own
   backtest — of which **18.7% is residual** (non-market). The worst
   stretch (2026-07-24 → 07-29) lost **−8.63%** while SPY fell −1.18%;
   the ex-ante beta at the time (+0.45) implies only **−0.53%** — the
   drawdown was concentrated residual churn, not market.

## What this means for the live experiment

- **The live-IC arm is undamaged.** IC is measured per cycle on the
  committed predictions; book cadence does not enter it. The prediction
  logs, control arm, and revision monitor remain a valid experiment.
- **The equity curve is the artifact of a cousin strategy** — the same
  signal at 12× the turnover of the backtest that motivated it. Its
  −2.18% / Sharpe −0.65 tells us little about the backtest null either
  way; its vol tells us mostly about daily decile churn.
- The 2026-07-23 mass short rejection and the KVUE not-shortable rejects
  (06-24, 07-09) cost basis-points, not the vol. Wiring
  `borrow_*.json` into a pre-trade shortability screen (§5.3) remains
  worth doing for cleanliness, zero trials.

## The registered fix (owner sign-off required to run)

**H14** (PROPOSED, frozen 2026-08-06, in
`writeup/preregistered_hypotheses.md`): a three-arm construction variant
on the SAME deployed signal — {as-live daily refresh} vs {21-day cadence}
vs {21-day cadence + at-rebalance SPY overlay hedging w·β_SPY→0} —
through the identical cost-aware backtester on identical dates. It
pre-registers pass/fail closure criteria (vol within [0.75, 1.5]× of the
deployed backtest's 5.59%; residual SPY beta |mean| ≤ 0.05 and p95 ≤ 0.15,
i.e. graduation criterion 4; net SR not degraded beyond noise) and a
machinery gate (a planted index loading must be removed by the overlay
without inventing alpha in a noise world). One trial. Note the direction:
§5.0 guessed *tighter* cadence might be needed; the data says the daily
refresh IS the anomaly — the variant matches the backtest's cadence.

Deploying any change to the LIVE arm afterwards is a separate owner
decision (directive: the live config is frozen mid-experiment; a
construction change resets what the equity record means).

## Limitations (declared)

- 37 broker-return observations; every beta carries a wide CI (iid OLS
  SEs reported, optimistic under autocorrelation). The tight evidence for
  the tilt is the ex-ante path (38/38 positive), not the realized beta.
- Intended book marked close-to-close at yfinance adjusted prices while
  the broker executes at next open — the ±40 bp/day tracking noise
  (gap ann vol 8.8%) is mostly this timing convention, near-zero-mean so
  far (−1.23% cum). The broker record stays authoritative.
- Ex-ante betas recomputed from held-names data (2025-06 →), not the
  construction's exact member-mean inputs — the like-for-like check
  (vs own market: mean |w·β| = 0.10) bounds the reconstruction error;
  the vs-SPY finding (all-positive path) is robust to it.
- Sector map is as-of-today (cached table), not PIT.
- The monitor's own 11-day mark (report.md: ann vol 17.89%) is consistent
  with the full-record 20.3% here.
