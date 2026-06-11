# Do standard price signals survive an honest backtest? A falsification-first study on point-in-time S&P 500 data

**DRAFT SKELETON — Phase 7.** Every number below is traceable to a row in
`research_log.md` or a JSON in `results/` (law #8); sections marked TODO
fill in as live cycles mature. Target: 6–10 pages, AQR-note structure.

## Abstract (write last)

TODO. One paragraph: question, method, the headline *negative* result, the
survivorship-bias exhibit, live verification status.

## 1. Question

Do five standard price-only cross-sectional signals (12-1 and 6-1 momentum,
1-month reversal, 3-month realized vol, 52-week-high proximity) carry
exploitable alpha in large-cap US equities, once the backtest is honest:
point-in-time universe, embargoed walk-forward validation, costs, declared
trial count, and a falsification harness that must recover planted signals
and reject pure noise?

## 2. Why the bar is set where it is (literature)

- Published anomalies decay 26% OOS, 58% post-publication (McLean–Pontiff).
- Most published factors are false discoveries at honest thresholds
  (Harvey–Liu–Zhu).
- High-turnover anomalies die to costs (Novy-Marx–Velikov).
- Max-of-N backtests look great on noise; the Deflated Sharpe Ratio prices
  that in (Bailey–López de Prado). Trial count here: **N = 7, logged**.

## 3. Data and universe

- Daily prices via yfinance, 2009→2026; PIT S&P 500 membership rebuilt from
  Wikipedia's changes table: 810 members ever, 661 priceable (81.6%).
- Residual bias quantified, not waved at: 149 dead names unpriceable, no
  delisting returns (Shumway) — both push results UP, stated in §8
  (`results/sp500_pit_coverage.json`).
- The biased static-universe run is kept deliberately as a measured
  exhibit: same code, same features — IC 0.033 / net SR 0.82 (biased) vs
  IC 0.005 / net SR −0.01 (PIT). **Survivorship bias manufactured the
  entire result** (trials #1 vs #2). This is the paper's centerpiece
  exhibit, reproduced in-house.

## 4. Methodology

- Walk-forward ridge (α=10) on 5 z-scored features; embargo ≥ label
  horizon; nested per-roll tuning shown to not inflate trials (log, infra).
- Falsification gate in CI: planted signal must be recovered (DSR 0.992),
  noise must be rejected (DSR 0.078) — on every push.
- Newey–West t-stats throughout: overlapping 21d labels autocorrelate
  daily ICs; naive t overstates ~√21 ≈ 4.6× (measured: 7.76 → 2.00 on the
  planted panel).
- Costs 10 bps linear; turnover a headline metric. Sector demean +
  ex-ante-beta-zero projection (rolling 252d, past-only); realized residual
  beta measured, not asserted (mean 0.05, p95 0.23).
- Baselines (law #5): equal-weight and one-line 12-1 momentum, same OOS
  window, same costs. A baseline once caught a bug a 9-test suite missed
  (EW SR 3.34 → pad-filled phantom returns for dead names).

## 5. Results (all OOS, all net of costs, N = 7 trials declared)

| Trial | Config | IC | t_NW | net SR | DSR |
|---|---|---|---|---|---|
| 1 | biased universe, ridge | 0.0333 | (7.77 naive) | 0.82 | 0.998* |
| 2 | PIT universe, ridge | 0.0052 | 0.54 | −0.01 | 0.29 |
| 3 | + sector/beta neutral | 0.0052 | — | −0.38 | 0.01 |
| 4 | 63d horizon/rebal | −0.0278 | −1.95 | −0.35 | 0.01 |
| 5 | residual label | +0.0225 | 1.91 | −0.77 | ≈0 |
| 6 | GBR, residual label | +0.0077 | 0.80 | −0.12 | 0.04 |
| 7 | MLP, residual label | +0.0093 | 1.21 | −0.28 | 0.008 |

\* DSR at N=1 before the universe was honest — kept to show why DSR alone
doesn't save you from a biased universe.

**Verdict:** no configuration produced a defensible edge (best DSR 0.04;
no |t_NW| ≥ 2 in the right direction) across linear, tree, and shallow-net
model classes on identical features and harness.

## 6. What failed (mandatory section)

- The headline "alpha" of trial #1 was survivorship bias, in full.
- Sector/beta neutralization removed (noisy) factor *return*, not masking —
  nothing was hiding underneath (trial #3).
- Slower rebalancing fixed turnover but the IC flipped negative — and we
  explicitly declined to trade the sign-flip (max-of-N trap, logged, trial #4).
- Residual-label IC "improved" while gross P&L worsened: IC and P&L are
  different objects; we report both and trust the P&L (trial #5).
- TODO: anything Phase 6 live operation breaks or reveals.

## 7. Capacity and execution realism

TODO: square-root impact capacity sweep on the PIT config
(`results/capacity_*.json`); state the AUM at which costs eat the
(already absent) edge; one-day execution assumption.

## 8. Live verification (Phase 6, running)

- Daily CI-driven paper trading on Alpaca; complete-label training (live
  inherits the backtest's leakage discipline); predictions logged before
  orders, committed to the repo, **write-once**.
- First cycle 2026-06-10: 100-name book, 100/100 orders accepted.
- Live IC vs backtest IC once cycles mature (21 trading days); t_NW needs
  >23 matured cycles. TODO: table + `results/live/live_ic.png` after
  ~2026-07-09.
- Known asymmetry: cycle #1 logged weights only; the live-IC record is one
  cycle shorter than the trading record.

## 9. Limitations (keep synced with README)

PIT-residual bias (149 unpriceable dead names; no delisting returns);
sectors as-of-today; free daily data; linear costs in headline results;
estimated (drifting) betas; single-market, single-period.

## 10. What institutional-grade would require

Point-in-time fundamentals/GICS, delisting returns (CRSP), borrow
costs/locates, impact model calibrated to fills, multi-market replication.

## References

Bailey & López de Prado (2014); Gu, Kelly & Xiu (2020); Harvey, Liu & Zhu
(2016); Jegadeesh & Titman (1993); López de Prado (2018); McLean & Pontiff
(2016); Novy-Marx & Velikov (2016); Shumway (1997).
