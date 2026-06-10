# Research Log — qr-alpha-lab

**Global trial count (feeds `--n-trials` for the DSR): N = 2**

Rules (from CLAUDE.md law #3): every strategy variant, hyperparameter tweak,
feature set, or horizon evaluated on **real data** gets one row and increments N.
N never resets. Synthetic planted/noise runs are harness validation, not
alpha-seeking trials — they are logged below but do not increment N.
Infrastructure changes are logged with a falsification-gate re-run (law #2).

| # | Date | Type | Hypothesis / change | Config | OOS result | Conclusion |
|---|------|------|---------------------|--------|------------|------------|
| — | 2026-06-10 | infra | Vectorize hot paths (IC, weights, walk-forward slicing) without changing results | ridge, planted/noise, defaults | planted: IC 0.0629, net SR 0.8646, DSR 0.9919 → PASS; noise: DSR 0.0004 → rejected. **All metrics byte-identical to pre-optimization baseline.** Core compute 2.25s → 0.51s (4.4×). | Optimization is behavior-preserving; pinned by 3 new equivalence tests against naive reference implementations. |
| — | 2026-06-10 | infra | Add baselines (law #5): 12-1 momentum decile L/S + equal-weight 1/N, same OOS dates, same backtester, same costs | quantile 0.1, monthly rebalance, 10 bps | On *planted* panel: momentum baseline net SR **1.19** vs ridge net SR **0.86** — the one-line baseline beats the ML model. On noise: momentum SR 0.19 (nothing, as expected). | Expected and instructive: the planted signal *is* 12-1 momentum, so ridge dilutes it across 5 features. Lesson for real data: a multi-feature model is not automatically better than the strongest single feature. Reported via `beats_mom_baseline` in every metrics JSON. |
| — | 2026-06-10 | infra | CI must enforce the falsification gate, not just run unit tests | `--fail-if-dsr-below 0.95` (planted), `--fail-if-dsr-above 0.5` (noise) in GitHub Actions | Gate passes locally with exit code 0; non-zero exit on violation verified by flag logic. | Leakage introduced by any future commit fails CI loudly. |
| — | 2026-06-10 | infra | Cache-key bug: yfinance cache keyed on `len(tickers)` → two same-size universes silently collide | `data.py` | n/a (no real-data runs yet) | Fixed: key on MD5 of sorted ticker list. Found before it could corrupt a real-data result. |

| 1 | 2026-06-10 | **trial** | Default config on real data: do the 5 standard features carry any cross-sectional signal in a liquid US universe? | yfinance 57 names (2010→now, 3 dropped by 90% coverage filter), ridge α=10, 21d horizon, decile L/S, monthly rebal, 10 bps | OOS 3126d: IC 0.0333 (t=7.77*), net SR 0.82, gross SR 0.89, turnover 3.81×/yr, maxDD −29%, DSR 0.998 @ N=1. Beats 12-1 momentum baseline (net SR 0.34); equal-weight long-only SR 1.00. | **Encouraging but NOT yet trustworthy.** (a) Universe = today's winners → survivorship bias inflates everything, esp. reversal ("buy the dip" works when every name is known to have survived); EW SR 1.0 shows the universe itself was a money printer. (b) *t-stat overstated: daily ICs of overlapping 21d labels are autocorrelated; ic.sem() assumes independence — needs Newey–West. (c) No sector/beta neutralization yet. Number goes in no write-up until PIT universe (Phase 2) + neutralization (Phase 3) are done. |
| — | 2026-06-10 | infra | Nested per-roll hyperparameter tuning must be possible without trial-count inflation | `--model ridge_cv`: inner expanding walk-forward inside each outer train window selects α from {1,10,100,1000} by mean inner IC; outer test never touched | Planted panel: ridge_cv IC 0.0626 / net SR 0.857 / DSR 0.991 vs fixed ridge IC 0.0629 / 0.865 / 0.992 — near-identical, as it should be. Gate re-passed (planted PASS, noise rejected). | Selection on training data only = in-sample model fitting, not a new trial. Leak-test pinned in test_tuning.py: corrupting all data after the train window cannot change the chosen α. 21 tests green. |

| 2 | 2026-06-10 | **trial** | Does trial #1's signal survive a survivorship-bias-aware universe? | Point-in-time S&P 500 (Wikipedia changes table walked backward): 810 members ever 2010→now, 661 priceable (81.6%), 149 dead names unpriceable. Same ridge defaults, `--n-trials 2`. | OOS 3378d: IC 0.0052 (t_NW **0.54**), gross SR 0.18, **net SR −0.01**, DSR 0.29, turnover 7.26×/yr. Momentum baseline net SR 0.03; EW (member-masked) SR 0.81. | **Trial #1's alpha was survivorship bias.** IC fell 0.033→0.005 and net Sharpe 0.82→−0.01 on the honest universe — McLean–Pontiff reproduced in-house. This is the project's strongest exhibit, not a setback. Residual bias remains (149 unpriceable dead names + missing delisting returns), quantified in `results/sp500_pit_coverage.json`. |
| — | 2026-06-10 | infra | Bug caught BY a baseline: PIT run showed equal-weight SR 3.34 — impossible (RSP ≈ 0.9), so something was broken | `pct_change()` default pad-fill fabricated frozen 0% daily returns for delisted names, crushing measured vol | After `fill_method=None` everywhere (features, backtest, baselines): EW SR 0.81 (sane); strategy results ~unchanged (net SR −0.03→−0.01), so the bug was isolated to return-series construction for dead names. Gate re-passed; regression test added (dead names must drop out of averages, never contribute phantom zeros). | A baseline so good it must be wrong is as informative as a strategy that looks too good. Law #5 paid for itself the first week it existed. |
| — | 2026-06-10 | infra | Newey–West IC t-stats (overlapping 21d labels autocorrelate daily ICs) | `metrics.newey_west_tstat`, lags = horizon, Bartlett kernel | Planted panel: t_naive 7.76 → t_NW 2.00 (≈√21 shrinkage, as theory predicts). Trial #2: t_naive 1.70 → t_NW 0.54. | Naive IC t-stats on overlapping labels overstate significance ~4×. All quoted t-stats are NW from now on. |

## Notes

- 2026-06-10: N = 2 (trial #1 biased universe, trial #2 PIT universe).
  The Newey–West caveat from trial #1 is now fixed; all t-stats quoted are NW.
- Phase 2 is functionally complete: real data, CI with falsification gate,
  PIT universe with quantified residual bias. Headline finding so far: the
  default 5-feature ridge has NO defensible edge on the honest universe.
- Next (Phase 3): sector/beta neutralization + risk report. Open questions for
  Phase 4: z-score features over members only (currently z-scored over all
  priceable names, mild contamination, no lookahead); turnover 7.3×/yr is high —
  slower signals or longer rebalance may help net results; residualized labels.
