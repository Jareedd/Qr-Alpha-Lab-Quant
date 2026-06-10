# Research Log — qr-alpha-lab

**Global trial count (feeds `--n-trials` for the DSR): N = 1**

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

## Notes

- 2026-06-10: N = 1. Trial #1 = the default ridge config on real data, logged above.
  ridge_cv on synthetic is harness validation (no real-data evaluation), so no increment;
  the first ridge_cv run on real data WILL be trial #2.
- Known caveat to fix soon: IC t-stats ignore overlap-induced autocorrelation
  (21d labels sampled daily). Add Newey–West (lag ≈ horizon) before quoting any t-stat
  in a write-up.
- Next: point-in-time S&P 500 membership (kill survivorship bias — the EW SR of 1.00
  is the bias made visible), then Phase 3 neutralization.
