# Research Log — qr-alpha-lab

**Global trial count (feeds `--n-trials` for the DSR): N = 0**

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

## Notes

- 2026-06-10: No real-data (yfinance) strategy run has been evaluated yet, so N = 0.
  The first real-data run of the default config will be trial #1.
- Next session (per ROADMAP weeks 1–2 → 3–4): pull real data through the pipeline,
  log trial #1, then start the point-in-time universe work to kill survivorship bias.
