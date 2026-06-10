# qr-alpha-lab

An honest cross-sectional alpha research pipeline: data → features → walk-forward ML → cost-aware backtest → deflated evaluation.

Most retail backtests are statistical fiction. The published evidence says so: anomaly returns fall 26% out-of-sample and 58% post-publication (McLean & Pontiff, JF 2016), most published factors fail honest multiple-testing corrections (Harvey–Liu–Zhu, RFS 2016), and the maximum in-sample Sharpe across enough trials looks brilliant even on pure noise (Bailey & López de Prado, 2014). This project is built around *not fooling myself*, which is the actual job of a quantitative researcher.

## What makes this pipeline defensible

**Planted-signal / pure-noise validation.** Before trusting any result on real data, the pipeline must pass two falsification tests:

```
python scripts/run_pipeline.py --data planted --fail-if-dsr-below 0.95  # known weak signal → must be recovered
python scripts/run_pipeline.py --data noise --n-trials 20 --fail-if-dsr-above 0.5  # no signal → must be rejected
```

Both checks run in CI on every push (`.github/workflows/ci.yml`) and fail the build on violation — leakage introduced by any future commit breaks CI, not just my confidence.

Current results: planted → out-of-sample rank IC 0.063 (Newey–West t = 2.0), net Sharpe 0.86, **DSR 0.99 → recovered**. Noise → IC −0.02, **DSR 0.0004 → correctly rejected**. A pipeline that "finds alpha" in noise has leakage; one that can't find a planted signal has bugs. This pipeline passes both.

**Walk-forward validation with embargo.** Expanding-window splits only; a 21-day embargo between train and test windows prevents overlapping forward-return labels from leaking future information into training (purged CV, López de Prado ch. 7). Standard k-fold on financial panels is silently invalid — this is the single most common fatal flaw in student projects.

**Costs are first-class.** Transaction costs are charged on every unit of turnover (default 10 bps/side), and annualized turnover is a headline metric, because almost no high-turnover published anomaly survives costs (Novy-Marx & Velikov, RFS 2016).

**Deflated Sharpe Ratio with an honest trial count.** The `--n-trials` flag forces you to declare how many strategy variants you have tried in total; the DSR then benchmarks your Sharpe against the expected maximum of that many noise draws. Tracking N is the discipline; the formula is the easy part.

**No lookahead by construction, verified by test.** Weights formed at date *t* earn returns only from *t+1*. The test suite includes a same-day-return exploit test that a buggy backtester fails loudly, plus a counter-test proving genuine foresight *would* profit (a check of the check).

**Baselines first.** Every run reports the model against (a) a one-line 12-1 momentum decile long-short and (b) an equal-weight 1/N portfolio, on the same out-of-sample dates, through the same cost-aware backtester. On the planted panel the momentum baseline (net SR 1.19) *beats* the ridge model (net SR 0.86) — the planted signal is literally momentum, and the model dilutes it across five features. If ML can't beat the baseline on real data either, that gets reported, not hidden. (The baselines also caught a real bug: an "equal-weight SR of 3.3" was impossible on its face and exposed pad-filled phantom returns for delisted names.)

**Survivorship bias, measured on this very pipeline.** The same ridge config earns net Sharpe **0.82** (IC 0.033) on a static universe of today's members — and net Sharpe **−0.01** (IC 0.005, Newey–West t = 0.5) on the point-in-time S&P 500. The entire "edge" was hindsight in the universe selection. McLean & Pontiff in miniature, reproduced in-house, and the single best exhibit this project owns.

**The honest bottom line after six logged trials.** Across horizons (21/63d), labels (raw/beta-residualized), models (ridge/GBR), and neutralization, no configuration of five price-only features earns a defensible net edge on the bias-corrected universe (best DSR 0.04). The pipeline provably recovers planted signals and rejects noise — the conclusion is about the signals, not the plumbing, and it matches the published record for heavily-arbitraged large caps. A negative result you can trust is the deliverable; the trials that produced near-misses (|t_NW| ≈ 1.9) are documented in `research_log.md` along with why we refuse to trade the sign-flip.

**Capacity is a first-class question.** `--capacity` sweeps AUM through a square-root impact model (trailing dollar-ADV, point-in-time, k = 1) and reports where net Sharpe dies — because "does it scale?" is the question that separates a backtest from a business.

**Neutrality is measured, not asserted.** `--neutralize sector|beta|both` demeans predictions within GICS sector and projects rebalance weights to zero ex-ante market beta (rolling 252-day betas, past data only). Every run — neutralized or not — emits a risk report (realized rolling beta, market correlation, sector net exposures), because a "market-neutral" label without measurement is marketing. On the planted panel, neutralization cuts p95 |rolling beta| from 0.32 to 0.03 while the planted (idiosyncratic) signal survives intact — which is exactly the pair of facts that proves the projection removes factor exposure and not signal.

**Label and cadence research, counted.** `--label residual` predicts beta-residualized forward returns (past-only rolling betas — the only return a dollar-neutral book can harvest); `--horizon` and `--rebalance` trade signal speed against turnover. Every per-window univariate feature IC ships as a CSV (`feature_ics_*.csv`) with sign-consistency printed per run, because a feature whose IC flips sign across walk-forward windows is an overfitting tell regardless of its pooled value. Each configuration evaluated on real data increments the trial count in `research_log.md` — no exceptions.

**Vectorized but pinned.** The IC computation, weight construction, and walk-forward slicing are vectorized (~4.4× core speedup), and each optimized path is pinned by a test against a naive per-date reference implementation, so a future "optimization" that drifts the numbers fails the suite.

## Layout

```
src/quantlab/
  data.py        # yfinance loader with parquet cache; chunked downloads
  universe.py    # point-in-time S&P 500 membership from Wikipedia changes table
  synthetic.py   # planted-signal and pure-noise panels for falsification tests
  features.py    # cross-sectionally z-scored: 12-1 momentum, 6-1, reversal, vol, 52w-high;
                 # member-masked z-scores; optional beta-residualized labels
  validation.py  # expanding walk-forward splitter with embargo
  models.py      # Ridge baseline + gradient boosting; per-date rank IC;
                 # ridge_cv = nested per-roll alpha tuning (train window only)
  baselines.py   # 12-1 momentum decile L/S + equal-weight 1/N benchmarks
  risk.py        # sector demean, beta-neutral weight projection, risk report
  backtest.py    # dollar-neutral decile long-short, linear costs, turnover
  metrics.py     # Sharpe, max DD, PSR, Deflated Sharpe, Newey-West t-stats
  impact.py      # square-root market impact, dollar-ADV, capacity curves
  env.py         # minimal .env loader (Alpaca keys, Phase 6)
scripts/run_pipeline.py   # end-to-end CLI (incl. CI falsification-gate flags)
tests/                    # 21 tests: leakage, costs, DSR monotonicity, lookahead,
                          # baselines, vectorized-vs-naive equivalence,
                          # nested-tuning leak checks
research_log.md           # every trial ever run; owns the honest --n-trials count
.github/workflows/ci.yml  # unit tests + falsification gate on every push
```

## Quick start

```
pip install -r requirements.txt
python -m pytest tests/ -q                              # 21 tests
python scripts/run_pipeline.py --data planted           # sanity check 1
python scripts/run_pipeline.py --data noise --n-trials 20   # sanity check 2
python scripts/run_pipeline.py --data sp500 --n-trials 2     # point-in-time S&P 500 (honest universe)
python scripts/run_pipeline.py --data yfinance --model gbr --n-trials 5  # static universe (biased, for comparison)
```

Outputs land in `results/`: metrics JSON + equity-curve PNG per run.

## Known limitations (deliberate honesty)

Sector data is **as-of-today** (Wikipedia only lists sectors for current members), so departed names share an UNKNOWN bucket and reclassifications are invisible; point-in-time GICS needs paid data. The `--data sp500` mode reconstructs **point-in-time S&P 500 membership** from Wikipedia's changes table, which removes the worst of survivorship bias — but not all of it: names that died (bankruptcy, acquisition) often have no Yahoo price history, so they drop out of the backtest even when membership says they were tradable; the run emits a `sp500_pit_coverage.json` quantifying exactly how many. Delisting returns (the final, usually ugly, price move of a dying stock) are missing entirely — a known upward bias in all free-data backtests (Shumway 1997). Names delisted within the label horizon lose their final partial period (the 21-day forward label needs a t+21 price). The legacy `--data yfinance` mode (today's members, fully biased) is kept deliberately so the two can be compared — measuring the bias is more interesting than removing it. Costs are linear with no market-impact model. No risk-model neutralization (sector/beta) yet. Free daily data only. Every one of these is a roadmap item, and naming them is part of the point.

## References

Jegadeesh & Titman (1993); McLean & Pontiff (2016); Harvey, Liu & Zhu (2016); Bailey & López de Prado (2014), "The Deflated Sharpe Ratio"; López de Prado (2018), *Advances in Financial Machine Learning*; Novy-Marx & Velikov (2016); Gu, Kelly & Xiu (2020).
