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

Current results: planted → out-of-sample rank IC 0.063 (t = 7.8), net Sharpe 0.86, **DSR 0.99 → recovered**. Noise → IC −0.02, **DSR 0.0004 → correctly rejected**. A pipeline that "finds alpha" in noise has leakage; one that can't find a planted signal has bugs. This pipeline passes both.

**Walk-forward validation with embargo.** Expanding-window splits only; a 21-day embargo between train and test windows prevents overlapping forward-return labels from leaking future information into training (purged CV, López de Prado ch. 7). Standard k-fold on financial panels is silently invalid — this is the single most common fatal flaw in student projects.

**Costs are first-class.** Transaction costs are charged on every unit of turnover (default 10 bps/side), and annualized turnover is a headline metric, because almost no high-turnover published anomaly survives costs (Novy-Marx & Velikov, RFS 2016).

**Deflated Sharpe Ratio with an honest trial count.** The `--n-trials` flag forces you to declare how many strategy variants you have tried in total; the DSR then benchmarks your Sharpe against the expected maximum of that many noise draws. Tracking N is the discipline; the formula is the easy part.

**No lookahead by construction, verified by test.** Weights formed at date *t* earn returns only from *t+1*. The test suite includes a same-day-return exploit test that a buggy backtester fails loudly, plus a counter-test proving genuine foresight *would* profit (a check of the check).

**Baselines first.** Every run reports the model against (a) a one-line 12-1 momentum decile long-short and (b) an equal-weight 1/N portfolio, on the same out-of-sample dates, through the same cost-aware backtester. On the planted panel the momentum baseline (net SR 1.19) *beats* the ridge model (net SR 0.86) — the planted signal is literally momentum, and the model dilutes it across five features. If ML can't beat the baseline on real data either, that gets reported, not hidden.

**Vectorized but pinned.** The IC computation, weight construction, and walk-forward slicing are vectorized (~4.4× core speedup), and each optimized path is pinned by a test against a naive per-date reference implementation, so a future "optimization" that drifts the numbers fails the suite.

## Layout

```
src/quantlab/
  data.py        # yfinance loader with parquet cache; 60-name liquid US universe
  synthetic.py   # planted-signal and pure-noise panels for falsification tests
  features.py    # cross-sectionally z-scored: 12-1 momentum, 6-1, reversal, vol, 52w-high
  validation.py  # expanding walk-forward splitter with embargo
  models.py      # Ridge baseline + gradient boosting; per-date rank IC;
                 # ridge_cv = nested per-roll alpha tuning (train window only)
  baselines.py   # 12-1 momentum decile L/S + equal-weight 1/N benchmarks
  backtest.py    # dollar-neutral decile long-short, linear costs, turnover
  metrics.py     # Sharpe, max DD, PSR, Deflated Sharpe Ratio
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
python scripts/run_pipeline.py --data yfinance --model gbr --n-trials 5  # real data
```

Outputs land in `results/`: metrics JSON + equity-curve PNG per run.

## Known limitations (deliberate honesty)

The default real-data universe is today's liquid names — **survivorship-biased**, which inflates long-side returns; fixing this with point-in-time membership is on the roadmap. Costs are linear with no market-impact model. The label is a 21-day forward return; no risk-model neutralization (sector/beta) yet. Reported IC t-stats assume independent daily observations, but overlapping 21-day labels are autocorrelated — they're overstated until a Newey–West correction lands. Free daily data only. Every one of these is a roadmap item, and naming them is part of the point.

## References

Jegadeesh & Titman (1993); McLean & Pontiff (2016); Harvey, Liu & Zhu (2016); Bailey & López de Prado (2014), "The Deflated Sharpe Ratio"; López de Prado (2018), *Advances in Financial Machine Learning*; Novy-Marx & Velikov (2016); Gu, Kelly & Xiu (2020).
