# Do standard price signals survive an honest backtest? A falsification-first study on point-in-time S&P 500 data

**DRAFT — Phase 7.** This is a full working draft for the owner to rewrite
in his own voice; every number is traceable to a row in `research_log.md`
or an artifact in `results/` (law #8). Live-results sections update as
cycles mature.

## Abstract

We test whether five standard price-only cross-sectional signals (12-1 and
6-1 momentum, 1-month reversal, 3-month realized volatility, 52-week-high
proximity) carry exploitable alpha in large-cap US equities, under a
research protocol designed to make self-deception difficult: a planted-
signal/pure-noise falsification harness enforced in CI, point-in-time
index membership, embargoed walk-forward validation, Newey–West inference
on overlapping labels, cost-aware dollar-neutral construction, a declared
trial count feeding the Deflated Sharpe Ratio, and live paper-trading
verification. The headline result is negative, and the most instructive
exhibit is how the positive result died: on a static present-day universe
the pipeline produces IC 0.033 and net Sharpe 0.82; on the honest
point-in-time universe the same pipeline produces IC 0.005 and net Sharpe
−0.01. Across seven logged trials spanning horizons, labels,
neutralization schemes, and three model classes (ridge, gradient boosting,
shallow neural network), no configuration achieved a Deflated Sharpe above
0.04 or a correctly-signed |t_NW| ≥ 2. We quantify the capacity
consequences of turnover with a square-root impact model and verify the
pipeline live against a real broker API. We argue the negative result is
the expected one given the publication-decay literature, and that the
infrastructure for *establishing* it — which recovered every planted
signal and rejected every noise panel on demand — is the transferable
contribution.

## 1. Question

Can the most widely published price-only cross-sectional signals be
traded profitably in the S&P 500 universe today, net of costs, once every
known form of backtest inflation available to a free-data researcher is
removed? If not — and the literature predicts not — can a research
pipeline demonstrate that null convincingly, in a way that would survive
an adversarial review?

## 2. Why the bar is set where it is

Four empirical facts shaped every design decision:

1. Published anomaly returns decay 26% out-of-sample and 58%
   post-publication (McLean & Pontiff 2016). Whatever was in these signals
   when published, less of it remains.
2. At honest multiple-testing thresholds, most published factors are false
   discoveries (Harvey, Liu & Zhu 2016). A t-stat of 2 means little after
   hundreds of tries — so we count our tries: **N = 7, logged in
   `research_log.md`, never reset**.
3. Anomaly profits concentrate in high-turnover implementations whose
   costs exceed their gross returns (Novy-Marx & Velikov 2016). We never
   report a gross-only number; turnover is a headline metric.
4. The expected maximum Sharpe of N noise strategies grows with N (Bailey
   & López de Prado 2014). The Deflated Sharpe Ratio benchmarks every
   result against that expected maximum.

## 3. Data and universe

Daily adjusted prices via yfinance, 2009–2026. Index membership is
reconstructed point-in-time by walking Wikipedia's S&P 500 changes table
backward from the current constituent list: 810 distinct members ever in
the 2010–2026 window, of which 661 (81.6%) have retrievable price
history. Features are masked to members-as-of-date before cross-sectional
normalization, so departed names cannot shift the statistics the model
sees.

The residual bias is quantified rather than waved at
(`results/sp500_pit_coverage.json`): 149 dead names are unpriceable on
free data, and delisting returns — the final, usually catastrophic, price
move of a dying stock — are missing (Shumway 1997). The second hole is
**bounded, not assumed**: re-running the backtest with a synthetic −30%
final print forced onto every name whose price series dies mid-window
(29 of 661; most index-removed names keep trading) moves net Sharpe by
only **+0.006** versus a 0% control in the same environment — the book
was, on average, slightly short the dying names
(`results/metrics_sp500_ridge_dlret*.json`). We bound rather than impute
because delistings are missing *not at random* — they are missing
because the company died, and any imputation fit on survivors re-injects
survivorship bias by construction. The 149 never-priced names remain the
unbounded residual; closing it requires CRSP.

The static present-day universe is retained deliberately as a measured
exhibit rather than deleted as a mistake. Same code, same features, same
costs:

| | static (biased) universe | point-in-time universe |
|---|---|---|
| mean rank IC | 0.0333 | 0.0052 |
| net Sharpe | **0.82** | **−0.01** |
| equal-weight long-only SR | 1.00 | 0.81 |

Survivorship bias did not inflate the result — it *was* the result. A
universe composed of known survivors makes "buy the dip" unfalsifiable:
every drawdown in the panel is, by construction, one the company
survived. Reproducing McLean–Pontiff-scale decay in-house, from a single
universe correction, is the project's centerpiece exhibit.

## 4. Methodology

**Validation.** Expanding walk-forward with an embargo at least the label
horizon: a model trained through date t sees no row whose 21-day forward
label overlaps the test window. k-fold cross-validation is unusable on
financial panels — random folds put tomorrow in the training set of
yesterday — and the embargo width follows from the label construction,
not from convention.

**The falsification gate.** Before any real-data result is read, the
pipeline must (a) recover a synthetic planted signal (DSR 0.992 achieved)
and (b) find nothing in a pure-noise panel of identical shape (DSR 0.078
at N=1). Both checks run in CI on every push with hard exit-code gates;
a future commit that introduces leakage fails the build. During
development the harness caught real bugs, described in §6.

**Inference.** Daily rank ICs of overlapping 21-day labels share 20 days
of label information; treating them as independent overstates t-stats by
roughly √21. On the planted panel the naive t of 7.76 collapses to a
Newey–West t of 2.00 (lags = horizon, Bartlett kernel) — the shrinkage
theory predicts. All quoted t-stats are Newey–West.

**Construction and costs.** Decile dollar-neutral long-short, 10 bps
linear costs on turnover, monthly rebalance unless stated. Predictions
are sector-demeaned (12 GICS-as-of-today buckets) and weights projected
to zero ex-ante beta using rolling 252-day past-only betas. Exposure is
*measured*, not asserted: ex-ante beta 0.009 realizes as mean 0.05 with
p95 0.23 — beta estimation drift between rebalances is real and reported
per run.

**Baselines.** Every model must beat equal-weight and a one-line 12-1
momentum rank, same out-of-sample dates, same costs (law #5). On the
planted panel — where the planted signal *is* momentum-like — the
baseline beats the 5-feature ridge (net SR 1.19 vs 0.86): a multi-feature
model is not automatically better than its strongest ingredient. The
baseline also caught a bug a 9-test suite had missed (§6).

**Hyperparameters.** Per-roll nested walk-forward selection (inner splits
inside each outer training window) chooses ridge α from {1, 10, 100,
1000} without touching outer test data; a pinned leak-test verifies that
corrupting all post-train-window data cannot change the chosen α.
Selection on training data is in-sample model fitting, not a new trial.

## 5. Results

All out-of-sample, all net of 10 bps, all on the point-in-time universe
unless noted, N = 7 trials declared to the DSR:

| Trial | Config | IC | t_NW | net SR | DSR | turnover |
|---|---|---|---|---|---|---|
| 1 | static universe (exhibit) | 0.0333 | — | +0.82 | 0.998* | 3.8× |
| 2 | PIT universe, ridge, raw label | 0.0052 | +0.54 | −0.01 | 0.29 | 7.3× |
| 3 | + sector/beta neutralization | 0.0052 | — | −0.38 | 0.01 | 7.4× |
| 4 | 63d horizon & rebalance | −0.0278 | −1.95 | −0.35 | 0.01 | 2.4× |
| 5 | residualized label | +0.0225 | +1.91 | −0.77 | ≈0 | 3.5× |
| 6 | gradient boosting, residual label | +0.0077 | +0.80 | −0.12 | 0.04 | 5.4× |
| 7 | shallow MLP, residual label | +0.0093 | +1.21 | −0.28 | 0.008 | 5.8× |

\* DSR at N=1 on the biased universe — retained to demonstrate that the
DSR cannot rescue a backtest from a dishonest universe. It deflates for
multiplicity, not for survivorship.

**Verdict: no configuration produced a defensible edge.** Best DSR 0.04;
no correctly-signed |t_NW| ≥ 2. The model-class ablation (linear, trees,
shallow net on identical features and harness) localizes the null in the
*features*, not the learner: before each real-data run, every model class
recovered the planted signal (MLP more weakly than ridge — the Gu, Kelly
& Xiu shallow-net result) and rejected noise.

## 6. What failed (and what the harness caught)

This section is mandatory by project constitution.

- **The headline alpha was survivorship bias, in full** (trials 1→2). Not
  partially: IC fell 84% and net Sharpe went from +0.82 to −0.01.
- **A baseline too good to be true caught a data bug.** The first PIT run
  showed equal-weight SR 3.34 (impossible; RSP ≈ 0.9). Cause: pandas
  `pct_change()` pad-filling delisted names into frozen 0% daily returns,
  crushing measured volatility. After `fill_method=None` everywhere,
  EW SR 0.81. The strategy numbers barely moved — the bug lived only in
  dead names' return construction — but we would not have known that
  without the baseline. A wrong number that *looks* wrong is a gift.
- **Neutralization revealed nothing was hiding** (trial 3). Removing
  sector/beta exposure cut volatility 10.0%→6.5% and *worsened* net
  Sharpe: factor exposure had been adding noisy return, not masking alpha.
- **We declined to trade a sign-flip** (trial 4). At the quarterly
  horizon the IC went negative (t_NW −1.95). "My signal works reversed"
  after N variants is textbook max-of-N mining; if quarterly reversal is
  a real hypothesis it gets its own pre-registered test on fresh data.
- **IC and P&L are different objects** (trial 5). Against a residualized
  label the IC "improved" to +0.0225 (partly definitional — the label is
  cleaner) while the portfolio lost money gross. We report both and trust
  the P&L.
- **A cache-key bug was caught before it could corrupt anything**: price
  caches were keyed on the *number* of tickers, so two same-size
  universes would silently collide. Found in code review before the first
  real-data run; keys are now content hashes.
- **The live deployment shipped with its measurement broken** (caught on
  day 1): the daily cycle logged portfolio weights but not predictions,
  and the backtest IC is computed on pre-neutralization predictions —
  live IC could never have been compared like-for-like. One cycle was
  lost to this; the prediction log is now written, write-once, before any
  order exists.
- **A missing transitive dependency would have killed the first
  unattended cloud cycle**: `lxml` (pandas' HTML parser backend) was
  installed ad hoc on the development machine but absent from
  `requirements.txt`, which is all CI installs. A fresh environment is
  the only honest test of a requirements file.
- **The synthetic lab caught a confound that would have faked our own
  registered hypothesis.** While building regime-detection machinery
  (falsification-first: the detector had to provably not see the future
  before touching real data), signal-free synthetic worlds showed
  residualized-label momentum diagnostics carrying vol-regime-dependent
  IC of +0.06 to +0.13 — beta-estimation error and label machinery
  interacting with volatility regimes. On real data this masquerades
  exactly as "momentum works conditionally on volatility," the
  pre-registered H3/H4 hypothesis. Consequence: those hypotheses now
  require a paired artifact control before any run. Relatedly, the
  standard HMM regime construction (forward–backward smoothed state
  probabilities, the default output of off-the-shelf libraries) is
  demonstrably anticausal — perturbing the future moves "past" state
  estimates — and is pinned as a regression test, with only the causal
  forward filter exposed to strategy code.
- **Absolute levels lied where paired controls didn't — three separate
  times** (regime world, carry world, delisting bound). In each case an
  absolute-threshold test was seed-fragile or artifact-contaminated, and
  the fix was the same: difference two worlds that share every random
  draw and differ only in the planted effect. Paired controls are now
  house doctrine, not a technique.
- **The first data-revision measurement calibrated its own instrument.**
  Day one of diffing consecutive vendor downloads of the *same past*
  flagged 51% of 1.3M price cells at a 1e-9 tolerance: the vendor
  re-serves history with ~1e-7 relative float wobble. Above that noise
  floor sat real same-day rewrites: full-history dividend re-scalings
  (0.3–1.1%) and one 90% split-factor repair touching 2,059 return
  cells of a single name. The instrument now separates a noise band
  from revisions; the miscalibrated first fingerprint stays committed
  as the calibration record.

## 7. Capacity and execution realism

Square-root impact sweep (k = 1.0, spread 10 bps, one-day execution, ADV
coverage 87.6%) on the deployed config
(`results/capacity_sp500_ridge_both_residlabel.json`):

| AUM | impact + spread drag (ann.) | net SR |
|---|---|---|
| $1M | 1.36% | −0.85 |
| $10M | 2.27% | −1.00 |
| $100M | 5.16% | −1.42 |
| $1B | 14.30% | −2.04 |

The gross edge is negative, so formal capacity is $0. The informative
object is the **drag curve**: any strategy with this turnover profile
(3.46×/yr, ~500-name S&P book) must generate ≥ 1.4%/yr of true gross alpha
to exist at $1M and ≥ 5%/yr at $100M — a quantified statement of
Novy-Marx–Velikov cost mortality on our own book.

Reproducibility note discovered en route, then industrialized: a fresh
data download in a fresh environment reproduced trial #5 to ~1e-7
relative — yfinance *retroactively re-adjusts* history. "Point-in-time
data" therefore has a data-revision dimension, not just a membership
dimension, and the live infrastructure now measures it daily: every cycle
diffs its fresh download against the previous cycle's snapshot of the
same past and commits the fingerprint (`results/live/revisions_*.json`),
separating real revisions (dividend re-scalings, split repairs) from the
vendor's ~1e-7 serving noise (§6).

## 8. Live verification (running)

Why paper-trade a null result? Because live IC vs backtest IC is the
ultimate out-of-sample test *of the pipeline and of the null itself* — if
live ICs cluster meaningfully above the backtest's 0.0225, something is
wrong with the backtest, and that is worth knowing.

A daily CI job rebuilds the point-in-time universe, trains only on rows
whose labels are fully realized (live trading inherits the backtest's
leakage discipline), writes the full prediction cross-section to a
write-once log *before any order exists*, then submits integer-share,
per-name-capped orders to an Alpaca paper endpoint (the client refuses
non-paper URLs). The log is committed to the repository — an append-only,
timestamped record.

The experiment has a **control arm**: every cycle shadow-logs the 12-1
momentum baseline's values on the same names (no orders). If live IC
sags below backtest IC, the baseline's own live-vs-backtest gap
separates "the model decayed" from "the period was hostile to
everything" — a live test without a control cannot tell those apart.
Each cycle also commits a data-revision fingerprint (§7) and, since
2026-06-12, a daily short-borrow snapshot of the live universe (IBKR
availability and fee rates — an unbackfillable record collected under a
registered, collection-only protocol; first snapshot: median fee 1.2%,
99th percentile 164%).

- Cycles 1–3 (2026-06-10 → -12) traded clean: ~100-name books, all
  orders accepted, zero failures. Cycle 1 predates prediction logging,
  so the live-IC record is one cycle shorter than the trading record.
- First measurable live IC: ~2026-07-10 (cycles mature at 21 trading
  days). A Newey–West t needs ≥23 matured cycles; until then live ICs
  are reported but not interpreted.
- [TO UPDATE as cycles mature: live IC table, `results/live/live_ic.png`,
  realized-vs-broker P&L cross-check.]

## 9. Limitations

Residual survivorship bias (149 unpriceable dead names; no delisting
returns — both flatter the signals we nonetheless reject); sectors are
as-of-today; betas are estimated, with measured drift; linear costs in
headline results (impact priced separately in §7); free daily data; a
single market and period; and the live record is one cycle shorter than
the trading record (§6).

## 10. What institutional-grade would require

CRSP-quality delisting returns; point-in-time fundamentals and GICS;
borrow availability and fees on the short book (in-house daily collection
of exactly this began 2026-06-12 — institutional history would still
require a vendor); an impact model calibrated to actual fills rather than
a literature constant; multi-market replication; and an
order-of-magnitude more trials under the same logging discipline. New
trials are themselves now machine-enforced: real-data runs refuse to
start unless they name a pre-registered, still-PROPOSED hypothesis or
declare themselves a reproduction of logged work.

## References

Bailey, D. & M. López de Prado (2014), "The Deflated Sharpe Ratio";
Gu, S., B. Kelly & D. Xiu (2020), "Empirical Asset Pricing via Machine
Learning"; Harvey, C., Y. Liu & H. Zhu (2016), "...and the Cross-Section
of Expected Returns"; Jegadeesh, N. & S. Titman (1993); López de Prado,
M. (2018), *Advances in Financial Machine Learning*; McLean, R.D. & J.
Pontiff (2016), "Does Academic Research Destroy Stock Return
Predictability?"; Novy-Marx, R. & M. Velikov (2016), "A Taxonomy of
Anomalies and Their Trading Costs"; Shumway, T. (1997), "The Delisting
Bias in CRSP Data".
