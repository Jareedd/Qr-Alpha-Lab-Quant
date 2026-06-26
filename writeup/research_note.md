# Does anything survive an honest backtest? A falsification-first study across three asset classes

*A quantitative research note. Every number traces to a row in `research_log.md`
or an artifact in `results/`; the code that produced each is in the repository.*

---

## Abstract

I set out to answer a narrow question honestly: do the standard published
price signals still pay, in the markets a retail-data researcher can actually
reach, once you remove every form of backtest inflation you have the
discipline to remove? The answer, across thirteen logged trials, is mostly
no — and the *way* it is no is the contribution. On a survivorship-biased
universe my pipeline produces a net Sharpe of 0.82; on the point-in-time
universe, the same code on the same features produces −0.01. The alpha was the
bias. Seven equity trials spanning horizons, labels, neutralization, and three
model classes never cleared a Deflated Sharpe of 0.04. Then I pointed the same
machine at crypto-perpetual funding carry — a premium with a structural reason
to exist — and it found a real, strongly significant signal (t = −3.5, net
Sharpe 0.87) that *still* failed my pre-registered evidence bar (Deflated
Sharpe 0.865 < 0.95), is severely crash-skewed, and has decayed from Sharpe 2.3
to ~0.4 as the trade institutionalized. A ninth trial showed the famous S&P
deletion rebound is just matched small-loser mean reversion. A tenth ran the
same carry machine into the liquid tail and found a signal just as significant
(t = −3.6) that loses money net — the cleanest proof in the project that
statistical significance and tradability are different objects. An eleventh
cleared every pre-registered bar at once — net Sharpe 1.1, Deflated Sharpe
0.999 — until a one-week entry lag collapsed it to nothing, exposing a
microstructure artifact in a discount-reversion costume: the discipline
overturning its own best-looking number. Then the survivorship wall that
produced the project's *first* result finally fell — not to paid data but to a
free SEC name-to-CIK crosswalk plus delisting-inclusive prices — which let the
long-blocked quality-fundamentals hypothesis finally run (trial #12): its raw
profitability edge was real but evaporated under value-neutralization, a premium
that was value in disguise. A thirteenth trial took the most-cited free
alternative-data anomaly — opportunistic insider cluster-buying — and, this time
genuinely powered, found a clean null in which the "opportunistic" signal was no
better than the routine one it is supposed to beat. And a delta-neutral crypto
cash-and-carry audit closed the loop: a real funding harvest whose premium over
the risk-free rate has been arbitraged to zero, its headline Sharpe of 4.6 a
tail-risk illusion. The deliverable here is not a strategy.
It is a research process that detects a real premium where one exists, refuses
to claim it where one does not, and overturns its own result when it is too good
to be true — and the evidence that I can tell the difference.

---

## 1. The question

Can the most widely published price-only cross-sectional signals — 12-1 and 6-1
momentum, short-term reversal, low volatility, distance from the 52-week high —
be traded profitably in the S&P 500 today, net of costs, after I remove every
backtest inflation available to a free-data researcher? And if they cannot —
which the published record predicts — can I demonstrate that null *convincingly
enough to survive an adversarial interview*? Late in the project I added a
second question that turned out to matter more: when the same pipeline meets a
market where a premium genuinely exists, does it find it, and does my own
discipline still hold when the number finally looks good?

## 2. Why the bar is set where it is

I did not choose my methods for rigor's own sake. Four findings from the
literature dictated every design decision, and I treat them as constraints, not
suggestions:

1. **Anomalies decay.** Published anomaly returns fall 26% out-of-sample and
   58% post-publication (McLean & Pontiff 2016). Whatever was in these signals
   when they were published, less of it is left.
2. **Most published factors are false.** At honest multiple-testing
   thresholds, the majority of published factors are likely false discoveries
   (Harvey, Liu & Zhu 2016). A t-stat of 2 means little after hundreds of
   attempts — so I count mine. The global trial count is logged and never
   reset; it feeds the Deflated Sharpe directly. As of this note, N = 13.
3. **Turnover kills.** Anomaly profits concentrate in high-turnover
   implementations whose costs exceed their gross returns (Novy-Marx &
   Velikov 2016). I never report a gross-only number. Turnover is a headline.
4. **Max-of-N looks like alpha.** The expected maximum Sharpe of N noise
   strategies grows with N (Bailey & López de Prado 2014). The Deflated
   Sharpe Ratio benchmarks every result against that expected maximum, at my
   true N.

## 3. Data and universe — where the first result came from

Daily adjusted prices via yfinance, 2009–2026. I reconstruct point-in-time
S&P 500 membership by walking Wikipedia's changes table backward from the
current constituents: 810 distinct members ever held a seat in the 2010–2026
window, of which 661 (81.6%) have retrievable price history. Features are
masked to members-as-of-date *before* cross-sectional normalization, so a name
that left the index cannot shift the statistics the model sees while it was
out.

I keep the static, present-day universe deliberately — not as a mistake to
delete but as a measured exhibit. Same code, same features, same costs:

| | static (survivorship-biased) | point-in-time (honest) |
|---|---|---|
| mean rank IC | 0.0333 | 0.0052 |
| net Sharpe | **0.82** | **−0.01** |
| equal-weight long-only Sharpe | 1.00 | 0.81 |

Survivorship bias did not *inflate* the result; it **was** the result. A
universe of known survivors makes "buy the dip" unfalsifiable — every drawdown
in the panel is, by construction, one the company lived through. Reproducing
McLean–Pontiff-scale decay in-house, from a single universe correction, is the
centerpiece of this project. It is also the moment I learned to distrust my own
good numbers first.

I do not wave at the residual bias; I bound it. 149 dead names are unpriceable
on free data, and delisting returns — the final, usually ugly, move of a dying
stock — are missing (Shumway 1997). For the names that die mid-window (29 of
661), I forced a synthetic −30% final print and re-ran: net Sharpe moved by
**+0.006** against a 0% control. The missing delisting returns can neither
rescue nor sink the result. I bound rather than impute, because delistings are
missing *because the company died* — any model fit on survivors re-injects the
exact bias I just removed. The 149 never-priced names remain the unbounded
residual, and closing them needs CRSP. I would rather state that precisely than
pretend it away.

## 4. Methodology

**Validation.** Expanding walk-forward with an embargo at least the label
horizon: a model trained through date *t* never sees a row whose 21-day forward
label overlaps the test window. k-fold cross-validation is unusable here —
random folds put tomorrow in yesterday's training set — and the embargo width
is dictated by the label, not by convention.

**The falsification gate.** Before I read any real-data result, the pipeline
must clear two synthetic checks that run in CI on every push: recover a planted
signal (Deflated Sharpe 0.992) and find nothing in a pure-noise panel of
identical shape (0.078). If a future commit introduces leakage, noise mode
"finds alpha" and the build fails loudly. This gate is the single most useful
thing I built, and it earned its keep repeatedly (§6).

**Inference.** Daily rank ICs of overlapping 21-day labels share 20 days of
information; treating them as independent overstates significance by roughly
√21. On the planted panel a naive t of 7.76 collapses to a Newey–West t of 2.00
— exactly the shrinkage theory predicts. Every t-stat in this note is
Newey–West, lags = horizon.

**Construction and costs.** Decile dollar-neutral long-short, 10 bps linear
cost on turnover, sector-demeaned and projected to zero ex-ante beta with
rolling past-only betas. I measure exposure rather than assert it: ex-ante beta
of 0.009 realizes as a mean of 0.05 (95th percentile 0.23), and I report that
drift every run instead of pretending neutralization is exact.

**Baselines.** Every model must beat equal-weight and a one-line 12-1 momentum
rank on identical dates, net of costs. On the planted panel — where the planted
signal *is* momentum — the one-liner beats my five-feature ridge (1.19 vs 0.86).
A multi-feature model is not automatically better than its strongest
ingredient, and a baseline that looks too good is as informative as a strategy
that does (§6).

**Family-wise overfitting (PBO).** The Deflated Sharpe deflates a *single*
result for the number of trials; its family-wise complement is the Probability
of Backtest Overfitting (Bailey–López de Prado, combinatorially-symmetric
cross-validation), which asks of a family of comparable configs on the *same*
data: how often does the in-sample-best land below the out-of-sample median?
Over the comparable equity family (trials #2/#3/#5/#6/#7 on the shared
point-in-time universe — 3,253 aligned days, 12,870 symmetric splits) PBO is
0.24. A low PBO would normally reassure, but here it is *not* a green light: it
reflects persistent **structural** ordering among uniformly-unprofitable configs
(the residual-label variants reliably do worse than the raw-label one), while
the in-sample→out-of-sample Sharpe degradation slope (−0.89) and the 71%
probability that the selected config loses money out-of-sample confirm there is
no monetizable edge to overfit *to*. Read together the three numbers reaffirm
the equity null; read alone, PBO would mislead — which is the methodological
point, and the reason I report it alongside, not instead of, the per-trial DSR.
I never compute PBO *across* the thirteen trials: they live on different
universes, horizons and asset classes, so they share no return matrix to rank on
— and pretending otherwise is exactly the overfitting an overfitting metric
should be the first to refuse.

## 5. Results

All out-of-sample, all net of costs, all at the true N for the Deflated Sharpe.

| # | what I tested | IC | t_NW | net SR | DSR | turnover |
|---|---|---|---|---|---|---|
| 1 | static universe (exhibit, biased) | 0.0333 | — | +0.82 | 0.998\* | 3.8× |
| 2 | point-in-time universe | 0.0052 | +0.54 | −0.01 | 0.29 | 7.3× |
| 3 | + sector/beta neutralization | 0.0052 | — | −0.38 | 0.01 | 7.4× |
| 4 | quarterly horizon (turnover attack) | −0.0278 | −1.95 | −0.35 | 0.01 | 2.4× |
| 5 | residualized label | +0.0225 | +1.91 | −0.77 | ≈0 | 3.5× |
| 6 | gradient boosting | +0.0077 | +0.80 | −0.12 | 0.04 | 5.4× |
| 7 | shallow neural net | +0.0093 | +1.21 | −0.28 | 0.008 | 5.8× |

\*Deflated Sharpe at N=1 on the biased universe — kept only to show that the
DSR deflates for multiplicity, not for survivorship. No honest statistic
rescues a dishonest universe.

**No equity configuration produced a defensible edge.** Best Deflated Sharpe
0.04; no correctly-signed |t_NW| ≥ 2. The model-class ablation matters: linear,
trees, and a shallow net on identical features and harness are all null, and
each recovered the planted signal beforehand. That localizes the null in the
*features*, not the learner. I declined to trade trial #4's negative quarterly
IC — "my signal works reversed" after N attempts is textbook mining, and if
quarterly reversal is a real idea it gets its own pre-registered test on fresh
data, not a salvage of this one.

### 5.1 The first non-null — crypto-perp funding carry (trial #8)

Seven nulls share a weakness as evidence: a skeptic can say the pipeline simply
cannot find anything. So I ported the same machine to a market where a premium
has a structural reason to exist — perpetual-futures funding carry, the payment
made to whoever warehouses crowded leveraged-long demand — on a universe with
no survivorship gap, since delisted perp contracts keep their full terminal
histories in the exchange's own public dumps (729 USDT contracts ever, delisted
ones included).

The label is the part that silently invalidates a careless version of this
study. A perp position earns price return *and* a funding transfer; the premium
lives in the funding leg. A price-only label measures the wrong object — I
proved this on synthetic data, where the same book scores ≈ +1 Sharpe on a
funding-inclusive label and ≈ −2 on a price-only one. The registered config:
point-in-time top-30 by dollar volume, trailing-7-day funding signal,
funding-inclusive total-return label, dollar-neutral weekly quartile
long-short, 7 bps per side.

| metric | value | my bar | |
|---|---|---|---|
| IC t_NW | −3.54 | ≤ −2 | ✓ |
| net Sharpe | +0.87 | > 0 | ✓ |
| **Deflated Sharpe (N=8)** | **0.865** | **≥ 0.95** | **✗** |
| survives ex-top-3 | 0.68 | > 0 | ✓ |
| shuffled-funding control | 0.08 | ≈ 0 | ✓ |
| skew / max drawdown | −1.87 / −74% | — | crash-prone |

**Registered criteria not met — the one failed leg is the Deflated Sharpe — and
I did not relax it.** "0.865 is close" is exactly the rationalization the
pre-registration protocol exists to forbid. Before trusting the number I hunted
for the leak, because a 0.87 Sharpe in crypto is when you should: I delayed
entry one to five days and the Sharpe decayed gracefully (0.87 → 0.39) rather
than collapsing, which is the signature of a real edge and not a timing leak;
only two of twenty-eight delisted names were ever held at death, so there is no
delisting optimism; the shuffled-funding control is flat. The signal is real.
The honest killer is decay: Sharpe was 2.28 in the 2020–21 leverage mania and
fell to ~0.4 afterward as basis-trade funds industrialized the premium —
McLean–Pontiff, reproduced in a second asset class. The −1.87 skew is the
structural cost of selling funding: years of pennies, then a squeeze, and a
−74% drawdown to show for it. That is precisely why the skew-aware Deflated
Sharpe, not the raw Sharpe, is the right judge. This trial is the strongest
evidence in the project: the machine found a real premium where one exists,
which means the seven equity nulls were genuine absence, not impotence.

### 5.2 A second replication — the disappeared deletion effect (trial #9)

I tested whether discretionary S&P 500 deletions rebound in the 60 days after
the effective date, beyond a basket matched on size and trailing return (75
usable events, 2010–present). They do not: daily event-time portfolio net
Sharpe −0.04, t_NW −0.10, Deflated Sharpe 0.05. The method is the result.
Deleted names *do* bounce — about +4.8% over 60 days — but a size-and-momentum
matched control bounces +2.6% of it, and the +2.2% residual is insignificant (t
0.87) and negative before 2015. The rebound is small-loser mean reversion, not
an index-deletion effect, which is Greenwood & Sammon's (2025) "disappeared
index effect" reproduced in-house. A synthetic planted-event gate passed first,
so the null is the harness finding genuine absence. Without the control basket
you would see "deleted names rebound 5%" and imagine alpha; the control is the
discipline that separates a real anomaly from a mechanical artifact.

### 5.3 A third cut at carry — the liquid tail (trial #10)

Carry was real in the top-30; was a fresher, wider premium hiding in the tail
(dollar-volume ranks 31–150), beneath the basis-trade funds that decayed the
majors? I pre-registered the same machine on the disjoint tail universe — a
genuinely new registration, not a salvage of trial #8 — judged at the same bar,
with conservative 20 bps/side tail fills. The funding signal predicts the
cross-section just as strongly as the majors — IC t_NW −3.62 versus the top-30's
−3.54 — and it still loses money: net Sharpe −0.13 (gross 0.26). The
decomposition is the lesson: funding income contributes +1.23 in cumulative
P&L, but price drift gives −0.85 of it back, so the tail carry is almost fully
priced, and 20 bps fills finish it. This is the cleanest IC-≠-P&L exhibit in the
project — a t = −3.6 signal that loses money net — and it teaches at scale what
trial #5 taught once: a strongly significant cross-sectional signal and a
tradable edge are different objects, and only the net P&L decides.

### 5.4 The graduation that wasn't — closed-end-fund discount reversion (trial #11)

The one structurally-protected premium I had left was the closed-end-fund
discount. A CEF has no creation/redemption mechanism, so price can sit away from
NAV persistently, and in the sub-$400M tail no activist is large enough to
arbitrage it. A two-stage, zero-trial Stage-1 first cleared the survivorship
question that sinks most free-data ideas — the *opposite* way: I enumerated 151
dead CEFs from SEC filings and found 94% died at NAV (liquidation, merger,
open-ending, term maturity) with zero distress delistings, so omitting dead
funds biases a discount-long *against* itself. For once, the survivorship gate
cleared in my favor.

The registered Stage-2 run then did what none of the others had: it passed every
pre-registered criterion at once — net Sharpe 1.11, Deflated Sharpe 0.999 at
N=11, IC t_NW −10.4, positive skew, beating the equal-weight baseline and
surviving the controls. The prime directive says a result that good is first a
suspected bug, so I ran the diagnostic that decides it: an entry-lag sweep. The
Sharpe collapsed from 1.11 to 0.10 with a single week of delay, then went
negative. The entire "edge" was a one-week bounce — a microstructure /
shared-price artifact, because the discount at week *w* is built from the same
noisy close the next week's return divides by; you cannot trade the print you
measured. The implementable version is null. H6 did not graduate, and I did not
relax the criteria to pretend otherwise. The lesson is now a registered
requirement, not a footnote: a reversion strategy must pass an
entry-lag/implementability gate, not merely the in-sample bar.

### 5.5 The survivorship wall finally falls — quality fundamentals (trial #12)

The hypothesis blocked since trial #2 was fundamental quality, because the thing
that makes survivorship bias matter — dead names — is exactly what free
fundamentals data drops: the SEC's ticker→CIK map is present-day only, so an
acquired or bankrupt constituent's filings become unreachable. I had measured
that a naive free recovery lifts coverage only 73%→75%. The unlock was to stop
matching on the recycled *ticker* and match on the *company name*: SEC's
`cik-lookup-data.txt` resolves ~94% of dead S&P names to their operating CIK
(reassignment-immune by construction, because it follows the name through each
rename to the right filer), and Tiingo supplies delisting-inclusive prices. A
full 290-name spot-audit put the recovery at ~99% correct. The wall that defined
trial #2 fell — on free data, no CRSP.

So H1 finally ran. The registered construction (amended pre-data from an
openassetpricing review): cash-based operating profitability, CBOP/A, the
accruals-robust profitability measure; value-weighted quintile long-short;
Financials and Real Estate excluded (no cost-of-goods line); and — the load
bearing choice — graded on a **value-neutralized** arm, because Novy-Marx's
large-cap profitability is an FF-alpha riddled with HML exposure, and a "quality"
premium that is merely value re-labeled must not count. A synthetic two-world
gate proved in-environment that the neutralization could actually tell a
value-disguised edge (must collapse) from a value-orthogonal one (must survive)
before any real number was trusted.

| arm | net SR | t_NW | DSR (N=12) | |
|---|---|---|---|---|
| RAW CBOP/A | +0.58 | +2.30 | 0.72 | looks edged |
| **HML-NEUTRAL** | **−0.18** | **−0.77** | **0.009** | the graduation arm |

The raw book looked tradable (t 2.3) but sat *below* the equal-weight baseline
(+0.99) and the deflation bar; the neutral arm — the one graduation is judged on
— is negative. The **raw-minus-neutral gap of +0.755** is the pre-registered
signature that the edge was repackaged HML exposure: the quality premium was
value in disguise, exactly as the dossier predicted. PBO across the
{raw,neutral}×{current,lagged-assets} family was 0.10 — and, as in the equity
trials, a low PBO in a null family is rank-persistence, not a green light. H1
does not graduate. The first survivorship-safe free-data fundamentals trial in
the project is a clean, mechanistic null. (n_obs is 64 quarters and market-cap
coverage 64%, but value-collinearity is structural — neutralizing HML flips the
sign — and robust to the coverage gap.)

### 5.6 The most-cited alternative-data anomaly, properly powered — insider cluster-buying (trials, and a power-abort, leading to #13)

The next candidate was opportunistic insider cluster-buying (Cohen–Malloy–
Pomorski): insiders who buy off their routine calendar carry information, and a
cluster of them is the signal. Form 4 is survivorship-safe on the signal side —
it persists under the issuer CIK after a ticker dies — which is the hole that
sinks most free-data ideas, closed for once.

The first registration (H10, top-decile long-vs-EW) hit a wall the project's
discipline is built to respect: a **pre-spend power gate**. Computing the real
cross-section required Form 4s across the whole universe — ~200k rate-limited
requests as a per-filing crawl — so I built a bulk-data pipeline reading SEC's
quarterly Form 345 datasets (~64 files), and cross-checked it byte-for-byte
against the crawl. The check surfaced a genuine completeness bug in the crawl
(it reads only the ~1,000 most-recent filings, silently dropping older ones for
prolific filers); the bulk source is the correct one. The full-universe power
gate then showed a median of ~24 cluster-eligible firms per month — real insider
clustering exists in large caps — but a *top-decile* book is ~2 names, far below
a tradable cross-section. **The trial aborted before spending N** — the trial-#10
fee-first precedent, now for alternative data: you do not run an underpowered
study just because you can.

The redesign (H12, trial #13) longed *all* cluster names equal-weight versus the
market — genuinely powered: 196 monthly observations, a minimum detectable
Sharpe of 0.42. It is a clean null: net Sharpe −0.13, t_NW −0.58, Deflated Sharpe
0.013, PBO 1.000, and — the decisive detail — the **opportunistic arm (−0.13) is
no better than the routine arm (−0.06)** it is supposed to dominate, so what
little is there is generic buying pressure, not information. Crucially, both the
synthetic machinery gate (planted Sharpe ~8) *and* the power gate passed, so this
is genuine economic absence, not the impotence-or-underpowering escape hatch — the
clean counterfactual the H10 power-abort could not provide.

### 5.7 The real-money question — delta-neutral crypto cash-and-carry (Stage-1 audit)

The one trade that pays cash, not a cross-sectional spread, is cash-and-carry:
long spot, short the perpetual, collect funding while delta-hedged. I ran it as a
feasibility audit (zero trials) precisely because the headline *looked* like the
edge I had been hunting — majors 12.9%/yr net carry at a Sharpe of 4.64 — and
the prime directive says a number that good is a suspected bug. The decomposition
acquitted the code and convicted the trade: carry ≈ funding, basis P&L ≈ 0, and
it survives an entry-lag, so it is the *real* funding harvest, not a
close-timing artifact. But its premium over the risk-free rate has been
arbitraged to zero — gross funding fell from 32.8% in the 2021 mania to ~4–5%
in 2025–26, i.e. roughly T-bills — and the 4.64 Sharpe is a *tail-blind illusion*:
funding income is smooth right up to the funding-flip / liquidation-cascade /
exchange-failure / stablecoin-de-peg event that delta-neutral reduces but cannot
remove. SOL even lost outright (basis drift). I deliberately did **not** run it
as a graded trial, because a naive Deflated Sharpe on that smooth stream would
"graduate" it misleadingly — the exact false win this project refuses. It is the
McLean–Pontiff decay confirmed a third time, and the honest verdict is that
cash-and-carry today is a dormant-but-armed vehicle: it sits flat when funding is
priced and would only harvest a future mania.

## 6. What failed, and what the harness caught

This section is mandatory by my own rules, and it is the part I am most willing
to be judged on.

- **The headline alpha was survivorship bias, in full** (trials 1→2): IC fell
  84%, net Sharpe went +0.82 → −0.01.
- **A baseline too good to be true caught a data bug.** The first PIT run
  showed equal-weight Sharpe 3.34 — impossible. Cause: pandas pad-filled
  delisted names into frozen 0% returns, crushing measured volatility. A
  baseline I expected to be ~0.9 reading 3.3 is a gift; the strategy numbers
  barely moved, but I would not have known without it.
- **Neutralization revealed nothing was hiding** (trial 3): removing factor
  exposure cut volatility 10.0% → 6.5% and *worsened* the Sharpe. The factor
  exposure was adding noisy return, not masking alpha.
- **The synthetic lab caught a confound that would have faked my own
  hypothesis.** Building regime-detection machinery, I found that
  residualized-label momentum carries a vol-regime-dependent IC of +0.06 to
  +0.13 *on signal-free data* — beta-estimation error interacting with
  volatility. On real data that masquerades exactly as "momentum works in calm
  regimes," a hypothesis I had registered. It now requires a paired artifact
  control before any run. Relatedly, the standard HMM regime construction
  (forward-backward smoothed states) is anticausal — perturbing the future
  moves "past" state estimates — and I pin that as a regression test, exposing
  only the causal forward filter to strategy code.
- **Absolute levels lied where paired controls did not — three separate
  times** (the regime world, the carry world, the delisting bound). Each time a
  threshold test was seed-fragile or artifact-contaminated, and the fix was to
  difference two worlds sharing every random draw. Paired controls are now
  house doctrine.
- **A carry-book bug, caught before the trial ran.** My weight construction
  carried stale positions across rebalances, which would have broken
  dollar-neutrality and inflated gross over a multi-month run. The existing
  tests missed it (constant-funding fixtures never change the selection); a
  pre-trial adversarial review did not.
- **"Point-in-time" has a data-values dimension, not just a universe
  dimension.** My live infrastructure diffs each day's fresh download against
  the prior day's snapshot of the same past. Day one flagged that the vendor
  re-serves history with ~1e-7 float noise, and above that floor, real same-day
  rewrites: full-history dividend re-scalings and one 90% split repair on a
  single name. The backtest and the live model literally trained on different
  versions of 2020, and now I measure by how much.
- **A pre-registered backtest that passed every bar, overturned by one
  diagnostic** (trial #11). CEF discount reversion cleared net Sharpe 1.1,
  Deflated Sharpe 0.999, and t = −10.4 — then an entry-lag sweep collapsed it
  from 1.1 to 0.1 in a single week, exposing a one-week microstructure bounce
  rather than reversion. The in-sample criteria were insufficient;
  implementability had to be tested too. The strongest number I produced is the
  one the discipline killed.
- **A degenerate run is not a null — and I refused to log it as one** (trial
  #12). The first quality run came back with zero market-cap coverage, Sharpe
  0.000, t = nan: shares outstanding live under SEC's `dei` namespace and the
  `shares` unit, not `us-gaap`/`USD` where the reader looked. A pipeline that
  produces nothing is not evidence of absence; I fixed the reader and re-ran
  rather than bank a free "null." Distinguishing a real null from a broken run is
  the whole game.
- **Three bugs in the alt-data build, each caught by a *different* guard.** A
  same-filed-date error in the insider classifier (a single multi-transaction
  Form 4 over-counted "routine") was caught by an adversarial pre-commit review,
  not its own author's test. A crawl that silently dropped older filings for
  prolific issuers was caught by an *independent* bulk-vs-crawl cross-check I ran
  rather than trusting the build agent's "match." A Binance change from
  millisecond to microsecond timestamps (which errored ~every symbol) was caught
  by reading the run's actual output instead of trusting that "it's running."
  Each is now pinned by a regression test. The lesson repeats: the guard that
  catches the bug is rarely the one its author wrote.
- **A Sharpe of 4.64, deflated rather than celebrated** (cash-and-carry, §5.7).
  The funding-vs-basis decomposition showed it was a real funding harvest — and
  then that the harvest had decayed to the risk-free rate and the Sharpe was
  blind to the tail. The reflex the prime directive demands — treat a great
  number as a suspected error first — is what turned a "GO" into the honest
  verdict.

## 7. Capacity and execution

Square-root impact sweep on the deployed equity config (spread 10 bps, ADV
coverage 87.6%):

| AUM | annual cost drag | net Sharpe |
|---|---|---|
| $1M | 1.36% | −0.85 |
| $10M | 2.27% | −1.00 |
| $100M | 5.16% | −1.42 |
| $1B | 14.30% | −2.04 |

The gross edge is negative, so formal capacity is $0. The informative object is
the drag curve: any strategy with this turnover profile (3.46×/yr) needs ≥1.4%
of true gross alpha per year just to exist at $1M, and ≥5% at $100M — a
quantified statement of cost mortality on my own book. The carry strategy, at
35×/yr turnover, would need far more, which is a second reason it does not
graduate beyond a logged result.

## 8. Live verification (running)

I paper-trade the best honest equity config against a real broker API, not
because I expect it to make money — I have shown it will not — but because live
IC versus backtest IC is the ultimate out-of-sample test of the *pipeline*. A
daily job rebuilds the point-in-time universe, trains only on fully-realized
labels, writes the full prediction cross-section to a write-once log *before*
any order exists, and submits integer-share, per-name-capped orders to a paper
endpoint. The experiment has a control arm — every cycle shadow-logs the
momentum baseline on the same names — so that if live IC sags, I can tell "the
model decayed" from "the period was hostile to everything." First measurable
live IC matures roughly 21 trading days after the first cycle; a Newey–West
t-stat needs more than 23 matured cycles, and until then I report the numbers
without interpreting them. I also collect, daily, two datasets that cannot be
backfilled: the data-revision fingerprints above and a short-borrow-fee
cross-section. The point of those is the moat — a record that only exists
because someone started snapshotting it.

## 9. Limitations

Residual survivorship bias (149 unpriceable dead names; missing delisting
returns, bounded at ±0.006 Sharpe for the priceable diers); sectors as-of-today;
estimated rather than known betas, with measured drift; linear costs in the
headline with impact priced separately; the carry universe's exchange-side
survivorship is small but nonzero; free daily data throughout; and three asset
classes over roughly fifteen and six years (equities and crypto) plus closed-end
funds over ~14 years of weekly free data, the last of which cleared every
in-sample bar but failed an entry-lag implementability check. Every one of these is
named on purpose. Naming them is not a weakness of the work; it is the work.

## 10. What institutional-grade would require

CRSP-quality delisting returns; point-in-time fundamentals and GICS;
borrow availability and fees on the short book (I have started collecting
exactly this); an impact model calibrated to fills rather than a literature
constant; multi-market replication; and an order of magnitude more trials under
the same logging discipline. New trials are already machine-enforced — a
real-data run refuses to start unless it names a pre-registered, still-open
hypothesis or declares itself a reproduction.

The binding constraint was supposed to be data. It turned out to be less binding
than I thought — for one case, and instructively so. The earlier draft of this
section claimed the survivorship hole that defined trial #2 "cannot be closed for
free," on the basis that a historical *ticker*→CIK recovery lifts coverage only
73%→75% and half of its hits are reassignment-prone. That was true for ticker
matching and false for the problem: matching on the *company name* instead
(SEC's `cik-lookup-data.txt`, which follows a firm through renames to its
operating CIK) recovers ~94% of dead S&P names at ~99% correctness, and Tiingo
supplies the delisting-inclusive prices. So the quality-fundamentals hypothesis
(H1) did not wait for CRSP — it ran on free, survivorship-safe data as trial #12
(§5.5), and the answer was a clean null: the raw profitability premium was value
in disguise. I am keeping the corrected reasoning in view rather than quietly
deleting the wrong call, because being wrong about a data boundary and then
measuring past it is the honest version of research.

What that leaves is not "blocked on data" but *empirical exhaustion*, now
demonstrated rather than screened. Across everything a free-data researcher can
reach — five price-only features and three model classes (trials 1–7),
fundamental quality on a survivorship-safe universe (#12), the most-cited
alternative-data anomaly properly powered (#13), crypto funding carry in the
majors, the liquid tail, and delta-neutral cash-and-carry (#8, #10, §5.7), an
index-deletion event study (#9), and a structurally-protected closed-end-fund
discount (#11) — nothing graduates, and each null comes with a mechanism. The
remaining frontier is the one institutional-grade data actually opens and free
data cannot: CRSP delisting returns to close the last 6% survivorship residual,
point-in-time analyst estimates for post-earnings drift, intraday and
cross-venue data for the microstructure and basis trades these daily/single-venue
audits could only bound. That boundary — between what a disciplined student can
prove with free data and what needs a desk's data budget — is now mapped from
both sides, and mapping it precisely is the honest version of "what's next."

## 11. The engine room — ready for the edge I don't yet have

A research process that graduates nothing still has to answer the question a
desk asks next: when something *does* clear the bar, how do you size and run it?
I built that machinery and proved its single most important property on
synthetic ground truth, because there is no real edge to drive it. The engine
composes a multi-signal combiner, factor-neutralization, position/gross/drawdown
limits, and integer-share execution around a sizing rule that levers on the
*lower confidence bound* of the trailing Sharpe (Lo 2002), not its point
estimate. Fed a real (planted) edge it commits capital — average gross exposure
1.7, profitable net — and ramps in slowly as evidence accumulates; fed an
identically-built null it sizes to **exactly zero**. That is the trial-#11
lesson cast in code: an engine that refuses to lever an edge it is not
statistically sure of, the structural opposite of the Kelly-on-a-mirage instinct
that ruins accounts. Order submission stays in the frozen, paper-only live path —
the engine only ever produces a plan. The infrastructure is ready; the missing
piece is the edge, and finding one honestly is the rest of the project.

## 12. Conclusion

I tested thirteen hypotheses across equities, crypto, and closed-end funds and
graduated none of them to production. Read carelessly, that is failure. Read
correctly, it is the whole point: a research process is only worth anything if it
can tell a real premium from a lucky one and a robust premium from a decaying,
skewed one — and then *act on that distinction even when the number in front of
you is good*. My pipeline destroyed its own best result the moment I fixed the
universe; it found genuine crypto carry and still refused it on a deflated-Sharpe
technicality it was right to enforce; it showed a famous anomaly to be a
matched-control artifact; it overturned its own best-looking result — a
discount-reversion backtest that cleared every bar — on a one-week
implementability test; it ran the long-blocked quality hypothesis the moment a
free name-to-CIK crosswalk could finally reach dead names, and showed the premium
was value in disguise; it powered the most-cited insider-trading anomaly and
found the "opportunistic" signal no better than the routine baseline it should
beat; and it took a cash-and-carry trade whose Sharpe looked like 4.6 and proved
it a funding harvest decayed to the risk-free rate, its Sharpe blind to the tail.
The strategies failed. The judgment did not. If I am going to act as though my
analysis means something, even knowing how often analysis is luck, I would rather
it be analysis that has earned the right — and this is the evidence that it has.

## References

Bailey & López de Prado (2014), "The Deflated Sharpe Ratio"; Greenwood & Sammon
(2025), "The Disappearing Index Effect," *Journal of Finance*; Gu, Kelly & Xiu
(2020), "Empirical Asset Pricing via Machine Learning"; Harvey, Liu & Zhu
(2016), "…and the Cross-Section of Expected Returns"; Jegadeesh & Titman (1993);
Lo (2002), "The Statistics of Sharpe Ratios," *Financial Analysts Journal*;
López de Prado (2018), *Advances in Financial Machine Learning*; McLean &
Pontiff (2016), "Does Academic Research Destroy Stock Return Predictability?";
Novy-Marx & Velikov (2016), "A Taxonomy of Anomalies and Their Trading Costs";
Shumway (1997), "The Delisting Bias in CRSP Data."
