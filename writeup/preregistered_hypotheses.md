# Pre-registered hypotheses (Phase 8+ candidates)

The Phase 4 verdict was that the five standard price-only features carry
no defensible edge on the honest universe — and that *new alpha
hypotheses get their own pre-registered trials*, not salvage runs. This
file is where that happens.

**Protocol.** A hypothesis is registered here — with its exact config and
success criteria — BEFORE the run. The run happens once, increments N,
and the outcome is logged whatever it says. Editing a registration after
seeing results is prohibited; a revised idea is a new registration. This
is the difference between testing seven hypotheses and running one
hypothesis seven times until it works.

Template:

```
### H<n>: <one-line hypothesis>
- Status: PROPOSED | RUN (trial #k) | ABANDONED (why)
- Economic prior: why would this be priced? who is on the other side?
- Point-in-time safety: one-line argument per new feature/label
- Exact config: data, features, label, horizon, model, neutralization, costs
- Success criteria (set BEFORE the run): e.g. t_NW ≥ +2 AND DSR ≥ 0.95 at
  the then-current N, net of costs, beats both baselines
- Failure interpretation: what we conclude if it fails
```

---

### H1: Fundamental quality tilts (profitability/accruals) carry cross-sectional signal where price features did not
- Status: PROPOSED — blocked on a data-source decision
- Economic prior: quality measures (gross profitability, accrual
  reversal) have survived publication better than price anomalies
  (Novy-Marx 2013); they rebalance slowly, so cost mortality is low —
  exactly the failure mode that killed trials 2–7.
- Point-in-time safety: the hard part. Fundamentals must be lagged by
  filing date, not period end (a ≥3-month report lag as a crude floor).
  Free sources (SEC XBRL frames, FMP free tier) need an explicit
  staleness audit before any run. **This hypothesis cannot run until the
  data question is answered honestly; that audit is its own session.**
- Exact config: PIT sp500, ridge, h=63d, rebalance 63d (quality is slow),
  features = {GP/A, total accruals/A}, sector demean + beta projection,
  10 bps.
- Success criteria: t_NW ≥ +2 and DSR ≥ 0.95 at then-current N, net SR >
  both baselines.
- Failure interpretation: large-cap US quality is also arbitraged away at
  free-data fidelity; the write-up's conclusion extends from price-only
  to price+fundamental features.

### H2: The same pipeline finds (or honestly rejects) carry in crypto perpetuals, where survivorship bias and dead-name gaps do not exist
- Status: **RUN (trial #8, 2026-06-13). Outcome: registered criteria NOT MET
  — but the first NON-NULL the project has produced.** Net SR +0.87,
  IC t_NW −3.54 (correctly signed), funding & price P&L both positive,
  shuffled-funding control flat, survives ex-top-3. The single failed leg
  is the pre-registered DSR ≥ 0.95 (got 0.865), and the return is
  crash-skewed (−1.87, −74% maxDD). Adversarial diagnostics confirmed it
  is a REAL edge, not an artifact (entry-lag decays gracefully → no timing
  leak; only 2/28 dead names held at death → no delisting optimism) — but
  it has DECAYED from SR 2.28 (2020–21) to ~0.4 post-2021 as basis farms
  scaled (McLean–Pontiff in a second asset class). Criteria were NOT
  relaxed post-hoc. Full record: research_log.md trial #8,
  `results/metrics_h2_carry.json`, `results/h2_carry_diagnostics.json`,
  `writeup/h2_carry_design.md`. (Was: PROPOSED.)
- Economic prior: funding-rate carry in perps is a payment for taking the
  crowded side; it is directly observable, not estimated, and the
  CLAUDE.md stretch goal. Exchange data is complete (no delisting-return
  problem — delisted contracts have full terminal histories).
- Point-in-time safety: funding rates and prices are timestamped exchange
  records; features use funding through t only.
- Exact config: top-30 perps by ADV (point-in-time, by listing date),
  feature = trailing funding rate percentile, h=7d, dollar-neutral
  cross-sectional rank portfolio, taker fees + realistic spread, BTC-beta
  projection.
- Success criteria: t_NW ≥ +2 and DSR ≥ 0.95 at then-current N net of
  fees, and the result must survive excluding the top-3 names.
- Failure interpretation: carry is consumed by fees/crowding at our
  fidelity — still a publishable section ("the pipeline generalizes;
  the free lunch does not").
- **Amendment, 2026-06-12 — PRE-DATA (no exchange data has been
  downloaded as of this edit; full design in `writeup/h2_carry_design.md`):**
  1. **Label corrected to funding-INCLUSIVE total return**
     (`mark_return − funding` for a long). The premium lives in the
     funding flows; a price-only label measures the wrong object —
     demonstrated in the synthetic lab, where the same planted-carry book
     scores ≈ +1 funding-inclusive and ≈ −2 price-only
     (`tests/test_synthetic_carry.py`).
  2. **Universe corrected: perps are NOT survivorship-free.** PIT top-30
     by trailing 30d dollar volume among contracts listed at t, with
     delisted contracts (enumerable from the exchange's own public dumps,
     full terminal histories) included through their final day.
  3. **Timestamp convention fixed:** features use only funding SETTLED at
     or before the decision time (8h settlements; trailing 21-settlement
     mean); labels accrue settlements inside the holding window only.
  4. **Machinery gate added:** `synthetic.make_perp_panel` planted_carry
     must be recovered and priced_carry (funding fully offset by drift —
     the true null) rejected, in the same environment, immediately before
     the real run.
  5. **Paired controls registered:** cross-sectionally shuffled-funding
     book must earn ~nothing; funding-income vs price-drag decomposition
     reported; result must not be one regime/period/name.
  6. Costs pinned: taker 5 bps/side + spread 2 bps/side + sqrt impact on
     perp dollar-ADV; weekly rebalance; quartile book on the top-30
     universe.
  Status remains PROPOSED; the run is trial #8 and awaits owner sign-off.

### H3: Momentum works conditionally — only in low-cross-sectional-vol regimes
- Status: PROPOSED — registered partly as a TRAP CHECK
- Economic prior: weak (momentum crashes cluster in high-vol reversals,
  Daniel–Moskowitz 2016) — but regime-conditioning is also the classic
  overfitting move, which is why the criteria are strict.
- Point-in-time safety: regime indicator = trailing 63d cross-sectional
  dispersion, known at t.
- Exact config: trial #2's exact config, weights scaled by the regime
  indicator's trailing percentile (no new fitted parameters — a fixed,
  pre-declared mapping).
- Success criteria: t_NW ≥ +2, DSR ≥ 0.95, AND the unconditional result
  must remain null (if everything improves, suspect leakage first, per
  the prime directive).
- Failure interpretation: regime gating on free daily data is noise; do
  not iterate on the mapping (that is mining with extra steps).

### H4: A causal volatility-regime gate (filtered 2-state HMM) improves the momentum baseline where the fixed dispersion gate (H3) may not
- Status: PROPOSED — machinery built and falsification-validated on
  synthetic data 2026-06-12 (`quantlab/regime.py`, `planted_regime` mode,
  `tests/test_regime.py`); REAL-DATA RUN NOT PERFORMED, awaiting owner
  sign-off. A separate registration from H3 on purpose: H3's fixed
  dispersion mapping stays as registered (editing it post-hoc is
  prohibited); this is a different detector and counts as its own trial.
- Economic prior: same as H3 (Daniel–Moskowitz momentum crashes cluster
  in high-vol regimes) — weak, trap-flagged. The HMM adds persistence
  modeling, not new economics; if H3 and H4 disagree, the disagreement
  itself is evidence of mining.
- Point-in-time safety: the gate is the forward-FILTERED P(calm | market
  returns through t), parameters re-fit on an expanding PAST-only window
  (`causal_regime_probs`). The smoothed (forward-backward) probabilities
  — what off-the-shelf HMM libraries return — condition on the future;
  tests pin both that our filter cannot see the future (perturbing the
  future moves nothing in the past) and that the smoothed version DOES
  (the leak is demonstrated, not asserted). The smoothed output is never
  exposed to strategy code.
- Exact config: trial #2's exact config; rebalance weights scaled by
  filtered P(calm) at each rebalance (fixed mapping, no tuned threshold);
  HMM refit_every=63, min_train=504, on the member-masked equal-weight
  market return.
- Success criteria: t_NW ≥ +2 and DSR ≥ 0.95 at then-current N, net of
  costs; unconditional momentum must remain null; AND the paired
  artifact control must pass — the same gate applied to a label-shuffled
  control must show no lift (the synthetic lab showed residualization ×
  vol-regime interactions can manufacture conditional IC of +0.06 to
  +0.13 from nothing; a real-data result that fails the paired control
  is the artifact, not alpha).
- Failure interpretation: regime gating on free daily data is noise
  regardless of detector sophistication; H3 need not be run separately
  (one trap is enough).

### H5: Vendor data-revision intensity is cross-sectionally informative (the dataset nobody else has)
- Status: PROPOSED — two-stage registration. Stage 1 (now): the idea is
  timestamped BEFORE any data exists to peek at; collection of per-cycle
  revision fingerprints (`results/live/revisions_*.json`, started
  2026-06-11) is fully automated and untouched by hand. Stage 2: when
  ≥ 60 cycles have accumulated (~September 2026), the exact test config
  and success criteria are registered in this file BEFORE the first
  analysis run. No analysis of the revision data is permitted before
  Stage 2 is written down.
- Economic prior: return-cell revisions cluster around dividends,
  splits and corrections — i.e., corporate-action density — and flag
  names whose historical record is least stable. Plausible links: (a)
  realized-vol differences, (b) feature reliability (a name whose past
  keeps changing has noisier features, arguing for down-weighting), (c)
  at minimum a data-quality risk control for the live book. The honest
  prior is that this is a RISK/quality signal, not an alpha signal.
- Point-in-time safety: trivially safe — the fingerprint at t is a
  function of downloads made at t and t−1, both timestamped by CI
  commits; it cannot reference anything later.
- Why this qualifies as an edge at all: the dataset is proprietary by
  construction (daily diffs of a vendor's claimed history exist only if
  someone snapshots daily — we started 2026-06-11 and commit the
  fingerprints publicly, making the record verifiable). Nobody can
  backfill it.
- Failure interpretation: revision intensity is idiosyncratic vendor
  noise with no cross-sectional structure — still a write-up section,
  because measuring it is the only way anyone finds out.

### H6: Closed-end-fund discounts mean-revert from extremes in the small-fund tail, net of costs
- Status: **RUN (trial #11, 2026-06-15). Outcome: registered criteria nominally
  MET at the registered config (net SR 1.11, DSR 0.999, IC t_NW −10.4, beats EW
  0.64, positive skew +0.80, −6% maxDD) — BUT adversarial diagnostics OVERTURN
  it.** The entry-lag sweep collapses the Sharpe **1.11 → 0.10 at a one-week lag**
  (then negative): the entire "edge" is a single-week bounce, not multi-week
  reversion — a microstructure / shared-price (bid-ask-bounce) reversal artifact
  (the discount uses price_w and the next return divides by price_w; you can't
  trade the noisy close you measured). The shuffle control is also seed-fragile
  (breaches 0.3 on 3/6 seeds, max 0.52; the registered seed drew 0.277 by luck).
  The implementable (1-week-lag) edge is **~null (SR 0.10)**. **H6 does NOT
  graduate.** No criteria relaxation. Lesson: the frozen criteria lacked an
  entry-lag / implementability gate (trial #8 ran it as a diagnostic; #11 shows
  it must be a registered criterion for any reversion strategy). Full record:
  research_log.md trial #11, `results/metrics_h6_reversion.json`,
  `results/h6_reversion_diagnostics.json`. N=11. (Was: PROPOSED.) Origin spec:
  `writeup/edge_candidates_2026-06-12.md` §H6.
- Economic prior: a closed-end fund has no creation/redemption mechanism, so
  price can diverge from NAV persistently and structurally — not a picked-over
  statistical factor subject to McLean–Pontiff decay. The sub-$400M tail's
  holder base is retail; tax-season and panic selling widen discounts
  mechanically; Saba-class activists have a scale floor the tail sits beneath.
  The claim is reversion of the discount **z-score vs its own history** (not the
  absolute discount level — that is the value-trap confound). Counterparty: the
  retail panic seller and the absent arbitrageur. Capacity is **$50k–$250k** —
  "too small for Saba" is the why-it-exists, stated plainly.
- **What Stage-1 settled (zero-trial, before this registration):**
  1. A tradable tail exists: **185 funds** (mcap < $400M, dollar-ADV ≥ $250k);
     median discount −7.6%, 101 funds at < −10% (`cef_stage1_census.json`).
  2. NAV staleness is minor: **95% publish daily NAV** (lag ≤ 1d); the
     daily-NAV-only filter retains almost the whole universe.
  3. **Survivorship direction is CONSERVATIVE** (the make-or-break): the
     dead-fund census (SEC EDGAR, 2021–2026, 151 CEF/BDC deaths) found **94%
     are NAV events** (liquidation/merger/open-end/term) and **zero distress
     delistings**. CEF deaths happen at ~NAV, so a current-funds-only backtest
     OMITS winners-at-NAV → it is biased *against* a discount-long, a
     conservative LOWER bound. This is the one idea where missing dead names is
     a tailwind (`cef_dead_fund_census.json`).
  4. Data cadence (constrains this registration): free NAV/discount history is
     **weekly** beyond the trailing year (CEFConnect `All`), back to ~2012;
     distribution-inclusive total-return price is daily via yfinance.
- Exact config (FROZEN):
  - Universe: current CEFConnect funds with mcap < $400M, dollar-ADV ≥ $250k,
    **CEF-only** (CEFConnect lists CEFs; any BDC-category names excluded — BDC
    deaths can distress, unlike CEFs, per the census). Stale-NAV funds are KEPT
    in the main run and isolated by the NAV-staleness control below (refined
    pre-run, before any result is seen, for control coherence — filtering
    daily-NAV upfront would make that control trivially identical to the main
    run). Current-listings-only, accepted as a conservative lower bound per
    Stage-1's survivorship finding (direction measured, and against us).
  - Signal: per fund, `z = (discount − mean_52w) / std_52w` on the **weekly**
    discount series (past-only; ≥ 26w min history), discount = (price − NAV)/NAV
    via `quantlab.cef.discount`.
  - Book: dollar-neutral equal-weight quintiles — **LONG the most-negative-z**
    (widest-discount extreme, expected to revert up), **SHORT the most-positive-z**
    (richest), rebalanced every **4 weeks**, held between.
  - Label / P&L: **weekly distribution-inclusive total return** (yfinance
    adjusted close resampled W-FRI). **Cadence is weekly by design** — the
    signal's information arrives weekly, so weekly is the honest observation
    frequency; this sets n_obs ≈ #weeks and a DSR hurdle of ~1.4 net SR, HIGHER
    than the daily-cadence ~1.1 the graduation doc assumed before Stage-1
    revealed the data is weekly. Choosing the harder, honest hurdle is the point.
  - Costs: **25 bps one-way** on turnover (linear headline; tail spreads are
    wide so 25 bps is the conservative floor — Stage-1 found nothing forcing it
    higher; sqrt-impact is the capacity dimension, reported separately).
  - DSR at **N = 11**.
- Point-in-time safety: discount at week w uses NAV/price published at or before
  w; the z uses a trailing past-only window; weights formed at w earn from w+1.
- Machinery gate (MUST pass in-env immediately before the real run, law #4):
  a synthetic `planted_reversion` CEF world (discounts mean-revert) must be
  recovered and a `random_walk` discount world (no reversion — the true null)
  rejected, as a paired per-seed differential. If the harness cannot tell
  reversion from its absence today, no real number is trusted — ABORT, no trial.
- Registered paired controls (all reported in the single run, not extra trials):
  1. **NAV-staleness control**: re-run on the daily-NAV-only subuniverse; the
     effect must persist (stale NAVs manufacture fake extremes).
  2. **Label-shuffle control**: forward returns permuted across funds within
     each week must earn ~nothing (|SR| < 0.3).
  3. **Seasonality subreport**: SR with and without Dec–Jan (tax-loss window),
     pre-declared, reported not optimized.
- Success criteria (FROZEN — met only if ALL hold): right-signed reversion IC
  with **|t_NW| ≥ 2** AND **net SR > 0** beating an equal-weight-CEF baseline net
  of costs AND **DSR ≥ 0.95 at N=11** AND survives removing the largest-mcap
  decile AND both paired controls pass.
- Kill criteria: machinery gate fails → ABORT (no trial spent); shuffle control
  earns (|SR| ≥ 0.3) or staleness control fails → artifact, logged as such;
  |t_NW| < 2 or wrong-signed → null; DSR < 0.95 → does not graduate (logged like
  trial #8, **NOT relaxed**). NO post-hoc z-window/quantile/holding-period/cost
  scans — each is +1 trial and none is authorized by this registration.
- Failure interpretation: a real-but-DSR-failing or null result is still a
  write-up section ("the one structurally-protected premium, tested honestly on
  free data, and the discipline's verdict") — citable next to the carry trial.

### H7: Daily borrow-fee/short-availability snapshots — collection-only (zero trials)
- Status: REGISTERED 2026-06-12 with owner sign-off, COLLECTION-ONLY —
  the H5 two-stage structure, second instance. This registration
  authorizes data COLLECTION only; it does not authorize any analysis
  and does not increment N. (Originated as candidate H7 in
  `writeup/edge_candidates_2026-06-12.md`, where 17 sibling ideas were
  killed on the record.)
- Economic prior: the borrow fee is the observable price of concentrated
  negative information (the shorting-premium literature is gross of fee;
  net-of-fee tradability is ambiguous — which makes this a dataset, not
  a trade). Honest uses for THIS book: (a) a risk veto for any future
  short leg; (b) a candidate feature for future registrations; (c) an
  unbackfillable public dataset — the moat. Nobody can reconstruct
  yesterday's borrow market tomorrow.
- Source, VERIFIED at registration time: IBKR public short-stock file
  (ftp2.interactivebrokers.com, user 'shortstock', usa.txt; ~20k
  instruments, pipe-delimited, carries its own '#BOF' timestamp).
- Mechanics: non-fatal step in the live cron after each trading cycle
  (`scripts/collect_borrow.py` → `results/live/borrow_{asof}.json`,
  write-once, committed with the prediction logs). Universe = the live
  experiment's own scored cross-section (latest predictions/weights),
  plus whole-file aggregates so format drift is visible. The live
  trading path (`live.py`, deployed config) is untouched.
- Point-in-time safety: trivial — snapshot at t, committed by CI at t;
  the public commit history is the verifiable timestamp.
- Stage 2 condition: ≥ 60 cycles accumulated; the exact analysis config
  and success criteria are registered HERE before the first look. No
  analysis of any kind before Stage 2 is written down.
- Operational success criterion (Stage 1): snapshots present for ≥ 95%
  of trading cycles; schema stable; zero trading-cycle failures
  attributable to the collector.
- Kill criteria: source becomes unreliable or terms-blocked → stop
  collection, log the attempt; the outcome still costs nothing and is a
  write-up sentence about data moats.

### H8: Discretionary S&P 500 deletions earn positive post-effective returns vs a matched control
- Status: **RUN (trial #9, 2026-06-13). Outcome: clean NULL — criteria not
  met.** Daily event-time portfolio net SR −0.04, t_NW −0.10, DSR 0.05
  (75 usable events). Deleted names rebound (~+4.8%/60d) but a
  size+momentum-matched control rebounds +2.6% of it; the +2.2% residual
  is insignificant (t 0.87) and negative pre-2015 — small-loser mean
  reversion, not an index-deletion effect. Greenwood–Sammon's disappeared
  effect reproduced in the post-effective window. Synthetic planted-event
  gate passed first (drift recovered, null rejected), so this is genuine
  absence, not harness impotence. Full record: research_log.md trial #9,
  `results/metrics_h8_events.json`. (Was: PROPOSED.) Power gate had passed
  at zero cost: 124 discretionary deletions since 2010 (`results/h8_event_census.json`).
- Economic prior (weak, declared): index trackers sell at the effective
  close with a tracking-error loss function; if residual overshoot exists
  post-arbitrage it reverts. Counter-prior: Greenwood–Sammon (2025) find
  the announcement→effective deletion effect decayed to ~0.1% for
  2010–2020. This tests the strictly different POST-effective window, with
  an expectation tilted toward a clean, citable null — bought for its
  information value either way.
- Construction: enter the close of effective-date + 1 (avoids the
  depressed forced-flow close), hold 60 trading days, long the deleted
  name vs a SHORT matched-control basket — the hedge IS the control:
  deleted names are mechanically small recent losers, so any rebound must
  beat a basket matched on size and trailing return, not zero. Daily
  event-time portfolio (overlapping events averaged) feeds SR/DSR/t_NW;
  10 bps/side cost per event.
- Point-in-time safety: effective dates are exchange facts; entry t+1 uses
  only data public at t; matching features use data through t only; the
  matched pool is index membership at t.
- **Amendment, 2026-06-13 (PRE-RUN): size matched by log trailing-63d
  dollar volume**, not log-mcap — free point-in-time market cap is
  unavailable; dollar volume is a standard PIT-computable size/liquidity
  proxy. Declared before the run, per law #5 (not a post-hoc change).
- Machinery gate (passed before the real run): synthetic planted post-
  event drift recovered (positive excess + Sharpe), null rejected
  one-sided (no manufactured rebound) — `tests/test_events.py`.
- Success criteria (frozen): right-signed daily-portfolio t_NW ≥ +2 vs the
  matched control AND net SR with DSR ≥ 0.95 at N=9 AND the effect is not
  concentrated in 2010–2014 (pre-declared subperiod report).
- Kill criteria: control rebounds comparably (it was a small-loser effect
  in an index costume); t_NW < 2 → null, logged. NO entry-timing or
  holding-period variants afterward (each is +1 trial, none authorized).
- Failure interpretation: a free replication-with-control of a famous
  disappeared anomaly — citable next to trial #2's survivorship exhibit
  as the project's second in-house reproduction of the published record.

### H9: Long-tail perp funding carry — does the carry premium survive in the liquid tail (ADV ranks 31–150) beneath the majors, where funding is far wider but fills worse?
- Status: **RUN (trial #10, 2026-06-14) — clean NULL, criteria not met** (net SR
  −0.13, gross 0.26, IC t_NW −3.62 correctly signed, DSR 0.024; funding P&L +1.23
  vs price P&L −0.85 → the tail carry is largely PRICED and 20 bps fills finish
  it; shuffled control flat at −0.28; machinery gate passed → genuine economic
  null; `results/metrics_h9_carry_tail.json`). Was PROPOSED as the C1 candidate
  from `writeup/edge_candidates_2026-06-12.md`,
  parked behind H2 with an explicit unparking condition: "if H2 shows
  gross-but-not-net carry in the top-30, the tail version becomes a candidate
  with a fee-first power analysis." H2 (trial #8) ran net-positive-but-DSR-failing
  → the condition is triggered. The **fee-first power analysis ran FIRST** (zero
  trials, `scripts/carry_tail_power.py`): tail = ADV ranks 31–150 has ~2081
  usable days (~8 yr) since 2020-09, median 120 names/day, gross funding spread
  **71.8%/yr** (median 42%) — 10–29× any realistic tail cost. Power exists;
  this is necessary, NOT sufficient (a funding-only spread ignores the
  price-drift offset and the crash skew). **NOT a salvage of #8:** a DISJOINT
  universe (the majors are excluded), pre-declared before the run, judged at the
  SAME bar with NO relaxation.
- Economic prior: carry is proven real in this exact pipeline (trial #8). The
  basis-trade farms that decayed the majors (Ethena-style) concentrate on
  BTC/ETH; the tail is too small and venue-risky for them, so the tail premium
  may be LESS decayed and is structurally wider (the power analysis confirms the
  gross spread). Counter-prior (why it may still fail our bar): in the tail the
  price drift may offset funding more completely (the priced-carry null), fills
  are worse (pre-declared 20 bps/side), and the crash skew is plausibly MORE
  severe than the majors' −1.87 — exactly the failure mode that sank #8 on DSR.
- Point-in-time safety: identical to H2 — funding rates and prices are
  timestamped exchange records; the signal uses funding settled through t only;
  the universe uses trailing ADV through t; delisted contracts fall out when
  their data ends (no survivorship; the cache holds 729 contracts ever, incl.
  delisted).
- Exact config (frozen): PIT tail universe = symbols whose trailing-30d
  dollar-volume rank is in [31, 150] among contracts trading at t (min 20 names
  to form quartiles); signal = trailing-7d mean daily funding; label =
  funding-INCLUSIVE total return (mark_return − funding); book = dollar-neutral
  equal-weight quartiles (SHORT top-funding quartile, LONG bottom), weekly
  rebalance; costs = **20 bps/side** (5 taker + 15 spread, conservative for the
  tail; linear headline, sqrt impact is the capacity dimension); `--n-trials 10`.
  Registered paired control = cross-sectionally shuffled funding (must earn ~0).
  Reported: funding-income vs price-drift decomposition, skew, maxDD,
  ex-top-3-by-weight robustness, and a 2020–21 vs 2022+ subperiod split (the
  decay check). Machinery gate (synthetic planted_carry recovered / priced_carry
  rejected, paired) runs in-env immediately before the real run.
- Success criteria (frozen, SAME as H2 — no relaxation): right-signed IC
  t_NW ≤ −2 AND net SR > 0 AND DSR ≥ 0.95 at N=10 AND survives ex-top-3 AND
  |shuffled-funding control SR| < 0.3. The skew-aware DSR is the judge precisely
  because crash skew is this trade's known weakness ("Sharpe lies on negatively
  -skewed books").
- Kill criteria: machinery gate fails → ABORT, no trial spent; shuffled control
  earns (|SR| ≥ 0.3) → artifact, not carry; t_NW wrong-signed or > −2 → null;
  DSR < 0.95 → does not graduate (logged like #8, NOT relaxed). NO post-hoc
  rank-band scans, cost re-tuning, holding-period or signal-lookback variants —
  each is +1 trial and none is authorized by this registration.
- Failure interpretation: real-but-DSR-failing/crash-skewed → "the tail carry
  premium is real but the discipline refuses it too," McLean–Pontiff in a third
  cut; outright null → "the tail carry is fully priced / consumed by tail fills."
  Both are write-up sections; neither is hidden.

---

Run log: H2 RUN as trial #8 (2026-06-13, criteria not met). H8 RUN as trial #9
(2026-06-13, clean null). H9 RUN as trial #10 (2026-06-14, clean null — tail
carry priced). H6 RUN as trial #11 (2026-06-15, criteria nominally met but
OVERTURNED by entry-lag diagnostic — microstructure reversal artifact, does not
graduate). **N = 11.** Each run requires owner sign-off, increments N by
exactly 1, and is logged in `research_log.md` regardless of outcome. H5/H7 are
collection-only/two-stage and exempt from the run/N language until their Stage-2
analysis is registered.
