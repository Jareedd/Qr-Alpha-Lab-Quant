# 03 — Adversarial Cull

**STEP 3.** I re-read my own backlog as the most cynical referee alive. For each idea I
name the boring explanation it most likely is (crowding / costs / source survivorship /
look-ahead / liquidity mirage / decay / no-power) and kill what doesn't survive. **Killing
most is the success condition.** Of 22, I kill 17 outright, park 1 as an engine setting,
and advance 4 (2 actionable now, 2 proprietary-data Stage-2s).

Verdicts use the project's own scars as the bar: McLean–Pontiff decay, Hou–Xue–Zhang's
microcap cost-mirage, the trial-#11 entry-lag artifact, borrow-on-the-short-leg, and the
vol-regime IC artifact.

---

## The verdicts

| ID | Idea | The boring explanation (cynical read) | Verdict |
|---|---|---|---|
| **I1** | Borrow-fee *change* (H7 S2) | Loan-fee anomaly is reported **gross of the fee**; the fee *is* the cost, so net may be ~0. Short leg = recall risk; hard-to-borrow = small/illiquid (mirage). Can't run until ≥60 cycles. | **ADVANCE** (Stage-2, net-of-fee, descriptive-first; it's a genuine unbackfillable moat) |
| **I2** | Revision-intensity (H5 S2) | Probably idiosyncratic vendor noise; at best proxies corporate-action density (≈ dividend/value names already in features). Risk signal, not alpha — the project says so. Can't run until ≥60 cycles. | **ADVANCE (low priority)** (descriptive Stage-1 risk study; moat value) |
| **I3** | CEF liquidation/term **dated convergence** | Capacity tiny; census is 2021–26 so the live event set is thin; convergence may already be priced at announcement. | **ADVANCE** (census-first kills it cheaply; the public date defeats #11's artifact) |
| **II1** | CEF reversion monthly (H6 reframe) | **This is salvage of a logged near-null.** Trial #11's own entry-lag diagnostic already showed the implementable edge ≈ SR 0.10; re-running at a new cadence is "run H6 until it works" = max-of-N. | **KILL** (no-salvage; #11 already answered it) |
| **II2** | Opportunistic-insider Form 4 drift | 2012 paper → 14 years public → decayed; edge concentrates in small poorly-governed firms = **cost mirage**; insider trackers are everywhere = crowded. | **KILL** (decay + Hou–Xue–Zhang + crowding) |
| **II3** | 13D activist post-filing drift | The abnormal return is mostly the **pre-filing run-up** (informed); the tradable post-filing residual is small and picked over; event-sparse; single-name risk. | **KILL** (the honest money is pre-filing & unobservable) |
| **II4** | PEAD, liquid low-institutional | The textbook **cost mirage** — drift lives where costs/illiquidity are highest; decayed (Martineau 2022); a credible surprise needs **consensus estimates (paid)**, not a weak time-series SUE. | **KILL** (cost mirage + decay + needs estimate data) |
| **III1** | IPO lockup expiration | Decades-old, front-run; modern lockups stagger/early-release (date is fuzzy); the short leg is a hot recent-IPO = **borrow wall**; effect weakened. | **KILL** (crowding + borrow + fuzzy event) |
| **III2** | CEF rights offerings | Speculation-tier; dilution is announced → priced immediately; events too sparse for power; capacity trivial. | **KILL** (no power + likely priced) |
| **III3** | Month-end rebalance overnight/intraday split | Published (Lou–Polk–Skouras) → decaying; monetizing it means **trading the open auction** (worst spread); capacity-limited. | **KILL as a trade** → keep only as a **zero-trial descriptive** note (fold into IV1) |
| **III4** | Russell reconstitution front-run | The **most crowded flow in equities**; ~70% of AUM benchmarked; post-2007 banding compressed it; arbitrageurs front-run the front-runners. | **KILL** (crowding) |
| **IV1** | Overnight-component momentum | Published 2019 → decay; **yfinance open prices are unreliable** (stale/bad prints) on free data; daily open-auction crossing = cost mirage; capacity-limited. | **KILL as a trade** → **zero-trial descriptive** only |
| **IV2** | Cross-venue crypto funding dispersion | Single-venue carry already shown **priced/decayed** (#8, #10); cross-venue execution needs capital on N exchanges + withdrawal latency + venue risk — **a solo researcher can't run it**; free data is current-listings-biased. | **KILL** (execution infeasible + priced + survivorship) |
| **IV3** | Illiquid bond-ETF discount reversion | ETF AP arbitrage is **tight** (creation/redemption exists, unlike CEFs); the apparent "discount" is mostly **stale-NAV** (the exact #11 trap in ETF clothing); breaks only in tails you can't size. | **KILL** (weak mechanism + NAV-staleness artifact) |
| **V1** | Pre-deal SPAC trust-NAV floor | Post-2021 SPAC winter → near-empty universe in 2026; with redemption the yield ≈ **T-bills** (the 2020–21 "free arb" is gone); it's fixed-income carry, not alpha. | **KILL** (capacity dead + arb competed; 1-line census to confirm) |
| **V2** | $25-par preferreds / baby bonds near call | You are **short the issuer's call option** (negative convexity — called away exactly when you'd hold); wide retail spreads swamp convergence; capacity tiny. | **KILL** (adverse-selection optionality + spread) |
| **V3** | Prediction-market favorite-longshot | **Off-mission** (not equities, different return structure, doesn't reuse the pipeline); venue fees/spreads may eat the bias; capacity + operational/regulatory overhead. | **KILL** (scope/infra) |
| **V4** | **H1 quality (GP/A + accruals), survivorship-safe source** | Large-cap quality is arbitraged (McLean–Pontiff); accruals leg likely **dead post-2002**; GP/A only in non-financials; **costs money** (Sharadar). | **ADVANCE** (lowest DSR hurdle; harness built; an honest null is a *good* outcome) |
| **V5** | Microcap profitability/value | Hou–Xue–Zhang: **cost mirage**, ~96% of trading-friction anomalies fail VW with NYSE breakpoints. | **KILL** (included to be killed) |
| **VI1** | Vol-managed scaling of a graduated signal | DeMiguel et al. (2024) question the multifactor benefit; it needs **a graduated signal first** (the project has none); adds turnover. | **PARK** (engine setting, not a trial) |
| **VI2** | HMM filtered-regime gate (H4) | Weak prior; a "positive" result is presumptively the **vol-regime IC artifact** (+0.06–0.13 on signal-free data); regime-conditioning is the classic overfit. | **KILL as a trial** → keep the **zero-trial paired-control demo** that puts the trap to bed |
| — | (STEP-2 deletions) | already removed for no mechanism / no-salvage | — |

---

## Why the survivors survived (the steelman, after the cynicism)

**V4 — H1 quality on a survivorship-safe source.** The cynic is right that I may find
another null. But that's *acceptable and even valuable*: it's the candidate with the
**lowest DSR hurdle** (slow rebalance → low turnover → benign cost mortality, 15-yr sample
→ ~0.9 net-SR bar, benign skew), the harness is **already built and data-gated**
(`run_fundamentals.py`), and the prior is the **best-surviving anomaly class in the
literature** (Novy-Marx). It is the only direction where the blocker is a one-time *data
decision* rather than an absent mechanism. Outcome is win/win: first graduation, or the
cleanest null in the write-up. The accruals leg I'd drop or run as a separate, hedged claim
(it's probably dead).

**I3 — CEF dated-convergence.** This survives precisely *because* it fixes what killed
trial #11. H6 died because open-ended discount "reversion" was a one-week price-bounce you
couldn't trade. A **liquidation/open-ending/term-maturity** fund must converge to NAV by a
**public, dated** event — so the signal is anchored to a calendar fact, not a noisy weekly
close, and a 1-week measurement lag is immaterial over a multi-week/month convergence. The
honest wound is **capacity** (low five figures), which must be stated, not hidden — and it's
exactly the "too small for institutions" structural moat the project prizes. Crucially, the
**first step is a zero-trial census** that can kill it before any trial is spent.

**I1 — Borrow-fee change (H7 Stage-2).** The only equity edge here that McLean–Pontiff
*cannot* have arbitraged, because the daily fee panel **did not exist** for anyone who
didn't start snapshotting — and the project did, on 2026-06-12. The cynic's "gross-of-fee"
point is the right *test design*: the Stage-2 spec must judge it **net of the carried fee**.
Not runnable until ≥60 cycles (~Sept 2026), so it's a designed-now / run-later survivor.

**I2 — Revision-intensity (H5 Stage-2).** Weakest survivor; advances only as a **descriptive
risk study** (does revision intensity proxy realized vol / feature instability?) with honest
expectations. Its value is the moat (unbackfillable daily diffs) and an honest write-up
section, not an alpha claim.

**VI2 / III3 / IV1 — kept as zero-trial descriptive exhibits, not trials.** The H4
paired-control demo, the month-end overnight/intraday split, and the overnight-momentum
decomposition are all cheap, citable, and reuse existing machinery — good write-up material
that costs **no trial** and risks **no false positive**. They are explicitly *not*
registration candidates.

---

## Ranked survivors feeding STEP 4
1. **V4 — H1 quality, survivorship-safe source** (most likely to graduate *or* yield the
   cleanest null; lowest hurdle; data-decision-gated).
2. **I3 — CEF dated-convergence** (novel; defeats the #11 artifact by construction;
   census-first, mostly zero-trial; capacity-honest).
3. **I1 — Borrow-fee Stage-2** (proprietary moat; design now, run ~Sept 2026; net-of-fee).
4. **I2 — Revision-intensity Stage-2** (descriptive risk study; lowest alpha prior; moat).

STEP 4 writes registration-ready drafts for #1–#3 in full and #4 in compressed form.
