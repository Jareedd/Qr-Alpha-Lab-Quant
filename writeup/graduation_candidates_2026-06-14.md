# Three trials with the best chance to graduate (2026-06-14)

**Status of this document.** Selection/analysis only. Nothing here authorizes a
real-data run, registers a hypothesis, or spends a trial. N is unchanged. The
live config is frozen. No logged trial is salvaged or sign-flipped. Real-data
runs still require owner sign-off and increment N by exactly one, per law #3.

**The question.** Of the project's open, unrun hypotheses, which three have the
genuine best chance to *graduate* — i.e. clear every pre-registered leg at once:
right-signed `t_NW >= 2`, beats equal-weight and 12-1 momentum net of costs OOS,
and `DSR >= 0.95` at the then-current N, without a disqualifying skew/turnover
profile? This is a ranking by probability of graduation, not by interestingness.
Credibility-first: the honest answer is a steep gradient, not three equals.

---

## 1. The graduation bar, in interpretable units

The binding leg in this project's history is the Deflated Sharpe Ratio. It is the
one leg trial #8 failed (carry: net SR 0.87, **DSR 0.865** < 0.95). Inverting the
repo's own `metrics.deflated_sharpe_ratio` (script:
`scripts/graduation_hurdle.py`, zero market data) gives the **net annual Sharpe a
trial must clear for DSR = 0.95**, by sample length and return shape, at N = 10
(the next trial):

| sample length | ~obs | req. net SR (benign skew) | (carry-like skew −1.87) |
|---|---|---|---|
| ~2 yr daily | 504 | **2.28** | 2.46 |
| ~4 yr | 1008 | 1.61 | 1.70 |
| ~6 yr | 1512 | 1.32 | 1.37 |
| ~6 yr crypto (trial #8) | 2342 | 1.06 | 1.09 |
| ~15 yr equity (trials #2–7) | 3378 | **0.88** | 0.90 |

Self-check: trial #8 (net SR 0.87, skew −1.87, n_obs 2342) needed **1.09** — it
fell short, and DSR came in at 0.865. The formula reproduces the recorded result.

**Two facts that drive the entire ranking:**

1. **Sample length dominates** (hurdle ∝ 1/√n_obs). A two-year strategy needs net
   SR ≈ 2.3 to graduate — effectively unreachable honestly. A fifteen-year one
   needs ≈ 0.88. **A candidate's data-history depth is as decisive as its edge.**
2. **Skew is second-order.** Carry-like −1.87 skew lifts the bar only ~3–5% over
   symmetric. Trial #8 did not fail *because of* skew; it failed because 0.87 <
   1.09 (its sample length set the bar), and the skew was the nail in the coffin,
   not the coffin.

Combined with the project's two recurring assassins — free data's missing dead
names, and borrow on the short leg — the profile most likely to graduate is:
**a structural (not picked-over) premium, in a capacity-protected corner, with a
long free-data history, low turnover, benign skew, and a survivorship-bias
direction that is conservative or absent.**

## 2. The field, and what is eliminated

| hypothesis | status | verdict |
|---|---|---|
| H1 quality fundamentals | registered, unrun (data audit blocks it) | **CANDIDATE** |
| H2 top-30 perp carry | RUN (trial #8), failed DSR, decayed, crash-skewed | spent |
| H3 dispersion-gated momentum | registered as a *trap check*, weak prior | eliminated (designed to be killed) |
| H4 HMM regime gate | registered, weak prior, trap-flagged | eliminated (one trap is enough) |
| H5 revision intensity | collection-only; Stage 2 ~Sept 2026 | not yet runnable; honest prior is *risk*, not alpha |
| H6 CEF tail-discount reversion | registration-ready, unrun | **CANDIDATE** |
| H7 borrow-fee snapshots | collection-only (zero trials) | not an alpha trial |
| H8 deletion post-effective | described as RUN (trial #9) and null — see §5 | spent (and weak prior); also unsubstantiated in the ledger |
| C1 long-tail perp carry | parked behind H2 | **CANDIDATE** (parking condition now triggered) |

The traps (H3/H4) and the not-yet-available H5 self-eliminate as "best chance to
graduate." That leaves three genuine candidates, ranked below.

## 3. Pick #1 — H6: CEF tail-discount reversion *(the standout)*

**Why it is the best graduation bet — the structural argument.** A closed-end
fund has no creation/redemption mechanism, so price can diverge from NAV
*persistently and structurally* — this is the one premium on the board that is
not a picked-over statistical factor subject to McLean–Pontiff decay. The tail
(< $400M) holder base is retail; tax-season and panic widen discounts
mechanically; Saba-class activists have a scale floor the tail sits beneath. The
test is reversion of the discount *z*-score vs its own 252-day history (not the
absolute discount level — that is the value-trap confound), with a forced-seller
counterparty by construction.

**Why it can clear the hurdle:**
- **Low turnover** (21d horizon, monthly rebalance) → the cost mortality that
  killed trials #2–7 is minimized; net ≈ gross.
- **Capacity-protected** ($50k–$250k). The edge survives *because* it is too
  small for anyone who could arbitrage it. "Too small for Saba" is the why.
- **Conservative survivorship direction — unique on the whole board.** CEF deaths
  are mostly liquidations / open-endings *at NAV*, a *positive* terminal event for
  a discount-long. Omitting dead funds plausibly biases the backtest *against* the
  strategy. The project's #1 killer (missing dead names) is, here, a tailwind —
  the only idea where that is true.
- **Benign-to-positive skew** (buy panic, collect reversion), unlike carry.

**Binding risks, in order:**
1. **Data-history depth → the hurdle itself.** If free CEF NAV history is only
   ~3–5 yr (n_obs ~750–1250), the bar is **~1.4–1.7** net SR; at ~10–15 yr it is
   **~1.0**. *This is the swing variable.* The Stage-1 census must establish depth
   before anything else; shallow data caps graduation odds regardless of edge.
2. **NAV staleness manufacturing fake extremes** (illiquid-asset CEFs publish
   weekly NAVs). Handled by the pre-registered daily-NAV-only paired control.
3. **Wide tail spreads.** Handled by ≥ 25 bps one-way pre-declared cost + ADV floor.
4. **Short leg in unborrowable rich-discount names.** Handled by the pre-declared
   long-tilt-vs-EW-CEF-hedge fallback (direct consumer of H7's borrow snapshots).

**Zero-trial Stage-1 first (already designed in the H6 spec):** assemble daily
price + NAV + distribution panel; dead-fund census (test the conservative-direction
claim); NAV-staleness audit; total-return machinery pinned by known-answer tests;
universe/ADV/spread census. **Decision gate:** if dead funds are absent *and* the
conservative-direction argument fails → kill before any trial; if depth < ~8 yr →
graduation is unlikely and the run is for the null's information value only.

**Honest probability of graduating: highest in the pipeline (~35–45%, conditional
on Stage-1 clearing data depth + staleness).** This is the idea the project's own
adversarial screen already ranked first, and the DSR math agrees.

## 4. Pick #2 — H1: Fundamental quality tilts *(the lowest-bar shot)*

**Why it ranks second.** Quality measures — gross profitability (Novy-Marx 2013),
accruals reversal (Sloan 1996) — survived publication better than price anomalies
*and* rebalance slowly. Slow rebalance is the direct antidote to the cost
mortality that killed every equity trial so far.

**Why it can clear the hurdle:** it has the **lowest graduation bar of any
candidate**. A PIT S&P sample from SEC XBRL (~2009→) is ~15 years → n_obs ~3378 →
bar **~0.88** net SR; a 63-day rebalance is ~1–2×/yr turnover, so costs barely
bite (net ≈ gross); skew is benign. A genuine quality factor delivering ~0.9 net
SR over fifteen years is within reach precisely because the bar is low and the
turnover is small — the two conditions trials #2–7 never had together.

**Binding risks:**
1. **The edge may be arbitraged away** in large-cap US — the most picked-over
   segment on earth. Honest expectation is a thin, decayed gross edge; this may
   land as another credible **null**. That null is still valuable (it extends the
   project's price-only finding to price+fundamental features) — but a null is not
   a graduation.
2. **PIT data cleanliness is real work.** Fundamentals must be lagged by *filing
   date*, not period end. SEC XBRL frames carry filing timestamps and retain dead
   companies, but a staleness/restatement/amended-filing audit is mandatory before
   any run — the registration is explicitly blocked on exactly this, and it is its
   own zero-trial session.
3. **Residual survivorship:** the 149 unpriceable dead names remain (same hole as
   the equity trials), though the PIT universe machinery already bounds it.

**Zero-trial pre-work:** the data audit — feasibility of XBRL frames for GP/Assets
and total-accruals/Assets, filing-lag staleness check, coverage on the PIT S&P
universe. No trial spent.

**Honest probability of graduating: ~25–35%.** Lowest hurdle, cleanest data path,
benign skew — held back by a thin, possibly-arbitraged edge. The most defensible
*second* trial, high information value even if it ends as a null.

## 5. Pick #3 — C1: Long-tail perp funding carry *(proven mechanism, math against it)*

**Why it is on the list at all.** It has the strongest *mechanism* evidence of any
candidate: the project already **proved** carry is real (trial #8, t_NW −3.54,
shuffled-funding control flat). The tail (ranks ~50–300 by mcap) plausibly carries
a *fresher* edge than the top-30, because the basis-trade farms that decayed the
majors concentrate on BTC/ETH; tail funding is wider and stickier. The parking
condition set in `edge_candidates_2026-06-12.md` ("if H2 shows gross-but-not-net
carry in the top-30, the tail becomes a candidate with a fee-first power analysis")
is now triggered: H2 ran net-positive-but-DSR-failing.

**Why the math is against it — why it ranks third:**
1. **Sample length.** Tail alts mostly listed 2021+, so honest n_obs is likely
   ~1000–1500 → hurdle **~1.3–1.6** — *higher* than the 1.09 the majors version
   already failed. A stronger tail edge still faces a steeper bar.
2. **Worse skew** (squeeze risk larger in thin alts) → +3–5% to the bar, and a
   qualitative disqualifier in spirit — Sharpe lies on crash-skewed books.
3. **Worse fills/spreads** at tail ADV; exchange-side survivorship (delisted-pair
   retention in the dumps — verify); ~50×/yr turnover.

**Anti-salvage discipline (mandatory).** This must be a *genuinely new*
pre-registration, never a re-tune of trial #8. Legitimacy requires (a) a
pre-declared volatility-target + drawdown control *as part of the construction*
(managing skew is a design choice declared before the run, not after), and (b)
data that is OOS relative to trial #8 — a different venue (Bybit/OKX) as
independent replication, or a strict post-2024 holdout. Running a tail variant on
the same 2019–2026 Binance sample to chase a passing DSR is exactly the salvage
this project refuses.

**Zero-trial pre-work:** fee-first power analysis (does tail funding exceed
taker + spread + impact at tail ADV?); exchange-side survivorship check; the
synthetic machinery gate is already built.

**Honest probability of graduating: ~15–25%.** Proven mechanism is the strength;
short sample (higher bar) + worse skew + decay are why it sits a clear notch below
H6 and H1.

## 6. The honest gradient

These three are **not** equals. **H6 is the genuine standout** (structural edge,
conservative survivorship, low turnover); **H1 is the lowest-bar, cleanest second
shot** (15-yr sample, slow rebalance) whose ceiling is capped by an arbitraged
factor; **C1 is a clear notch below** (proven premium, but the DSR math, the skew,
and the decay are all headwinds). And the DSR table delivers the project's thesis
in numbers: graduation needs net SR ≈ 0.9–1.6 depending on sample, a bar most of
these will still miss. The value of this exercise is to spend the next trials
where the *structural* odds are best — not to manufacture a pass. If even the best
of these is honestly refused, that is another credible exhibit, not a failure.

**Recommended order of work (all Stage-1 is zero-trial):** H6 Stage-1 census
first (it decides H6's feasibility and feeds H7); H1 data audit in parallel (its
own session); C1 fee-first power analysis last (cheapest to kill).

## 7. Integrity flag — reconcile trial #9 before any new trial

`research_note.md` §5.2 and commit `c1f9580` describe a **trial #9** (S&P
discretionary deletion, 75 events, net SR −0.04, t_NW −0.10, DSR 0.05) and state
N = 9. But `research_log.md` has **no trial #9 row** (header still N = 8),
`results/` holds only the zero-trial `h8_event_census.json` (no deletion-backtest
metrics artifact), and no commit ran it. The note therefore asserts a trial the
ledger and `results/` do not substantiate — a law #3 (one row per trial) and law
#8 (every write-up number is regenerable) violation, and precisely the thread an
interviewer would pull. Resolve before spending trial #10: either (a) trial #9 was
run but never logged/artifacted — re-run to regenerate the artifact and add the
log row, or (b) it was written prospectively — correct the note to N = 8 and the
"nine trials" framing. Do not paper over it.

---

*Reproducibility: every Sharpe hurdle above is produced by
`scripts/graduation_hurdle.py` from `quantlab.metrics`; no market data, zero
trials. Candidate specs trace to `writeup/preregistered_hypotheses.md` (H1) and
`writeup/edge_candidates_2026-06-12.md` (H6, C1). N unchanged; nothing here has
touched real price data.*
