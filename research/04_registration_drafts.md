# 04 — Registration Drafts (for owner review — NOT to be merged)

**STEP 4.** Registration-ready specs for the 4 cull survivors, in the repo's
`preregistered_hypotheses.md` format. **These are DRAFTS for your review. I have not
written them into `preregistered_hypotheses.md`, and nothing here has been run.** Draft
IDs (DA–DD) are placeholders; you assign the real H-number on registration.

### DSR hurdle table — computed at the CORRECT incremented N (next trial = **#12**, since N=11)
Pure-arithmetic replication of `quantlab.metrics.expected_max_sharpe` + PSR inversion
(stdlib only, no market data, no trial). **Validated**: reproduces the
`graduation_candidates_2026-06-14` N=10 hurdles exactly (0.880/1.057/1.316/2.284).

| Cadence / sample | n_obs | DSR≥0.95 hurdle @ **N=12** (net annual SR) |
|---|---|---|
| Daily, ~15 yr (S&P PIT) | 3378 | **0.90** (0.90–0.91 even at skew −0.5) |
| Daily, ~10 yr | 2520 | 1.05 |
| Daily, ~6 yr | 1512 | 1.35 |
| Weekly, ~16 yr | 834 | **0.83** |
| Weekly, ~10 yr | 520 | 1.05 |
| Weekly, ~6 yr | 312 | 1.36 |

*Cost of the next trial:* N=11→12 lifts the 15-yr daily hurdle only 0.893→0.904 (≈+0.01
SR). Sample length dominates; the N-increment is second-order. Skew is third-order for
benign books (the carry trades' −1.87 skew mattered; these survivors don't carry it).

**Every draft inherits these standing controls** (from the audit + cull):
- **Value-weighted, NYSE-breakpoint-equivalent liquid universe** (Hou–Xue–Zhang guillotine
  — no equal-weight microcap mirages).
- **Machinery gate first** (synthetic planted vs. true-null, paired, in-env) — ABORT, no
  trial, if the harness can't tell signal from its absence today.
- **Entry-lag / implementability gate as a PASS/FAIL criterion** for any reversion or
  cross-sectional-mean idea (the trial-#11 lesson).
- **Vol-regime artifact control** where any regime/vol conditioning enters (the +0.06–0.13
  signal-free IC documented 2026-06-12).
- **Costs netted, turnover reported, no gross-only numbers.**

---

## DA — Quality tilts (gross profitability + accruals) on a survivorship-safe source
*(refines the existing H1; the only change vs. the live registration is making the data
gate and the non-financials/accruals splits explicit, and fixing the DSR at N=12)*

- **Status:** DRAFT — PROPOSED, **data-source-gated**. Harness already built/tested/gated
  (`fundamentals.py`, `fundamentals_data.py`, `make_quality_panel`, `run_fundamentals.py`);
  the free SEC path is already correctly refused by the DATA GATE (~73–75% coverage).
- **Hypothesis:** Gross profitability (GP/A, non-financials) carries right-signed
  cross-sectional return predictability in liquid US large caps where price-only features
  did not — **because it rebalances slowly, so cost mortality (the killer of trials #2–7)
  is low.**
- **Exact testable prediction:** On a survivorship-safe PIT large-cap universe, a
  `z(GP/A)` quintile long-short (non-financials), 63-day horizon, 63-day rebalance,
  sector+beta neutral, 10 bps/side, earns IC `t_NW ≥ +2` (right-signed) and **net annual
  SR ≥ 0.90** (the N=12, ~15-yr daily DSR hurdle), beating both baselines.
- **Universe + dates:** S&P 500 PIT membership **or** Sharadar large-cap (mcap-ranked,
  liquid), 2005→2026 (~15 yr → n_obs ≈ 3378 daily). **GP/A restricted to non-financials**
  (exclude GICS Financials + Real Estate — they have no COGS line, per Novy-Marx). Value-weighted.
- **Label definition:** forward 63-day return, cross-sectionally z-scored (`build_labels`,
  horizon=63). Fundamentals lagged by **filing date** (not period end), TTM/10-K-only
  flows over point-in-time Assets (the audit's annualization refinement).
- **Cheapest kill test (ZERO-TRIAL, already half-built):**
  1. **Data gate:** does a reachable source give **≥90% survivorship-safe coverage** of the
     PIT universe (mapped dead names included)? Free SEC = 73–75% → **REFUSE, no trial**
     (already enforced). Only proceed on Sharadar/Compustat/CRSP.
  2. **Machinery gate:** `make_quality_panel` `planted_quality` recovered (SR≫0) and
     `null_quality` rejected (~0), paired, in-env.
- **Success criteria (frozen):** `t_NW ≥ +2` right-signed **AND** `DSR ≥ 0.95 at N=12`
  (≈ 0.90 net annual SR @ ~15 yr) **AND** net SR > equal-weight and > 12-1 momentum, net
  of costs **AND** survivorship-safe coverage ≥ 90%.
- **Kill criteria:** machinery gate fails → ABORT (no trial); coverage < 90% → no trial
  (data gate); `|t_NW| < 2` or wrong-signed → null (logged); `DSR < 0.95` → does **not**
  graduate (logged like trial #8, **NOT relaxed**). **No** post-hoc feature/horizon/quantile
  scans (each is +1 trial).
- **Paired controls / known artifacts:**
  - **Sector-coverage artifact control:** run the SAME GP/A book on **financials only**
    (where GP is undefined/degenerate) — it must show **no** quality effect; a "signal"
    there flags a coverage/sector construction bug, not alpha.
  - **Accruals leg reported SEPARATELY**, not blended: accruals/A is sector-agnostic
    (~93% coverable) but **likely dead post-2002** (Green–Hand–Soliman) — keep it as its
    own hedged claim so a GP/A result isn't contaminated by (or credited to) a dead signal.
  - Standard `risk_report` betas/sector-tilts reported (neutralization is measured, not
    asserted).
- **Honest dollar capacity:** **HIGH.** Large-cap, 63-day rebalance → low turnover (the
  inverse of the carry trades). Plausibly **$100M+** before impact bites (confirm via the
  existing `--capacity` sqrt-impact sweep). **The binding constraint is alpha existence and
  the one-time data cost (Sharadar pricing / WRDS sponsorship), not capacity.**
- **Failure interpretation:** a clean null extends the project's headline from "price-only
  features are arbitraged in large caps" to "…and so is free-data-fidelity quality" — a
  citable result. A DSR-failing-but-right-signed result is logged like trial #8, unrelaxed.

---

## DB — Closed-end-fund **dated-convergence** to NAV (the H6 mutation that fixes #11)

- **Status:** DRAFT — PROPOSED, **Stage-1-census-gated**. Distinct from H6: H6 tested
  open-ended discount *reversion* (killed as a 1-week bid-ask-bounce artifact); this tests
  convergence anchored to a **public terminal-NAV date**, which defeats that artifact by
  construction. **Not a salvage of #11** — different signal (dated event, not z-reversion),
  different universe (event-conditioned), pre-declared.
- **Hypothesis:** A CEF with a **publicly announced terminal-NAV event** (liquidation,
  open-ending/conversion, or fixed-term maturity) trading at a persistent discount
  converges to NAV by the event date, earning a dated structural return — **and the edge
  survives a 1-week entry lag** (i.e., it is NOT the #11 microstructure artifact).
- **Exact testable prediction:** A long book of discounted funds with a public terminal
  date within the holding window, **entered ≥1 trading week after the event/discount is
  public** and held toward the date, earns right-signed convergence return net of 25 bps/side,
  with `t_NW ≥ +2`, beating an equal-weight-CEF baseline, and **the 1-week-lag version is
  itself ≥ the no-lag version** (the registered implementability gate).
- **Universe + dates:** CEF-only (exclude BDCs — their deaths can distress, per the census),
  2010→2026. Events from `cef_deaths.py` **extended to include forward-dated term funds**
  (funds whose terminal date was public at t — strictly PIT), plus completed
  liquidations/open-endings already in the census. Discounts from CEFConnect; total return
  from yfinance (W-FRI). Cadence weekly → n_obs ≈ #fund-weeks (multi-yr → ~520–834).
- **Label definition:** distribution-inclusive total return from entry (public date + 1-week
  lag) to the terminal date, or a fixed 60/120-day horizon, whichever is shorter.
- **Cheapest kill test (ZERO-TRIAL, the H5/H6 Stage-1 pattern):**
  1. **Descriptive census:** assemble funds with public terminal-NAV dates; chart **discount
     vs. days-to-event**. If discounts are already **≈0 at announcement** (already priced)
     or the PIT event set is **< ~50 events** (no power), **KILL before any trial.**
  2. **Machinery gate:** synthetic `planted_reversion`(→convergence) recovered /
     `random_walk` rejected, paired.
- **Success criteria (frozen):** right-signed convergence `t_NW ≥ +2` **AND** net SR
  clearing `DSR ≥ 0.95 at N=12` (weekly hurdle ≈ **0.83 @ ~16 yr**, ≈ 1.05 @ 10 yr) **AND**
  beats EW-CEF baseline net of costs **AND** ≥ ~50 PIT events **AND** the **1-week-lag
  implementability gate passes** (lagged SR ≥ no-lag SR, both > 0).
- **Kill criteria:** census shows pre-priced discounts or < 50 events → no trial; **1-week-lag
  version null/negative → KILL** (it was the #11 artifact again); shuffle control earns
  (|SR| ≥ 0.3) or NAV-staleness control fails → artifact; `DSR < 0.95` → does not graduate,
  logged, **not relaxed**. No holding-period/quantile/cost scans afterward.
- **Paired controls / known artifacts:**
  - **Entry-lag gate** (the headline control — the #11 lesson made a pass/fail criterion).
  - **NAV-staleness control:** re-run on the daily-NAV-only subuniverse (stale NAVs
    manufacture fake convergence).
  - **Label-shuffle control:** terminal returns permuted across funds within each week earn
    ~0.
  - **Survivorship:** completed-death funds (94% NAV-events, conservative direction) are
    fine; forward-term funds must be identified strictly PIT (terminal date public at t).
- **Honest dollar capacity:** **VERY LOW — $25k–$150k.** Sub-$400M funds, partial fills,
  wide tail spreads. This is structurally "too small for Saba" (the why-it-exists) but it
  caps the trade at personal-account scale. **Frame it honestly as a structural-premium
  *demonstration*, not a scalable strategy** — capacity is the headline limitation, reported
  up front, not buried.
- **Failure interpretation:** if the census kills it (pre-priced/thin), that's a clean
  zero-trial finding about CEF efficiency. If it runs and fails the lag gate, it confirms the
  #11 artifact generalizes — also citable. If it graduates, it's the project's first
  structurally-protected, capacity-honest edge.

---

## DC — Borrow-fee **change** as a net-of-fee short-side signal (H7 Stage-2 analysis spec)

- **Status:** DRAFT — Stage-2 analysis spec for the **already-registered, collection-only
  H7**. Designing it now is allowed (no trial, no live-config change); it pre-commits the
  ≥60-cycle analysis so it can't be reverse-engineered from the data. **Run only after ≥60
  cycles (~Sept 2026) and explicit sign-off.**
- **Hypothesis:** The daily **change** in IBKR borrow fee (Δfee) is cross-sectionally
  informative about next-period residual returns on the short side, **net of the carried
  fee** — testable only on the proprietary daily snapshot the project began accruing
  2026-06-12 (unbackfillable; the one edge here McLean–Pontiff cannot have arbitraged).
- **Exact testable prediction:** A dollar-neutral book short the largest-positive-Δfee names
  (informed-short surge) / long the lowest-Δfee names earns right-signed residual return
  **after subtracting the borrow fee accrued over the holding period**, `t_NW ≥ +2`, DSR
  clearing the then-current-N hurdle.
- **Universe + dates:** the live experiment's own PIT-scored cross-section (S&P), borrow
  snapshots 2026-06-12 → ≥60 cycles. n_obs is small at first → expect a **high** weekly DSR
  hurdle (≈ 1.3–1.4 at ~60 weekly obs) — power, not sign, is the early constraint.
- **Label definition:** forward residual return over the holding horizon **minus the borrow
  fee accrued** on the short leg (the net-of-fee object — the whole point; a gross number is
  forbidden here because the fee *is* the cost).
- **Cheapest kill test (ZERO-TRIAL descriptive, at the Stage-2 gate):**
  1. Does Δfee correlate with next-period residual return **gross** at all? If not → null,
     no trial.
  2. Is the fee tail (measured p99 ≈ 163%/yr) so wide that net is negative for exactly the
     high-Δfee names the book would short? If the fee swamps the signal → "**risk veto
     only**," the honest H7 fallback, no trial spent on a foregone conclusion.
- **Success criteria (frozen):** right-signed `t_NW ≥ +2` **net of fee** **AND** `DSR ≥ 0.95
  at then-current N` **AND** the short-leg borrow cost explicitly netted.
- **Kill criteria:** gross sign absent → null; net sign absent → risk-veto-only (logged, not
  a trade); < 60 cycles → wait. No level/quantile scans post-hoc.
- **Paired controls / known artifacts:**
  - **Level-vs-change decomposition:** is it the fee **level** (already public, likely
    priced) or the **change**? Only the change is the claim.
  - **Shuffle control:** Δfee permuted cross-sectionally earns ~0.
  - **Recall-risk caveat** (Engelberg–Reed–Ringgenberg): short-selling risk is itself
    priced; report it, don't assume it away.
- **Honest dollar capacity:** **LOW** — short-side only, on hard-to-borrow (small/illiquid)
  names, with recall risk. Frame as a **risk overlay + small PA short book**, not a scalable
  long-short.
- **Failure interpretation:** even a null is a write-up asset — "the strongest reported
  cross-sectional predictor (loan fees) does not survive net of the fee at our fidelity,"
  measured on a dataset nobody else has.

---

## DD — Vendor data-revision intensity as a risk/down-weight signal (H5 Stage-2, compressed)

- **Status:** DRAFT — Stage-2 descriptive spec for the **already-registered, collection-only
  H5**. Lowest alpha prior of the survivors; advances as **risk work first**, honestly.
- **Hypothesis (risk-first):** cross-sectional return-cell revision intensity proxies feature
  instability / realized vol; down-weighting high-revision names stabilizes a book's IC. The
  honest prior is **risk/quality, not alpha** (matches the live H5 registration).
- **Cheapest kill test (ZERO-TRIAL descriptive, at ≥60 cycles):** correlate per-name
  return-cell-revision counts with (a) realized vol and (b) feature-IC instability across
  walk-forward windows. **No structure → idiosyncratic vendor noise → a write-up paragraph,
  done.** Only if a stable signed relationship exists does a *separate, later* trial ask
  whether down-weighting improves an existing book's net DSR.
- **Success criteria (Stage-2 descriptive):** a stable, signed cross-sectional relationship
  (reported with NW t-stats); the alpha question is deferred to its own future registration.
- **Paired control:** shuffle revision-counts cross-sectionally → must explain nothing.
- **Honest dollar capacity:** N/A (a risk overlay, not a standalone book).
- **Failure interpretation:** "revision intensity is idiosyncratic vendor noise" is itself
  the finding — the only way anyone learns it is to have measured it, which the moat dataset
  uniquely allows.

---

## One-line ranking of the drafts (full rationale in 05_summary.md)
1. **DA (H1 quality)** — lowest hurdle (~0.90), harness built, win-or-clean-null; gated only
   by a data-source decision.
2. **DB (CEF dated-convergence)** — novel, defeats the #11 artifact, census-first
   (zero-trial kill); capacity-honest but small.
3. **DC (borrow Δfee, H7 S2)** — proprietary moat; design now, run ~Sept 2026, net-of-fee.
4. **DD (revision intensity, H5 S2)** — risk-first descriptive; weakest prior, real moat.
