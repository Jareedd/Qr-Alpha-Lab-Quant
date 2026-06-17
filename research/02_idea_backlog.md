# 02 — Idea Backlog

**STEP 2.** 22 candidate directions, biased toward edges a solo researcher can reach and
institutions structurally won't: capacity-constrained corners, forced/structural flows,
market plumbing, data-assembly moats (including the project's *own* logs), honest-timestamp
event studies, and unusual venues.

**Format per idea:** MECHANISM (who pays and why) · WHY-STILL-ALIVE (the structural reason
it isn't arbitraged) · DATA (specific source) · EVIDENCE STATUS
`[documented-but-niche | plausible-extension | speculation]` · CHEAPEST-KILL-TEST
(zero-trial wherever possible).

**Self-imposed screens (from the audit + scan):** (a) no relitigating a logged null
(index-deletion drift, spot-perp carry); (b) Hou–Xue–Zhang guillotine — anything that needs
microcaps/equal-weight is presumed a cost mirage; (c) anything with a short leg must price
borrow; (d) any reversion/mean-cross idea must specify an **entry-lag/implementability gate**
(the trial-#11 lesson). Ideas with **no mechanism are deleted at the bottom**, on the record.

IDs are stable (used by 03_cull and 04_registration_drafts).

---

## Group I — Data-assembly moats (the project's structural advantage)

### I1. Borrow-fee *change* as a cross-sectional short-side signal (H7 Stage-2 core)
- **MECHANISM:** A jump in the borrow fee is the observable price of a surge in informed
  short demand or a withdrawal of lendable supply. Whoever is long into a fee spike is
  paying the squeeze; the marginal lender is extracting it. Engelberg et al. find loan
  fees the strongest single cross-sectional predictor.
- **WHY-STILL-ALIVE:** The loan-fee anomaly is reported *gross* of the fee — and the fee
  *is* the cost, so net tradability is genuinely unsettled. More importantly the daily
  fee panel is **unbackfillable**: it exists only for someone who snapshotted it, which
  the project started 2026-06-12.
- **DATA:** `results/live/borrow_*.json` (IBKR ftp2, already collecting; ≥60 cycles ≈ Sept 2026).
- **EVIDENCE:** documented-but-niche (Engelberg–Evans–Leonard–Reed–Ringgenberg loan-fee
  anomaly; Engelberg–Reed–Ringgenberg short-selling risk).
- **KILL-TEST (descriptive, zero-trial):** at Stage-2, regress next-period residual return
  on ΔlogFee, **net of the fee carried over the holding period**. If the predictive sign
  vanishes once you subtract the fee you'd actually pay, it's a risk veto, not a trade.

### I2. Vendor data-revision intensity as a feature-reliability / risk down-weight (H5 Stage-2 core)
- **MECHANISM:** A name whose vendor price-history keeps getting rewritten (dividend/split
  re-adjustments, corrections) has noisier point-in-time features and denser corporate
  actions; down-weighting it should stabilize any book's IC. Nobody "pays" — this is honest
  risk control, not alpha.
- **WHY-STILL-ALIVE:** The daily-diff dataset is **proprietary by construction** (the
  project commits `revisions_*.json` publicly, so the timestamp is verifiable, but the
  series only exists because someone snapshotted daily since 2026-06-11).
- **DATA:** `results/live/revisions_*.json`.
- **EVIDENCE:** speculation → plausible (ALFRED proves data-vintage is a study-able object
  in macro; no equity analogue surfaced — consistent with the moat claim).
- **KILL-TEST (descriptive, zero-trial):** at Stage-2, does cross-sectional return-cell-
  revision count correlate with realized vol and feature instability? If revisions are
  idiosyncratic noise with no structure, it's a write-up paragraph, not a signal.

### I3. CEF liquidation / term-maturity discount convergence (uses the dead-fund census)
- **MECHANISM:** A closed-end fund that has announced a liquidation, open-ending, or has a
  *fixed term* must converge to NAV by a **known date**. A persistent discount on such a
  fund is a dated, structural convergence trade. The discount seller (impatient retail)
  pays the holder who waits for the event.
- **WHY-STILL-ALIVE:** Tiny funds "too small for Saba" (the project's own H6 framing);
  the convergence date is buried in N-2/N-8F/proxy filings that nobody assembles.
- **DATA:** `cef_deaths.py` (SEC EDGAR 25-NSE/N-8F, already built) for the event set;
  CEFConnect discounts + yfinance prices for the path.
- **EVIDENCE:** plausible-extension (Pontiff CEF predictability + the census's 94%-NAV-event
  finding).
- **KILL-TEST (descriptive census, zero-trial):** for funds with a *public* liquidation/term
  date, chart discount vs. days-to-event. If discounts don't tighten approaching the date,
  there's no convergence to harvest.

---

## Group II — Honest-timestamp event studies (reuse the H8/H9 machinery)

### II1. CEF discount reversion, MONTHLY, with a registered entry-lag gate (H6 reframe)
- **MECHANISM:** A closed-end fund has no creation/redemption, so price can diverge from
  NAV persistently; retail panic and tax-season selling widen discounts mechanically, and
  they revert as the panic passes. The retail panic seller pays the patient holder.
- **WHY-STILL-ALIVE:** Capacity $50k–$250k — structurally beneath Saba-class activists;
  the no-arb structure (Pontiff) is not a picked-over statistical factor.
- **DATA:** CEFConnect weekly discount + yfinance distribution-inclusive daily total return.
- **EVIDENCE:** documented (Pontiff) — but the project's trial-#11 *implementation* died on a
  one-week bid-ask-bounce artifact, **so the reframe is the whole point.**
- **KILL-TEST (one trial, but with a pre-registered implementability gate):** test reversion
  at a **monthly** signal cadence with the holding entered at a **1-week lag** as a PASS/FAIL
  criterion (not a diagnostic). The synthetic `planted_reversion`/`random_walk` gate runs
  first (zero-trial). If the lagged version is null, the edge was microstructure — kill.

### II2. Opportunistic-insider (Form 4) cluster-buy post-filing drift
- **MECHANISM:** Insiders trade for many reasons; the *opportunistic* subset carries private
  information, and the market underreacts to the public Form 4. The uninformed counterparty
  selling into insider accumulation pays.
- **WHY-STILL-ALIVE:** Concentrated in smaller, poorly-governed, attention-starved names;
  classifying routine vs. opportunistic is data work most won't do.
- **DATA:** SEC EDGAR Form 4 (free; filed within 2 business days — an honest public stamp).
- **EVIDENCE:** documented-but-niche (Cohen–Malloy–Pomorski ~82 bps/mo VW for opportunistic;
  likely decayed post-2012).
- **KILL-TEST:** planted-event synthetic gate (reuse `inject_post_event_drift`) → then a
  matched-control post-filing drift on **liquid** names only (dodge the cost mirage). Null or
  control-explained → kill.

### II3. Schedule-13D activist filing post-announcement drift
- **MECHANISM:** Activists target undervalued firms and create value over the campaign; the
  market underreacts at the public 13D filing. Pre-campaign passive holders pay.
- **WHY-STILL-ALIVE:** Idiosyncratic, hard to scale, legal/headline risk; the *pre-filing*
  run-up is informed, so only the **post-filing** drift is honestly tradable.
- **DATA:** SEC EDGAR 13D (free).
- **EVIDENCE:** documented (Brav–Jiang–Partnoy–Thomas ~7–8% around filing; no long-term
  reversal) — but verify the **post-2024 shortened 13D deadline** changes the pre/post split.
- **KILL-TEST:** matched-control drift entered the day **after** the filing timestamp. If the
  abnormal return is all pre-filing (the run-up), the tradable residual is ~0 — kill.

### II4. PEAD in liquid, low-institutional-ownership names with honest 8-K timestamps
- **MECHANISM:** Limited attention → underreaction to earnings surprises; drift continues as
  information diffuses. Inattentive holders pay the attentive.
- **WHY-STILL-ALIVE:** Stronger where institutions are sparse — but that overlaps with high
  costs (the central tension).
- **DATA:** 8-K Item 2.02 earnings timestamps (EDGAR) + yfinance.
- **EVIDENCE:** documented but decaying (Chordia–Subrahmanyam–Tong; Martineau 2022 vs.
  Meursault et al. 2023).
- **KILL-TEST:** value-weighted, net-of-cost drift with a size/momentum-matched control,
  **restricted to liquid names**. If the edge needs the illiquid tail, Hou–Xue–Zhang says
  it's a mirage — kill.

---

## Group III — Forced / structural flows

### III1. IPO lockup-expiration supply shock + reversion
- **MECHANISM:** At the 180-day lockup expiry, a large block of insider supply becomes
  sellable on a **known date**, pressuring price; over-shoot reverts. Locked-up insiders
  diversifying pay the liquidity provider.
- **WHY-STILL-ALIVE:** Persistent, well-documented flow that's hard to scale and event-sparse;
  most quant books ignore single-name calendar events.
- **DATA:** S-1/424B lockup terms (EDGAR) + prices.
- **EVIDENCE:** documented-but-niche.
- **KILL-TEST:** census of lockup dates → matched-control price path in a [−10, +20]d window.
  No pre-expiry drift or post-expiry reversion beyond control → kill.

### III2. CEF dilutive rights-offering events
- **MECHANISM:** A transferable rights offering issues new shares below NAV, mechanically
  diluting the discount; terms are public in advance. Non-participating holders pay
  participants.
- **WHY-STILL-ALIVE:** Tiny, retail-dominated CEFs; corporate-action mechanics few model.
- **DATA:** CEF rights-offering announcements (N-2/424B on EDGAR) + CEFConnect.
- **EVIDENCE:** speculation.
- **KILL-TEST:** event census of rights offerings → discount path around the record date.
  No predictable move → kill.

### III3. Month-/quarter-end rebalance pressure (overnight vs. intraday split)
- **MECHANISM:** Calendar-driven pension/fund rebalancing forces predictable late-month
  flows; Lou–Polk–Skouras tie the overnight/intraday return split to institutional
  rebalancing need. Forced rebalancers pay the patient.
- **WHY-STILL-ALIVE:** Structural calendar constraint; the *decomposition* (not a standalone
  factor) is what's exploitable.
- **DATA:** yfinance daily **open + close** (no intraday ticks needed).
- **EVIDENCE:** documented-but-niche (Lou–Polk–Skouras adjacent).
- **KILL-TEST (zero-trial descriptive):** compute month-end overnight vs. intraday returns
  across the universe; if the split is flat, there's nothing to condition on.

### III4. Russell 2000 reconstitution front-run
- **MECHANISM:** End-of-May market-cap ranking makes June reconstitution adds/drops
  predictable; passive funds must trade them. Index funds pay the front-runner.
- **WHY-STILL-ALIVE:** It mostly **isn't** — this is the most pre-announced, most-arbitraged
  flow in equities (~70% of institutional AUM benchmarked to Russell). Included to be killed
  on the record.
- **DATA:** Russell methodology + prices.
- **EVIDENCE:** documented-but-(likely)-dead (Madhavan 2003; verify post-banding decay).
- **KILL-TEST (zero-trial):** check whether the add/drop window abnormal return post-2010 is
  still positive net of the crowded entry — expect ~0.

---

## Group IV — Market plumbing

### IV1. Overnight-component momentum on the liquid S&P universe
- **MECHANISM:** Lou–Polk–Skouras: momentum profits accrue **overnight**; the intraday leg
  is an offsetting institutional-rebalancing drag. Forced intraday rebalancers pay
  overnight-holders.
- **WHY-STILL-ALIVE:** It's a structural decomposition of a known factor, not a new factor;
  the overnight/close-to-close gap is sticky.
- **DATA:** yfinance daily open + close (already in the data layer).
- **EVIDENCE:** documented (Lou–Polk–Skouras, *JFE* 2019).
- **KILL-TEST:** does the overnight-only momentum book beat close-to-close momentum **net of
  the open-auction spread** on the PIT S&P universe? If the spread eats the gap, kill.

### IV2. Cross-venue crypto funding-rate dispersion
- **MECHANISM:** When collateral can't move freely across exchanges, funding for the same
  perp diverges; the trapped (over-levered) side keeps paying. The cross-venue spread is the
  price of the plumbing friction.
- **WHY-STILL-ALIVE:** Collateral/withdrawal frictions and venue/counterparty risk; the
  majors' single-venue carry is already shown decayed (trial #8) so this is a *different*
  mechanism (dispersion, not level).
- **DATA:** CoinGlass `/fundingRate/exchange-list` or FundingPulse (freemium) + per-exchange
  dumps for delisted contracts.
- **EVIDENCE:** plausible-extension.
- **KILL-TEST (descriptive, zero-trial):** measure the funding spread for a few liquid perps
  across venues; if median dispersion < plausible cross-venue execution cost, there's no net
  trade.

### IV3. Closed/illiquid bond-ETF discount reversion under AP stress
- **MECHANISM:** ETF authorized-participant arbitrage normally pins price to iNAV, but in
  stressed/illiquid corners (HY/EM/muni bond ETFs) the AP mechanism widens premiums/discounts
  that then revert. Stressed sellers pay liquidity providers.
- **WHY-STILL-ALIVE:** AP capacity and balance-sheet limits in odd corners; the effect is
  episodic, not a steady factor.
- **DATA:** ETF price + published iNAV/NAV (free-ish from issuer/exchange).
- **EVIDENCE:** plausible-extension (the CEF logic in ETF clothing — but ETFs usually arb
  tightly, so this is borderline).
- **KILL-TEST (descriptive):** distribution of premium/discount for illiquid bond ETFs; if it
  rarely exceeds the bid-ask + creation cost, there's no reversion to trade.

---

## Group V — Capacity-constrained corners & unusual venues

### V1. Pre-deal SPAC trust-NAV floor
- **MECHANISM:** A pre-merger SPAC holds ~$10/share in trust, redeemable at NAV; trading
  *below* trust value is a near-riskless yield + deal optionality. Forced/liquidity sellers
  pay the patient holder who can redeem.
- **WHY-STILL-ALIVE:** Post-2021 SPAC winter, tiny capacity, redemption mechanics and
  deadlines are fiddly; institutions left the space.
- **DATA:** SEC filings (trust value, redemption/extension dates) + prices.
- **EVIDENCE:** documented-but-niche (SPAC-arbitrage literature; verify current universe size
  — may be too small to matter in 2026).
- **KILL-TEST (descriptive census):** how many live pre-deal SPACs trade below trust today,
  and what's the annualized yield-to-redemption net of fees? If the count is single digits or
  the yield ≈ T-bills, kill on capacity.

### V2. Exchange-traded $25-par preferreds / "baby bonds" near call/maturity
- **MECHANISM:** Retail-dominated $25-par preferreds and baby bonds have **known call and
  maturity dates**; price converges to par/call on a schedule, and retail mis-prices the
  convergence. Index funds ignore them entirely.
- **WHY-STILL-ALIVE:** Too small/illiquid for institutions; not in major equity indices;
  prospectus call schedules are unassembled.
- **DATA:** prices (yfinance/stooq) + prospectus call/maturity terms (EDGAR/issuer).
- **EVIDENCE:** speculation → plausible.
- **KILL-TEST (descriptive census):** for preferreds within ~12 months of a call, chart price
  vs. call price net of the wide retail spread. If the spread swamps the convergence, kill.

### V3. Prediction-market favorite-longshot / calibration edge (Kalshi/Polymarket)
- **MECHANISM:** Retail bettors overpay for longshots and underpay favorites (favorite-longshot
  bias); resolved-market calibration reveals systematic mispricing. The longshot bettor pays.
- **WHY-STILL-ALIVE:** Small, regulated/novel venues; "not a security," so most quant capital
  can't or won't touch it.
- **DATA:** Kalshi / Polymarket public APIs (free-ish) for prices + resolutions.
- **EVIDENCE:** documented-but-niche (favorite-longshot bias is old and robust in betting
  markets; verify it holds on these specific venues).
- **KILL-TEST (descriptive, zero-trial):** build a calibration curve on resolved markets
  (predicted price vs. realized frequency). If well-calibrated, no edge; if biased, size vs.
  fees/liquidity. **Caveat:** likely outside the project's equity scope — keep as a flagged
  "unusual venue" entry.

### V4. Quality (GP/A + accruals) on a survivorship-safe fundamentals source (H1, properly sourced)
- **MECHANISM:** Profitable, low-accrual firms out-earn; the premium rebalances slowly, so it
  survives costs where price factors didn't (Novy-Marx). Growth/glamour buyers pay.
- **WHY-STILL-ALIVE:** Slow-decaying relative to price anomalies (though decaying); the
  project's *specific* barrier is data access, not crowding.
- **DATA:** **Sharadar SF1** (survivorship-free, paid) or WRDS/Compustat — the free SEC path
  is survivorship-blocked (~73–75%). Non-financials restriction for GP/A.
- **EVIDENCE:** documented-but-niche (Novy-Marx; the accruals leg may be dead post-2002).
- **KILL-TEST:** the harness is **already built and gated** (`run_fundamentals.py`); the kill
  test is the **data audit** — does a reachable source give ≥90% survivorship-safe coverage?
  If only the free SEC source is available, the DATA GATE already refuses it (zero trial).

### V5. Microcap profitability/value (included to be killed)
- **MECHANISM:** Standard quality/value premia, larger in small caps.
- **WHY-STILL-ALIVE:** It largely **isn't, net** — Hou–Xue–Zhang: microcap anomalies are
  "more apparent than real," ~96% of trading-friction anomalies fail with NYSE breakpoints.
- **DATA:** yfinance/Sharadar small-cap universe.
- **EVIDENCE:** documented-but-dead-net.
- **KILL-TEST (zero-trial):** value-weight the signal and apply NYSE breakpoints; the premium
  should collapse — confirming the cost mirage.

---

## Group VI — Conditioning / regime (trap-flagged)

### VI1. Volatility-managed scaling of a *graduated* signal (not a standalone gate)
- **MECHANISM:** Moreira–Muir: scaling exposure down when realized vol is high raises Sharpe
  because vol isn't offset by proportional expected-return changes. The investor who holds
  constant risk into vol spikes pays.
- **WHY-STILL-ALIVE:** A time-series risk-management overlay, not a cross-sectional alpha
  claim — and it's applied to an *already-graduated* book, so it can't manufacture an edge
  from nothing.
- **DATA:** the graduated strategy's own returns + realized vol (no new data).
- **EVIDENCE:** documented (Moreira–Muir 2017) with a real cost critique (DeMiguel et al. 2024).
- **KILL-TEST:** only meaningful once a signal graduates; then, does vol-scaling improve the
  **net** DSR? If not, it's complexity for its own sake. **Not a trial on its own** — an
  engine setting.

### VI2. HMM filtered-regime gate on momentum (H4 as registered — trap)
- **MECHANISM:** Momentum crashes cluster in high-vol panics (Daniel–Moskowitz); a causal
  vol-regime filter could sidestep them.
- **WHY-STILL-ALIVE:** Weak prior; the project flags it as a **trap check**, because the
  vol-regime IC artifact (+0.06 to +0.13 IC on signal-free data) mimics exactly this result.
- **DATA:** member-masked market return (in-repo).
- **EVIDENCE:** plausible-but-trap-flagged.
- **KILL-TEST:** the **paired label-shuffle artifact control** — registered before any run. If
  the shuffled control shows the same "conditional lift," it's the artifact, not alpha. This
  is the single cheapest kill in the whole backlog and it's already half-built.

---

## Deleted in STEP 2 — no mechanism (on the record)
- **Generic "ML on more price features."** Trials #2–7 already exhausted price-only features
  on the honest universe; no new mechanism, just more knobs. Deleted (also forbidden as
  salvage).
- **Index-deletion / addition drift redux.** Logged null (trial #9); no new mechanism.
  Deleted (no-salvage rule).
- **Spot-perp cash-and-carry as a fresh idea.** This is H2/H9 in a hat; mechanism already
  tested and shown decayed/priced. Deleted (no-salvage rule).
- **Weather/commodity-seasonality micro-futures, social-sentiment scrapes, lottery-stock
  "buy the hype."** No identifiable counterparty-who-pays that the project can articulate and
  test cheaply; data is noisy or unobtainable PIT. Deleted (no mechanism).
- **Reg-A / crowdfunding instruments.** No liquid, clean, PIT-priced data. Deleted (no data
  ≈ no testable mechanism).
