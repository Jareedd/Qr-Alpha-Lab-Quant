# Edge candidates — diverge / attack / register (2026-06-12)

**Status of this document.** Ideation only. Nothing here authorizes a real-data run.
N = 7 and stays 7. The live config is frozen. No trial from the log (#2–#7) is
salvaged, sign-flipped, or post-hoc conditioned here. The Phase 3 specs below are
written in the format of `writeup/preregistered_hypotheses.md` but are NOT entered
into that file — registration happens only on owner sign-off, per protocol.

**Selection pressure applied up front.** Every candidate was screened against the
machine's own scars before being written down: trial #2 (survivorship turned net
SR 0.82 into −0.01), the capacity drag curve (a 3.46×/yr-turnover book needs
≥1.4%/yr true gross alpha to exist at $1M), trial #5 (IC and P&L are different
objects), trial #4 (the sign-flip salvage trap), and the 2026-06-12 regime entry
(label machinery × vol-regime interactions can manufacture conditional IC from
nothing — paired controls are mandatory, not optional).

Factual claims I cannot guarantee are tagged `(verify)`. No invented citations;
the handful of confirmed sources are listed at the end.

---

## PHASE 1 — DIVERGE: 20 candidates

### Bucket a — capacity-constrained corners

**A1. Microcap post-earnings drift, timed off EDGAR acceptance timestamps**
- MECHANISM: in sub-$300M names with zero analyst coverage, the incumbent holder
  base is small and inattentive; earnings information diffuses over weeks. The
  counterparty is the existing holder who neither buys the surprise nor sells the
  miss promptly. EDGAR acceptance datetimes give the honest event clock (vendor
  earnings calendars embed look-ahead; filing timestamps cannot).
- WHY STILL ALIVE: capacity. Institutions cannot put $50M into names that trade
  $200k/day; the documented persistence of PEAD net of costs is concentrated
  exactly where it cannot be scaled (Novy-Marx–Velikov cost-mortality logic).
- DATA: EDGAR full-text index + acceptance timestamps (free, complete, includes
  dead companies); prices via yfinance.
- EVIDENCE STATUS: documented-but-niche (PEAD is documented; its microcap
  cost-limited persistence is documented (verify the specific net-of-cost
  microcap result)).

**A2. Crypto long-tail weekly cross-sectional reversal (ranks ~50–300 by mcap)**
- MECHANISM: retail momentum chasers in small alts pay for immediacy; a weekly
  contrarian rank portfolio is selling them liquidity. The counterparty is the
  chaser whose demand curve is taste, not value.
- WHY STILL ALIVE: capacity too small — no desk can deploy meaningful size in
  rank-200 alts; custody/exchange risk adds career risk for any institution.
- DATA: Binance public dumps at data.binance.vision (klines, trades, funding —
  confirmed to exist; retention of delisted pairs (verify)).
- EVIDENCE STATUS: documented-but-niche (academic crypto cross-section papers
  exist (verify)); the tail-specific version is plausible-extension.

**A3. Sub-$500M US weekly reversal (liquidity provision)**
- MECHANISM: institutions demanding immediacy in illiquid names pay a liquidity
  premium; short-horizon reversal is the market-maker's rent. Counterparty:
  the impatient block seller/buyer.
- WHY STILL ALIVE: costs — the rent is documented to be consumed by the spread
  for anyone trading at scale; the claim to test is whether a tiny, patient,
  limit-order book keeps any of it.
- DATA: yfinance daily bars.
- EVIDENCE STATUS: documented-but-niche (short-term reversal is documented;
  documented to die net of costs (verify the specific paper)).

### Bucket b — structural & forced flows

**B1. S&P 500 discretionary-deletion overshoot and post-effective reversal**
- MECHANISM: index trackers must sell the deleted name at the effective-date
  close; their loss function is tracking error, not price. If the forced flow
  overshoots, the weeks after the effective date revert. Counterparty: the
  indexer, paying by construction.
- WHY STILL ALIVE: lumpy event P&L (~15–25 events/yr) is career risk for a PM
  judged monthly; deleted names are falling knives nobody wants on a factsheet.
  Honest headwind, stated now: Greenwood–Sammon (J. Finance 2025) find the
  *announcement-to-effective* deletion effect shrank to ~0.1% in 2010–2020. The
  surviving question is strictly the post-effective window on *discretionary*
  deletions — a different object, but the prior is weak.
- DATA: already in the repo — the PIT membership change table; removal-reason
  classification from the same Wikipedia table + press releases.
- EVIDENCE STATUS: documented-but-niche for the long-horizon deletion rebound
  (verify — often attributed to Research Affiliates work on deletions; do not
  cite without finding it); documented (Greenwood–Sammon) that the event-window
  effect is gone.

**B2. Russell reconstitution front-running at the small-cap boundary (June)**
- MECHANISM: Russell trackers trade the June reconstitution close; predictable
  rank-day demand. Counterparty: the indexer.
- WHY STILL ALIVE: claim is it largely isn't — kept in the list to be killed on
  the record.
- DATA: FTSE Russell provisional add/delete lists (published free in June
  (verify)); yfinance prices.
- EVIDENCE STATUS: documented — and documented to be heavily arbitraged
  (Greenwood–Sammon mechanism: predictability attracts front-runners).

**B3. December tax-loss selling bounce in illiquid small caps**
- MECHANISM: retail and advisor tax harvesting is calendar-driven and
  price-insensitive; losers in illiquid names get an extra supply shock in
  December that reverts in January. Counterparty: the tax-motivated seller.
- WHY STILL ALIVE: capacity plus seasonality — one trade per year is
  institutionally pointless.
- DATA: yfinance; YTD-loser × low-ADV ranks are computable from bars alone.
- EVIDENCE STATUS: documented-but-niche (January/tax-loss literature is old and
  large (verify current-decade persistence)).

**B4. IPO lockup-expiry short**
- MECHANISM: insiders and VCs are contractual sellers at a public, dated event
  (typically day 180). Counterparty: the pre-expiry holder who hasn't priced
  the supply.
- WHY STILL ALIVE: borrow. The names with the biggest expected drop are the
  hardest and costliest to borrow — which is exactly the attack on it below.
- DATA: lockup dates from S-1/424B4 filings on EDGAR (parseable, tedious);
  prices via yfinance.
- EVIDENCE STATUS: documented-but-niche (Field–Hanka-era literature (verify);
  also documented that borrow costs consume it (verify)).

### Bucket c — market plumbing

**C1. Long-tail perp funding-rate extremes (beyond H2's top-30)**
- MECHANISM: leveraged directional longs in small perps pay funding to hold the
  crowded side; extreme funding is payment for taking unpopular inventory plus
  squeeze risk. Counterparty: the levered long, directly observable in the
  funding print.
- WHY STILL ALIVE: capacity and venue risk — tail perps are untouchable at
  institutional size; funding extremes there are larger and stickier than in
  BTC/ETH (verify).
- DATA: data.binance.vision funding-rate dumps (confirmed); Bybit/OKX equivalents
  (verify).
- EVIDENCE STATUS: documented for majors (the carry mechanism is H2's prior);
  plausible-extension for the tail.

**C2. Borrow-fee / short-availability cross-section**
- MECHANISM: a 50–300% borrow fee is the price of concentrated negative
  information. The documented "shorting premium": high-fee names underperform
  gross of fee (verify — Drechsler–Drechsler working paper era). Counterparty:
  the constrained optimist holding the unborrowable name.
- WHY STILL ALIVE: data too tedious — fee history is not free to backfill;
  whoever wants the history must have been snapshotting it. Same moat structure
  as H5: unbackfillable by construction.
- DATA: IBKR Short Stock Availability data — fee rates, shares available,
  lender count; historical indicative rates downloadable as CSV per IBKR's own
  pages (confirmed); the legacy FTP endpoint specifics (verify).
- EVIDENCE STATUS: documented-but-niche gross of fee; net-of-fee tradability
  ambiguous — which is precisely what makes it a collection candidate, not a
  trading candidate.

**C3. FINRA daily short-sale-volume ratio cross-section**
- MECHANISM: informed short flow is printed daily in FINRA's short-sale volume
  files; underreaction by everyone who doesn't parse them. Counterparty: the
  buyer ignoring publicly printed informed flow.
- WHY STILL ALIVE: assembly tedium (per-venue daily files, symbol churn);
  documented signal is modest and fast (verify).
- DATA: FINRA Reg SHO daily short-sale volume files (free, history to ~2010
  (verify)).
- EVIDENCE STATUS: documented-but-niche, with mixed results across studies
  (verify).

### Bucket d — data-assembly moats

**D1. Form 4 insider cluster-buys in small names (EDGAR full feed)**
- MECHANISM: insiders buying with their own money, in clusters (≥3 insiders in a
  window), against recent price weakness, in names no analyst models.
  Counterparty: the sentiment seller dumping shares to the people with the best
  information in the building.
- WHY STILL ALIVE: alleged tedium of parsing the full Form 4 feed — challenged
  below, because free screeners already did this.
- DATA: EDGAR Form 4 XML, full history, acceptance timestamps (free, PIT-perfect).
- EVIDENCE STATUS: documented-but-niche (insider cluster/"opportunistic" buy
  literature, Cohen–Malloy–Pomorski era (verify)).

**D2. 8-K Item 4.02 (non-reliance on prior financials) negative drift**
- MECHANISM: a company announcing its own past financials cannot be relied upon
  is a severity signal holders reprice slowly; quality-screened funds become
  forced sellers later. Counterparty: the slow holder and the index fund that
  cannot act on filings.
- WHY STILL ALIVE: event scarcity and form-type obscurity — nobody builds
  production systems around a few hundred events/yr (verify count).
- DATA: EDGAR full-text search on Item 4.02, acceptance timestamps (free).
- EVIDENCE STATUS: documented-but-niche (restatement-drift literature is largely
  pre-2010 (verify)).

**D3. Share-count dilution intensity from XBRL cover pages (the ATM machine)**
- MECHANISM: a company running an at-the-market offering is the informed seller,
  printing shares into retail enthusiasm. `dei:EntityCommonStockSharesOutstanding`
  appears on every 10-K/10-Q cover with a filing timestamp — a free, PIT-safe
  share-count series. Short heavy diluters / long non-issuers. Counterparty: the
  retail buyer absorbing ATM supply at the pump.
- WHY STILL ALIVE: in liquid names this is the documented net-issuance factor
  (Pontiff–Woodgate (verify)) and presumably arbitraged; the live version is in
  small names — where the assembly from raw XBRL is tedious and the borrow is
  the problem (attacked below).
- DATA: SEC XBRL frames API / company facts (free); S-3 and 424B5 filings for
  ATM program identification.
- EVIDENCE STATUS: documented (net issuance) + plausible-extension (microcap ATM
  intensity version).

**D4. The project's own data-revision fingerprints (already registered as H5)**
- MECHANISM/STATUS: listed to honor the bucket; this is H5, two-stage, and its
  protocol forbids analysis before Stage 2 registration (~Sept 2026, ≥60
  cycles). No new idea is permitted to touch the revision data early. Action
  here: none. The correct move with this bucket is to widen the moat (see H7
  below), not to peek at the existing one.
- EVIDENCE STATUS: pure-speculation by design — that is what Stage 2 exists to
  find out.

### Bucket e — event-driven with honest timestamps

**E1. LULD/T12 trading-halt resumption drift**
- MECHANISM: halts cluster in manipulated or news-shocked small names; at
  resumption, margin liquidations and panic produce forced flow. Counterparty:
  the liquidated holder.
- WHY STILL ALIVE: messy intraday microstructure nobody wants to productionize;
  but see the attack — the daily-bar version may not contain the event.
- DATA: Nasdaq Trader halts feed (current halts free; deep historical archive
  with timestamps (verify) — likely a collect-forward dataset).
- EVIDENCE STATUS: plausible-extension at daily bars; pure-speculation for the
  tradable version.

**E2. Deletion-to-OTC forced-selling rebound**
- MECHANISM: when a name is delisted to OTC, mandate-constrained institutions
  must sell at any price; the overshoot reverts. Counterparty: the
  mandate-constrained seller.
- WHY STILL ALIVE: OTC data quality is so bad nobody can even measure it —
  which cuts both ways (attacked below).
- DATA: Form 25 filings (EDGAR, timestamped); OTC prices (poor, source (verify)).
- EVIDENCE STATUS: documented-but-niche (verify); high risk the documentation
  itself is a data artifact.

**E3. NT 10-K / NT 10-Q late-filing notifications**
- MECHANISM: filing an NT ("notification of inability to timely file") is a
  public red flag priced slowly because the form type is obscure. Counterparty:
  the inattentive holder.
- WHY STILL ALIVE: form-type obscurity; tedium of joining NT filings to
  subsequent outcomes.
- DATA: EDGAR form-type index (free, timestamped).
- EVIDENCE STATUS: documented-but-niche (verify the drift studies).

### Bucket f — the "no one else" bucket

**F1. Prediction-market longshot bias (Polymarket/Kalshi)**
- MECHANISM: lottery-preference retail overpays for 3–10c contracts; the
  favorite–longshot bias is among the most documented results in betting
  markets. Systematically selling longshots collects the lottery premium.
  Counterparty: the lottery buyer.
- WHY STILL ALIVE: regulatory friction — no institution will touch it; capacity
  is $10–50k; resolution/oracle risk is unpriceable for a fund's ops team.
- DATA: Polymarket/Kalshi APIs and historical order books (existence and depth
  of free history (verify); fee schedules (verify)).
- EVIDENCE STATUS: documented in sports betting; plausible-extension to
  event-contract venues.

**F2. Closed-end fund deep-discount reversion in the sub-$400M tail**
- MECHANISM: CEFs have no creation/redemption arb to anchor price to NAV; the
  holder base is retail; year-end tax selling and bear-market panic widen
  discounts mechanically. Activist arbitrageurs (Saba-style) patrol only funds
  large enough to fight for board seats (verify their practical floor). The
  tail mean-reverts from extremes with no professional competition.
  Counterparty: the retail panic seller, and the absent arbitrageur.
- WHY STILL ALIVE: capacity too small for anyone who could do it
  professionally; data assembly (daily price + NAV + distributions per fund) is
  tedious enough that nobody retail does it correctly.
- DATA: CEFConnect / cefdata.com daily NAVs and discounts (free tiers confirmed
  to exist; historical depth, API terms, and dead-fund retention (verify));
  exchange prices via yfinance; N-PORT/N-CEN filings as a slow PIT cross-check.
- EVIDENCE STATUS: documented-but-niche (the CEF discount puzzle is classic
  literature; reversion-from-extremes profitability net of costs (verify,
  Pontiff-era work)).

**F3. Exchange-listing pump/fade in crypto**
- MECHANISM: FOMO retail buys the listing-day pump on a major-exchange listing;
  day-2 onward fades. Counterparty: the listing-day chaser.
- WHY STILL ALIVE: claim is it isn't anymore — kept to be killed on the record.
- DATA: exchange listing announcements (timestamped blog/API), data.binance.vision.
- EVIDENCE STATUS: documented-but-niche historically (verify); decay after
  publication is the expected finding.

---

## PHASE 2 — ATTACK

**The meta-kill first, because it executes half the list.** Trial #2 is this
project's own proof that universe survivorship converts nothing into something
(net SR 0.82 → −0.01). Every US small/microcap idea above requires a
point-in-time universe **with price histories for the dead names**. The universe
*list* is solvable free — EDGAR filer indexes retain dead companies forever —
but their *prices* are not: yfinance drops delisted tickers, and microcap
delisting is frequent and MNAR (they die *because* of the returns we'd be
missing — law #7 territory; the log's delisting-bound exercise showed even the
S&P-universe version of this hole needs CRSP to close fully). Any idea that
lives where companies die, and can't bound that bias, is dead on arrival at
free-data fidelity. This single argument kills or maims A1, A3, B3, D1, D2,
D3, E2, E3.

**A1 microcap PEAD — KILL.** Meta-kill (dead microcap prices unavailable), plus
liquidity mirage: microcap closing prints sit inside 1–5% effective spreads
(verify) that the backtester's 10 bps assumption would laugh at. The 25 bps
spread setting wouldn't save it; honest microcap costs need their own model.
Resurrect only with paid CRSP-grade data — i.e., not this project, this decade.

**A2 crypto tail reversal — KILL.** The reversal premium in the tail *is* the
market-maker's rent; collecting it requires being the maker. At taker fees
(~7.5–10 bps (verify)) plus tail effective spreads, weekly turnover ≈ 50×/yr
shreds it — the repo's own drag curve says 3.46×/yr needs 1.4%/yr gross at $1M;
scale that intuition by 15×. Delisted-pair retention in the dumps unconfirmed
(verify) → exchange-side survivorship on top. The data is feasible; the
economics are not.

**A3 small-cap weekly reversal — KILL.** Textbook Novy-Marx–Velikov cost
mortality, documented before I was tempted. ~50×/yr turnover in names with
10–50× the S&P's spread. The "patient limit orders" defense is adverse-selection
denial: the limit orders that fill are the ones you'll regret.

**B1 S&P deletion reversal — SURVIVES, barely, ranked last.** The cynical read:
Greenwood–Sammon shows the forced-flow price impact is mostly arbitraged to
zero since 2010, so what overshoot is left to revert? Surviving distinctions:
(i) their result is the announcement→effective window; the post-effective
reversal is a different object; (ii) discretionary deletions ≠ migrations to
MidCap (their own decomposition says migrations drove much of the "disappearance");
(iii) any rebound must beat a size- and trailing-return-matched control, because
deleted names are mechanically small losers and small-loser bounce is not an
index effect. Why it survives despite a weak prior: the data is already in the
repo, the harness validation is zero-trial, one real run settles it, and a
clean null is a write-up section ("we tested the famous one against a matched
control and it's gone — here's the control methodology"). Cheap test, high
information, low expectation — priced accordingly in Phase 3.

**B2 Russell reconstitution — KILL.** Crowding, documented: predictability
attracted professional front-runners (the Greenwood–Sammon mechanism), and prop
desks with rank-day microstructure infrastructure own the residual. One event
per year; a solo researcher brings nothing to this fight but order flow to be
eaten.

**B3 tax-loss bounce — KILL.** Statistical power, fatally: ~16 Decembers in any
free sample = 16 correlated draws, whatever the within-December cross-section
pretends. No honest path to t_NW ≥ 2 plus a DSR hurdle at N ≥ 8. Plus the
meta-kill on the illiquid tail where the effect supposedly lives.

**B4 lockup expiry — KILL.** The documented anomaly approximately equals the
documented borrow fee (verify) — the market cleared, the fee is the anomaly. A
retail account can't even get locates on hot IPOs. This trade exists for prime
brokerage clients, which is the institution this project doesn't have.

**C1 tail funding extremes — PARK (process kill, not idea kill).** H2 is
registered and unrun. Spinning up a correlated variant of the same mechanism
before H2 executes is trial-proliferation — two draws at one hypothesis,
N-inflation with extra steps. Parked with a condition: if H2 runs and shows
gross-but-not-net carry in the top-30, the tail version (wider funding, worse
fills — both effects larger) becomes a candidate registration with a fee-first
power analysis. Until then, nothing.

**C2 borrow-fee cross-section — SURVIVES, in collection-only form.** The attack
on the *trading* version is decisive: the documented shorting premium is gross
of fee; net of fee the no-arbitrage answer is ~zero, and a retail account
captures neither the lending fee nor reliable locates. But the attack on the
*dataset* fails: fee history cannot be honestly backfilled (free archives have
unknown completeness (verify)), the live cron is already committing daily
artifacts, and the marginal cost of one more non-fatal snapshot step is ~zero.
This is H5's logic pointed at a second vendor surface. Zero trials. Registered
below as collection-only.

**C3 FINRA short-sale volume — KILL.** The documented signal is weak, fast, and
disputed (verify); fast signals die by the project's own turnover arithmetic.
Tedium is real but tedium without a surviving mechanism is a hobby, not an edge.

**D1 Form 4 clusters — KILL.** The tedium moat evaporated years ago:
OpenInsider-class free screeners serve cluster-buy lists to every retail trader
(verify current state). Where the literature says the signal is strongest
(opportunistic buys in microcaps), the meta-kill applies; where the data is
clean (PIT S&P 500), the literature says the signal is weakest. The idea lives
in exactly the intersection we can't build honestly.

**D2 8-K Item 4.02 — KILL.** Low-hundreds of events/yr (verify), mostly
microcaps (meta-kill), short side needs borrow on names already cratering
(unavailable or at fees that are the anomaly), and the drift evidence is
largely from the pre-2010 regulatory regime (verify). Event scarcity also means
the DSR hurdle is unreachable at honest event counts.

**D3 XBRL dilution intensity — KILL as standalone; salvage one limb as an H1
note.** The L/S needs its short leg precisely in unborrowable pump-names; the
borrowable-universe version is a documented factor in every commercial library,
presumably arbitraged. But the *data observation* survives: XBRL cover-page
share counts are a free, filing-timestamped, PIT-safe series, and share-count
growth belongs in H1's fundamental feature set when H1's data audit happens.
Cost today: zero. (This is a note to H1's eventual registration, not a new
hypothesis — H1 is unrun and its registration is unedited.)

**D4 revision intensity — N/A by protocol.** Already H5. Stage 2 only. Any
"idea" here before September is a peek with a costume on.

**E1 halt resumption — KILL.** The event lives intraday; daily bars contain its
corpse. Historical halt archives with honest timestamps are unconfirmed
(verify), so it'd be collect-forward — and the collection budget is better
spent on C2, where the mechanism-to-trade path doesn't run through unborrowable
halted pumps and knife-catching skew.

**E2 delist-to-OTC rebound — KILL.** The purest liquidity mirage on the list:
OTC "prices" are quotes nobody can hit at size, spreads run 5–20% (verify), and
any documented rebound is plausibly bid-ask bounce inside the measurement. The
documentation itself is suspect — survivorship inside the data source.

**E3 NT late filers — KILL.** Tradable side is short, on distressed smalls,
without borrow. The long side ("avoid red-flag names") improves nothing — trial
#3 already taught this book that there is no long alpha to protect. A risk
screen in search of a portfolio.

**F1 prediction-market longshots — KILL, with respect.** The most "no one else"
idea here and the honest referee still wins: the pipeline's machinery (daily
cross-sectional panel, dollar-neutral, SR/DSR on marked returns) doesn't fit
binary terminal payoffs with thin, gappy order books — marking those books
daily *is* the liquidity mirage. Fees and oracle risk unverified (verify).
Selling longshots is also short-a-lottery-ticket skew: years of pennies, then a
mispriced "impossible" resolves true. Revisit only as a separate side-harness
with its own falsification design — not as a guest in this one.

**F2 CEF tail discounts — SURVIVES, ranked first.** Attacks and why they fail
to kill: (i) *Crowding?* Saba-class activists need scale and board fights;
sub-$400M bond CEFs at 12% discounts are beneath their cost structure (verify
floor). (ii) *Survivorship in the data source?* Real risk (dead/merged funds
vanish from CEFConnect (verify)) — but note the direction: CEF deaths are
mostly liquidations/open-endings at NAV, a *positive* terminal event for a
discount-long. Omitting them plausibly biases the backtest *against* the
strategy. That argument is a hypothesis to verify in Stage 1, not a fact — but
it's the only idea on this list where the suspected bias direction is
conservative. (iii) *Liquidity mirage?* Partial hit: tail CEF spreads are wide;
answered with a 25 bps+ pre-declared cost assumption and an ADV floor, sized in
Phase 3. (iv) *The discount is fair value (value trap)?* The test is reversion
from *extremes vs own history*, not absolute discount levels — and the paired
NAV-staleness control below exists because stale NAVs manufacture fake
extremes. (v) *Distributions?* CEF returns are mostly distributions;
price-only returns would fabricate losses — total-return construction is
pinned by known-answer tests before any run. Survives to registration.

**F3 listing pump/fade — KILL.** Patterns published on crypto Twitter decay in
weeks; exchanges changed listing mechanics repeatedly (verify); shorting day-2
requires a perp that often doesn't exist yet. Survivorship in the
announcement-to-data join. Nothing left to test.

**Phase 2 scorecard: 15 killed, 1 parked (C1), 1 protocol-blocked (D4), 3
survive → 17 of 20 removed from contention.** The recurring assassin was not
crowding — it was (a) free data's missing dead names and (b) borrow on the
short leg. Both are now named, permanent screens for future ideation: *an idea
for this project must either live where nothing dies (crypto, index members,
funds), die conservatively (bias direction provably against us), or be
collection-forward (we build the PIT record ourselves).*

---

## PHASE 3 — REGISTER: three registration-ready specs

Numbered to follow H1–H5. Written in the repo's template; **not** entered in
`writeup/preregistered_hypotheses.md` — paste on sign-off only. N = 7 until
then. Recommended order: H6 (richest), H7 (free), H8 (cheapest closure of a
famous question).

### H6: Closed-end fund discounts mean-revert from extremes in the small-fund tail, net of costs
- Status: REGISTRATION-READY (not registered). Two-stage, on the H5 pattern.
- Economic prior: no creation/redemption mechanism anchors CEF price to NAV;
  the tail's holder base is retail; tax-season and panic selling widen
  discounts mechanically; activist arbitrage has a scale floor the tail sits
  under (verify). The honest prior includes the classic finding that discounts
  are *persistent* — the claim is reversion from z-extremes vs own history,
  not convergence of all discounts.
- Exact testable prediction (Stage 2 will freeze it; declared now): funds in
  the widest decile of `z = (discount_t − mean_252d) / std_252d` outperform the
  narrowest decile over the next 21 trading days on **total** (distribution-
  inclusive) returns, net of pre-declared costs.
- **Stage 1 — descriptive census, ZERO trials** (no signal-vs-forward-return
  computation permitted, on H5's precedent):
  - Assemble daily price + NAV + distribution panel, all US-listed CEFs;
    sources: CEFConnect/cefdata free tiers (depth and terms (verify)), sponsor
    NAV publications, yfinance prices, N-PORT/N-CEN as slow PIT cross-checks.
  - Dead-fund census: how many CEFs merged/liquidated/open-ended in sample;
    are they retained in the source (verify); terminal outcome distribution.
    This tests the "bias direction is conservative" argument before it is
    relied on. If dead funds are absent AND the conservative-direction argument
    fails the census → kill before any trial.
  - NAV-staleness audit: publication frequency per fund (daily vs weekly);
    illiquid-asset CEFs with stale NAVs manufacture fake discount volatility —
    quantify before Stage 2 sets the universe filter.
  - Total-return machinery pinned by known-answer tests (a 10% special
    distribution must not register as discount widening or a price crash).
  - Universe/capacity facts: fund count, mcap and ADV distributions, spread
    estimates (close-to-close vs intraday range proxy).
- Stage 2 (registered AFTER Stage 1, BEFORE any test run; one real run = one
  trial at then-current N): universe = US CEFs, mcap < $400M, ADV ≥ $250k,
  daily-published NAV only (filter justified by the staleness audit); feature =
  discount z vs own 252d trailing; h = 21d, monthly rebalance; the PRIMARY
  spec is the single-feature rank portfolio (law #5: the baseline IS the
  strategy; ML only as a registered comparison, never a substitute); label =
  21d forward total return residual vs CEF-universe equal-weight; costs ≥ 25
  bps one-way pre-declared (Stage 1 spread census may push this higher, never
  lower); shorts restricted to the most-liquid tercile of the rich-z side, and
  if Stage 1 shows tail-CEF borrow is fantasy (verify via IBKR availability —
  synergy with H7's snapshots), the registered fallback is long-tilt vs an
  equal-weight CEF hedge basket, declared before the run, not after.
- Point-in-time safety: discount_t uses NAV published at or before t (NAV
  publication lag respected explicitly — using same-evening NAV for a 4pm
  decision is look-ahead unless publication time is verified); distributions
  ex-date aligned; one-line PIT argument per field required at Stage 2.
- Known artifacts requiring paired controls (cf. the vol-regime IC artifact,
  log 2026-06-12):
  - NAV staleness: re-run signal on the daily-NAV-only subuniverse; effect must
    persist (control is part of the single registered run's report, not a
    second trial).
  - Label-shuffle control on the H4 pattern: the same construction on
    date-shuffled labels must show no lift.
  - Seasonality confound: report with and without Dec–Jan (one run, pre-declared
    subreport).
- Success criteria (Stage 2 will freeze; declared intent): right-signed t_NW ≥
  +2 on the IC AND net SR > 0 with DSR ≥ 0.95 at the then-incremented N AND
  survives removal of the largest-mcap decile AND both paired controls pass.
- Kill criteria: Stage 1 census kills (dead-fund bias unboundable and
  direction not conservative); staleness control fails (artifact, not alpha);
  effect lives only in unborrowable shorts (not implementable — logged as
  null-for-us); t_NW < 2 (null, logged, no z-window scans — each scan is +1
  trial and none is authorized).
- Honest capacity: **$50k–$250k.** Sub-$400M funds at ADV-respecting
  participation. That is the entire point — say it plainly: this is a personal-
  book edge, and "too small for Saba" is the why-it-exists.

### H7: Daily borrow-fee/short-availability snapshots — collection-only registration (zero trials)
- Status: REGISTRATION-READY (not registered). Collection-only; the H5
  two-stage structure, second instance.
- Economic prior: borrow fee is the observable price of concentrated negative
  information (shorting-premium literature, gross of fee (verify)). The honest
  prior for THIS book: (a) a risk veto for any future short leg — never short
  a name whose fee exceeds the expected alpha (H6's short tercile is a direct
  consumer); (b) a candidate feature for future registrations; (c) an
  unbackfillable public dataset, which is the moat.
- Action on sign-off: add a non-fatal step to the live cron (the `revisions.py`
  pattern — a measurement bug can never cost a trading cycle): snapshot IBKR
  short-stock availability (fee rate, shares available, lender count) for the
  PIT S&P membership + live-book names + (if H6 proceeds) the CEF universe;
  commit `borrow_{asof}.json` with the prediction logs (write-once, same
  `assert_write_once` discipline). Source: IBKR's published availability data
  and downloadable rates (confirmed to exist); exact endpoint/FTP format
  (verify at implementation).
- Point-in-time safety: trivial — snapshot at t, committed by CI at t; cannot
  reference anything later. Nobody can backfill it, including us; the public
  commit history is the verifiable timestamp (H5's argument verbatim).
- Stage 2 condition: ≥ 60 cycles accumulated; exact analysis config and success
  criteria registered in `preregistered_hypotheses.md` BEFORE the first look.
  No analysis of any kind before Stage 2 is written down.
- Success criterion (Stage 1, operational only): snapshots present for ≥ 95% of
  trading cycles; schema stable; zero trading-cycle failures attributable to
  the collector.
- Kill criteria: source access becomes unreliable/ToS-blocked (verify terms at
  implementation) → stop collection, log the attempt; that outcome costs
  nothing and is still a write-up sentence about data moats.
- Honest capacity: n/a — this is a dataset, not a trade. Cost: ~zero marginal
  infrastructure on an existing cron.

### H8: Discretionary S&P 500 deletions earn positive post-effective-date returns vs a matched control
- Status: REGISTRATION-READY (not registered). One trial when run. The weak
  prior is declared as part of the registration: Greenwood–Sammon (J. Finance
  2025) report the announcement→effective deletion effect at ~0.1% for
  2010–2020; this tests the *post-effective* window, a different object, with
  an expectation tilted toward a clean, citable null. The trial is bought for
  its information value either way, at near-zero data cost.
- Economic prior: trackers sell at the effective close with a tracking-error
  loss function; if residual overshoot exists post-arbitrage, it reverts in
  the following weeks. The counter-prior (GS) is stated above and in the
  failure interpretation — both outcomes are write-up sections.
- Universe + dates: the repo's existing PIT membership change table,
  2010→present. Removals classified by reason (M&A/bankruptcy/restructuring
  vs discretionary index-committee deletion) from the change-table reason
  column + press releases. **Classification and event-count touch no price
  data → zero trials.** Pre-trial gate: if discretionary deletions < 100
  events, power is insufficient — kill at zero cost before any run.
- Construction: event-time overlapping portfolio. Enter at the close of
  effective date + 1 (pre-declared; avoids the depressed forced-flow close and
  its mechanical bounce). Equal-weight all active events; hold 60 trading
  days. The hedge IS the paired control: short an equal-dollar portfolio of
  non-deleted members matched per-event on log-mcap and trailing 126d return
  (deleted names are mechanically small losers; any rebound must beat matched
  losers, not zero). This emits a daily return series the existing
  SR/DSR/t_NW machinery consumes unchanged.
- Label definition: 60d post-event return in excess of the matched control
  basket; portfolio-level daily P&L net of 10 bps + the existing impact model.
- Point-in-time safety: effective dates are exchange facts; entry t+1 uses
  only information public at t; matching variables use data through t only.
  One residual hazard, named: the Wikipedia change table is itself a
  retrospective source — Stage 0 includes reconciling a random 20-event sample
  against contemporaneous press releases (no price data → zero trials).
- Cheapest kill test (zero trials, runs first): synthetic planted-event mode —
  inject a known post-event drift onto pseudo-events in the planted panel; the
  event harness must recover it (DSR > 0.95) and must reject on pseudo-events
  in the noise panel. The harness is not trusted until it passes both — same
  law as everything else in this repo.
- Success criteria: right-signed event-level t_NW ≥ +2 vs matched control AND
  event-portfolio net SR with DSR ≥ 0.95 at the then-incremented N AND the
  effect is not concentrated in 2010–2014 (pre-declared subperiod report
  inside the single run — decay since GS's sample is the expected failure
  mode).
- Kill criteria: control rebounds comparably (it was a small-loser effect
  wearing an index costume); t_NW < 2 → null, logged. NO entry-timing
  variants, NO holding-period scans, NO "announcement-date version" afterward
  — each is +1 trial and none is authorized by this registration.
- Failure interpretation: free replication-with-control of a famous
  disappeared anomaly — directly citable in the write-up next to trial #2's
  survivorship exhibit as the project's second in-house reproduction of the
  published record.
- Honest capacity: ~$1–10M (deleted names are recently-large caps with
  real ADV (verify per-event)); capacity is not the constraint — the
  constraint is ~15–25 events/yr of lumpy P&L, which is exactly why it can
  still exist.

---

## Confirmed sources (everything else above is tagged inline)

- Greenwood & Sammon, "The Disappearing Index Effect," Journal of Finance
  (2025): additions ~7.4% (1990s) → <1% (2010s); deletions ~−0.1% 2010–2020;
  migration and front-running decompositions.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4294297
- IBKR Short-Securities Availability (availability, indicative fee rates,
  historical rates downloadable):
  https://www.interactivebrokers.com/en/trading/short-securities-availability.php
- Binance public data dumps (klines/trades/funding incl. futures):
  https://github.com/binance/binance-public-data and https://data.binance.vision/
- CEF data surfaces: https://www.cefconnect.com/ and https://cefdata.com/
  (free tiers exist; historical depth, dead-fund retention, ToS all (verify)).

*End of document. N = 7. Nothing above has touched real price data in
anger, and nothing will until a registration is signed.*
