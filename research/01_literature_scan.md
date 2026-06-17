# 01 — Literature & Data Scan

**STEP 1.** For each open direction (H1, H2, H3/H4, H5) and the new directions I'll
carry into the backlog, the 3–5 most relevant sources in my own words with a real,
checkable citation, plus a catalog of free/near-free data sources with access paths and
gotchas.

**Citation discipline (read this first).**
- Every source listed here appeared in an actual web-search result this session; I give
  a URL you can open. I did **not** cite anything I could not see in a result.
- Every *empirical magnitude* (a return, a t-stat, a coverage %, a fee) is tagged
  **(verify)** with exactly what to check, because search snippets paraphrase and I have
  not read the full papers. Treat untagged statements as my framing, not fact.
- Where a paper's exact venue/year/author list was not unambiguous in the snippet, I say
  so and tag it (verify). A wrong citation is worse than no citation in this project.

---

## Part A — The methodology pillars (the project already owns these; restated so the new work inherits them)

1. **McLean & Pontiff (2016), "Does Academic Research Destroy Stock Return
   Predictability?", *Journal of Finance* 71(1):5–32.**
   [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12365) ·
   [SSRN 2156623](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2156623).
   Predictor returns are **~26% lower out-of-sample and ~58% lower post-publication**
   across **82–97** predictors (verify the exact decay figures and predictor count
   against Tables I–II). *Use:* the base-rate prior that any published signal is largely
   arbitraged; the reason new work must favor *structural* or *proprietary-data* edges.

2. **Harvey, Liu & Zhu (2016), "…and the Cross-Section of Expected Returns," *RFS*.**
   (Appeared as a project reference; not re-searched this session — **verify** the exact
   cite.) *Use:* the multiple-testing hurdle (t≈3 for a "new" factor) that motivates the
   DSR-at-N discipline.

3. **Hou, Xue & Zhang (2020), "Replicating Anomalies," *Review of Financial Studies*.**
   [NBER w23394](https://www.nber.org/system/files/working_papers/w23394/w23394.pdf) ·
   [SSRN 2961979](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2961979).
   With NYSE breakpoints + value-weighting, **~65% of 452 anomalies fail |t|≥1.96, ~82%
   fail at 2.78, and ~96% of the "trading frictions" category fail** (verify all three).
   Microcap anomalies are "more apparent than real" once costs bite. *Use:* the single
   most important cull screen — any idea whose edge concentrates in microcaps or
   equal-weighted small names is presumed illusory net of costs until proven otherwise.

4. **Bailey & López de Prado (2014), "The Deflated Sharpe Ratio"** and **López de Prado
   (2018), *Advances in Financial Machine Learning* (ch. 7, purged/embargoed CV).** (Project
   references; the DSR and embargo are implemented in `metrics.py`/`validation.py` — see
   audit §2.) *Use:* already the spine of the harness.

5. **Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning," *RFS*.**
   (Project reference.) *Use:* the result that shallow nets, not deep ones, win at this
   S/N — already reflected in the MLP ablation (trial #7).

---

## Part B — Open directions

### H1 — Fundamental quality (gross profitability, accruals)

- **Novy-Marx (2013), "The Other Side of Value: The Gross Profitability Premium,"
  *JFE* 108(1):1–28.**
  [NBER w15940](https://www.nber.org/papers/w15940) ·
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0304405X13000044) ·
  [EconPapers](https://econpapers.repec.org/RePEc:eee:jfinec:v:108:y:2013:i:1:p:1-28).
  Gross profits / total assets predicts the cross-section about as well as book/market;
  it is a *non-financials* construct (gross profit needs a COGS line). (verify: that the
  premium is value-weighted and survives in large caps, not just equal-weighted small
  caps — this is the crux for the project's S&P-500 universe.)
- **Sloan (1996)** accruals anomaly + its decay literature: **"Going, Going, Gone? The
  [Apparent] Demise of the Accruals Anomaly"** (verify authors — likely Green, Hand &
  Soliman, *Management Science* ~2011)
  [Penn State record](https://pure.psu.edu/en/publications/going-going-gone-the-apparent-demise-of-the-accruals-anomaly/) and
  **Lev & Nissim, "The Persistence of the Accruals Anomaly"**
  [PDF](http://www.columbia.edu/~dn75/The%20Persistence%20of%20the%20Accruals%20Anomaly.pdf).
  The accrual effect was strong for decades and **decayed after ~2002** as hedge-fund
  capital and analyst cash-flow forecasts arrived (verify the post-2002 decay magnitude
  and whether it is now reliably positive at all). *Implication:* accruals/A is the
  sector-agnostic, 93%-coverable variant (audit §3) but may already be dead.
- *Read of the direction:* H1's blocker is **data access, not idea quality** (audit §3,
  §6). The economic prior (slow-rebalance quality → benign cost mortality) is the
  cleanest fit to the project's "low-turnover or bust" lesson. Everything hinges on a
  **survivorship-safe, filing-date-PIT** fundamentals source (see Part D).

### H2 — Crypto-perpetual funding carry (RUN as trial #8; real-but-DSR-failing)

- **Inan, "Predictability of Funding Rates,"
  [SSRN 5576424](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5576424).** Finds
  out-of-sample predictability of next-period funding (BTC on Binance/Bybit) via
  double-AR models, but **stability varies over time** (verify the OOS R²/horizon and
  that it is funding-*rate* predictability, not return predictability — they are
  different objects, exactly the trap trial #8/#10 navigated).
- **"Designing funding rates for perpetual futures in cryptocurrency markets,"
  [arXiv 2506.08573](https://arxiv.org/abs/2506.08573)** and **Ackerer, "Perpetual
  Futures Pricing," *Mathematical Finance*
  [Wiley](https://onlinelibrary.wiley.com/doi/10.1111/mafi.70018).** Funding =
  premium + interest components; the premium term enforces mean reversion of the basis.
  *Use:* the theoretical backing for "the null is funding-fully-priced," which the
  synthetic `priced_carry` world encodes.
- **"Exploring Risk and Return Profiles of Funding Rate Arbitrage on CEX and DEX,"
  *ScienceDirect*
  [link](https://www.sciencedirect.com/science/article/pii/S2096720925000818).** Maps the
  spot-perp basis trade economics across venues (verify the net-of-cost APRs and which
  venues/period). *Use:* informs the cross-venue extension below.
- *Read:* the premium is **real but decaying** (trial #8: SR 2.28 in 2020–21 → ~0.4
  post-2021 as Ethena-style basis farms scaled) and **crash-skewed**. McLean–Pontiff in
  crypto. A fresh H2-style trial must be a *new* pre-registration on held-out data, never
  a salvage of #8.

### H3 / H4 — Regime-conditional momentum (registered trap checks, unrun)

- **Moreira & Muir (2017), "Volatility-Managed Portfolios," *JF* 72(4):1611–1644.**
  [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513) ·
  [NBER w22208](https://www.nber.org/papers/w22208) ·
  [SSRN 2659431](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2659431).
  Scaling factor exposure down when realized vol is high raised Sharpes for market,
  value, momentum, profitability, and the **currency carry trade** (verify the per-factor
  alphas and that the gains survive transaction costs — the well-known critique). *This
  is the strongest prior for H3/H4*, and notably it is a *time-series vol-timing* result,
  not a cross-sectional regime gate.
- **Daniel & Moskowitz (2016), "Momentum Crashes," *JFE*.** (Project reference; verify
  exact cite.) Momentum crashes cluster in panic/high-vol rebounds — the economic content
  H3/H4 leans on.
- **DeMiguel et al. (2024), "A Multifactor Perspective on Volatility-Managed
  Portfolios," *JF*
  [LBS PDF](https://lbsresearch.london.edu/id/eprint/3716/1/...).** A recent re-examination
  (verify the finding — my recollection is that vol-management's benefit is concentrated
  and not universal once you account for the multifactor structure). *Use:* the
  steelman-the-null source for H3/H4.
- **CRITICAL internal caution (audit §3):** the repo already proved that on **signal-free**
  data, residualized-label momentum shows **~+0.13 IC in stressed states with dispersed
  betas, ~+0.06 with uniform betas** — an artifact that *is* "momentum works conditionally
  on volatility." Any H3/H4-type run is **invalid without a paired label-shuffle artifact
  control registered before the run.** Literature can't rescue this; the harness already
  knows the trap.

### H5 — Vendor data-revision intensity (collection-only, Stage-2 ~Sept 2026)

- **ALFRED (Archival FRED), St. Louis Fed,
  [alfred.stlouisfed.org](https://alfred.stlouisfed.org/).** The macro-data analogue of
  the project's idea: every observation carries `(date, realtime_start, realtime_end)` so
  you can reconstruct "what was known when." (verify: ALFRED archives since ~2006;
  coverage varies by series.) *Use:* the conceptual citation that **data vintages are a
  first-class, study-able object** — exactly the H5 premise, ported to equities via the
  project's own daily yfinance diffs.
- There is a real **macro literature on data revisions and real-time data** (e.g., the
  Philadelphia Fed Real-Time Data Set; Croushore & Stark) — I did **not** surface a
  specific equities paper on "vendor price-revision intensity as a cross-sectional equity
  signal," which is consistent with the project's claim that **the dataset is proprietary
  by construction** (nobody else snapshots daily). (verify by a targeted scholar search
  for "real-time data set macroeconomists Croushore Stark" and for any equity revision
  study; absence strengthens the moat claim, presence sharpens the prior.)
- *Honest read (matches the registration):* the prior is a **risk/quality** signal
  (names whose past keeps changing have noisier features), not an alpha signal. Treat
  Stage-2 as descriptive first.

### H6 — CEF discount reversion (RUN as trial #11; overturned on implementability)

- **Pontiff (1995), "Closed-end fund premia and returns…," *JFE*** and **Pontiff (1996),
  "Costly Arbitrage: Evidence from Closed-End Funds," *QJE*** (verify exact years/venues;
  the search confirmed Pontiff's predictability result and the "costly arbitrage" theme).
  [UT-Dallas PDF "Persistence and Predictability of CEF Discounts"](https://personal.utdallas.edu/~yexiaoxu/CFDDP.pdf).
  Discounts predict future fund returns and the predictability **decays slowly**; the
  dominant driver is the **persistence of the discount itself** (verify). *Use:* the
  structural prior survives even though trial #11's *implementation* died on a one-week
  bid-ask-bounce artifact. The reframe (Part C) is to test **monthly** reversion with a
  **registered entry-lag/implementability gate**, killing the microstructure artifact by
  construction.
- **Lee, Shleifer & Thaler (1991)** investor-sentiment view of CEF discounts (project-adjacent;
  verify). *Use:* the "retail holder base / absent arbitrageur" counterparty story.

### H7 — Borrow-fee / short-availability (collection-only, Stage-2 ≥60 cycles)

- **Engelberg, Evans, Leonard, Reed & Ringgenberg, "The Loan Fee Anomaly: A Short
  Seller's Best Ideas,"
  [SSRN 3707166](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3707166)** (verify
  publication status/venue). Portfolios formed on **borrowing fees** produced larger L/S
  returns than any of 102 other anomalies — reportedly **~4.01%/month, monthly Sharpe
  ~0.66** (verify both, and crucially whether returns are **gross or net of the loan
  fee** — the fee *is* the cost, so net tradability is the whole question).
- **Engelberg, Reed & Ringgenberg (2018), "Short-Selling Risk," *JF*
  [Wiley jofi.12601](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12601).** Shorts
  face loan-recall and fee-spike risk; this *risk* is priced. *Use:* supports H7's honest
  framing as **a risk veto for any short leg + a candidate feature**, not a standalone
  trade. The fee tail the project already measured (p99 ~163%/yr) is the cost wall.

---

## Part C — New directions (with sources), biased toward solo-reachable / structural / data-moat edges

### C-1. Overnight vs. intraday return decomposition
- **Lou, Polk & Skouras (2019), "A Tug of War: Overnight Versus Intraday Expected
  Returns," *JFE* 134(1):192–213.**
  [LSE PDF](https://personal.lse.ac.uk/polk/research/TugOfWar.pdf) ·
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0304405X19300650).
  Momentum is earned **overnight**; value/profitability/investment **intraday**; there's a
  cross-period reversal tied to institutional rebalancing (variation "~2%/month")
  (verify the magnitudes and the 14-strategy decomposition). *Why solo-reachable:* needs
  only daily **open and close** (both in yfinance) — no intraday tick data. *Why alive:*
  it's a *decomposition*, not a standalone factor; the institutional tug-of-war is
  structural. *Gotcha:* overnight returns are mechanically tied to the open auction;
  capacity and the open/close spread are the kill risks.

### C-2. PEAD / earnings drift, honest-timestamp event study in costly-to-trade names
- **Fink (2020), "A Review of the Post-Earnings-Announcement Drift,"
  [Working paper PDF](https://static.uni-graz.at/fileadmin/sowi/Working_Paper/2020-04_Fink.pdf)**;
  decay evidence (Chordia–Subrahmanyam–Tong 2014; Martineau 2022) vs. **Meursault, Liang,
  Routledge & Scanlon (2023)** who find strong PEAD with text-based surprises 2008–2019
  (verify each). Drift is **stronger in low-liquidity, high-cost, low-institutional
  names** — which is exactly the Hou–Xue–Zhang warning that the edge may be a cost mirage.
  *Why it fits the repo:* it reuses the **event-study machinery already built for H8**
  (matched controls, planted-drift synthetic gate). *Free data:* earnings dates from SEC
  8-K (Item 2.02) timestamps; prices from yfinance. *Kill risk:* the drift sits where
  costs are highest — must be tested net, value-weighted, with the deletion-study's
  matched-control discipline.

### C-3. Honest-timestamp filing event studies — insider (Form 4) and activist (13D)
- **Cohen, Malloy & Pomorski (2012), "Decoding Inside Information," *JF* 67(3):1009–1043.**
  [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2012.01740.x) ·
  [NBER w16454](https://www.nber.org/papers/w16454). Stripping **routine** insider trades
  leaves **opportunistic** trades worth **~82 bps/month value-weighted** (verify, and
  verify it survives post-2012 — likely decayed). *Free data:* SEC Form 4 (EDGAR), filed
  within 2 business days — an honest, public timestamp.
- **Brav, Jiang, Partnoy & Thomas (2008)** on Schedule 13D activism:
  [Brav–Jiang review PDF](https://business.columbia.edu/sites/default/files-efs/pubfiles/4126/Hedge%20Fund%20Activism%20A%20Review.pdf).
  **~7–8% abnormal return around the 13D filing**, with abnormal volume in the **10 days
  before** the filing (the disclosure window), and **no long-term reversal** (Bebchuk–Brav–Jiang
  2015) (verify all). **Crucial timestamp gotcha:** the SEC **shortened the 13D deadline**
  (historically 10 calendar days; a 2023–24 rule moved it to ~5 business days — **verify the
  current rule and effective date**). The pre-filing run-up means the *tradable* window is
  post-filing only; the project's "enter the day after the public timestamp" convention is
  the right design. *Why alive:* legally-mandated disclosure creates a clean event; the
  drift, if any, is small and capacity-bound.
- *Read:* both are **event studies with public, honest timestamps** — the project's
  comparative advantage (it already proved discipline on H8/H9). Both are also prime
  Hou–Xue–Zhang cull candidates if the edge needs microcaps.

### C-4. Cross-venue crypto funding dispersion
- **CoinGlass** exposes a cross-exchange funding endpoint
  (`/api/futures/fundingRate/exchange-list`) across Binance/OKX/Bybit/Bitget/dYdX/BitMEX/
  Bitfinex/Gate
  [coinglass.com/CryptoApi](https://www.coinglass.com/CryptoApi); **FundingPulse** (Apify)
  is a cheaper alternative covering Bybit/Binance/OKX/Bitget/dYdX/Hyperliquid with a
  cross-exchange-spread feature
  [apify.com/fraktalapi/funding-pulse](https://apify.com/fraktalapi/funding-pulse).
  (verify: CoinGlass paid tiers ~$29–699/mo; what history depth the free tier gives; both
  are **current-listings-biased** unless you also pull delisted contracts from each
  exchange's own dumps.) *Idea:* the *dispersion* of funding across venues for the same
  asset is a plumbing/limits-to-arbitrage signal (who can't move collateral cross-venue).
  *Why hard:* execution across venues, collateral/withdrawal frictions, and the project's
  own trial-#8/#10 finding that funding is largely priced.

### C-5. Calendar / forced-flow seasonality
- **Tax-loss selling & turn-of-year:** the foundational **BKKM (1983), "Stock Return
  Seasonalities and the Tax-Loss Selling Hypothesis," *JFE*
  [Wharton PDF](https://faculty.wharton.upenn.edu/wp-content/uploads/2014/03/BKKM-JFE1983.pdf)**,
  and institutional tax-loss/turn-of-year evidence
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0165410113000906)
  (verify the modern magnitude — the January effect has **largely decayed**, per practitioner
  reviews). *Why mostly dead:* widely known, decayed, concentrated in small caps
  (Hou–Xue–Zhang territory). Keep only as a *conditioning overlay*, not a standalone trade.
- **Russell reconstitution flow:** **Madhavan (2003), "The Russell Reconstitution Effect,"
  *FAJ*
  [PDF](https://www.hillsdaleinv.com/uploads/The_Russell_Reconstitution_Effect,_Ananth_Madhaven,_Financial_Analysts_Journal,_JulyAugust_2003,_Pages_51-64.pdf).**
  Additions reportedly **~+4.6%** over the reconstitution window driven by predictable
  passive flows; deletions the reverse (verify, and verify the effect **post-2007**, after
  Russell's banding/transparency changes and after arbitrageurs crowded in). *Why likely
  dead:* the most pre-announced, most-arbitraged flow in equities; ~70% of institutional
  AUM benchmarked to Russell means the front-run is itself crowded.
- **CFTC Commitments of Traders positioning:**
  [CFTC COT](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm) — free,
  weekly (released Fri 3:30pm ET, reflects **prior Tuesday**, ~3-day delay). One
  cited study finds **COT signals do not produce significant returns** after the delay
  (verify
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1042443113000723)).
  *Why likely dead for us:* the publication lag + futures-only universe; keep as a *risk
  context* indicator at most.

---

## Part D — Free / near-free data sources (access path + gotchas)

| Source | What / access path | Key gotchas |
|---|---|---|
| **Chen–Zimmermann Open Source Asset Pricing** ([openassetpricing.com](https://www.openassetpricing.com/), [GitHub OpenSourceAP/CrossSection](https://github.com/OpenSourceAP/CrossSection), [SSRN 3604626](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3604626), `pip install openassetpricing`) | ~**200–300+** published cross-sectional predictors + portfolio returns, monthly 1925–2022, with replication code (verify counts/dates). *The single most useful free resource I found.* | Built on CRSP/Compustat, so **firm-level signals are not directly downloadable for free** (portfolio returns and the *recipe* are free; raw firm signals may need WRDS). Monthly only. Great as a **benchmark/control universe** and a "is my signal already known?" check. |
| **SEC EDGAR** — XBRL `frames` & `companyconcept` ([data.sec.gov APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)), full-text search (efts.sec.gov), submissions JSON, Form 4 / 13D / 25-NSE | Free, filing-date-PIT fundamentals + events. Already used by `fundamentals_data.py`, `cef_deaths.py`, `cik_history.py`. | **10 requests/second** hard cap across all `*.sec.gov` (403 + ~10-min IP block if exceeded); requires a contact **User-Agent**. **`company_tickers.json` is current-only** → dead/renamed tickers don't map (the project's 73–75% survivorship hole). GP/A capped ~59% (financials/REITs have no COGS). Frame periods are calendar-aligned, **not** fiscal — "be mindful of reporting dates." |
| **Sharadar SF1/SEP** (Nasdaq Data Link [SF1](https://data.nasdaq.com/databases/SF1); via [QuantRocket](https://www.quantrocket.com/sharadar/)) | **Survivorship-free** US fundamentals + prices, ~1998+, ~20k companies, point-in-time. The clean fix for H1. | **Paid** (verify current pricing/any academic or trial tier; I could not confirm a free sample). This is the most likely "near-free" unlock for H1; treat cost as the gating decision. |
| **yfinance** (already used; `data.py`) | Free daily OHLCV + dividends/splits. | **Survivorship** (dead names vanish), **retroactive re-adjustment** of whole-history series (the reason `revisions.py` exists; ~1e-7 serving noise + real dividend/split rewrites), informal/unsupported API, rate-limit flakiness. |
| **Wikipedia S&P 500 changes** (already used; `universe.py`, [WIKI_URL](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)) | Free PIT membership via the changes table, walked backward. | Community-maintained, sparse pre-~2005; **sectors as-of-today only** (departed names → UNKNOWN); ticker reuse unresolved. |
| **Binance public dumps** (already used; `perp_data.py`, [data.binance.vision](https://data.binance.vision/)) | Free perp OHLCV + 8h funding, **includes delisted contracts** (e.g. LUNAUSDT) → genuinely survivorship-free. | Older months are **headerless** (positional parse needed — a real bug the project fixed); one venue only; funding sign/settlement conventions must be exact. |
| **CoinGlass / FundingPulse** ([CryptoApi](https://www.coinglass.com/CryptoApi), [FundingPulse](https://apify.com/fraktalapi/funding-pulse)) | Cross-venue funding/OI aggregation. | Freemium (CoinGlass ~$29–699/mo; verify free-tier history depth); **current-listings-biased**; aggregator timestamps may not be exchange-native. |
| **FRED / ALFRED** ([alfred.stlouisfed.org](https://alfred.stlouisfed.org/), `fredapi`) | Free macro; **ALFRED gives vintages** (`realtime_start/end`) — true point-in-time macro. | Free API key; ALFRED vintages since ~2006; macro is low-frequency (regime/risk overlays, not cross-sectional alpha). |
| **CFTC COT** ([CFTC](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm)) | Free weekly futures positioning. | **~3-day publication lag**, futures-only, weak/unstable predictive record (verify). |
| **CEFConnect** (already used; `cef_data.py`, [api/v3](https://www.cefconnect.com/api/v3)) | Free CEF NAV/discount; daily for trailing ~1yr, **weekly** further back to ~2012. | **Current-listings-only** (dead funds absent — but the project proved this bias is *conservative* for a discount-long via the SEC 25-NSE dead-fund census). Unofficial API. |
| **IBKR short-stock file** (already used; `borrow.py`, ftp2.interactivebrokers.com `usa.txt`) | Free daily borrow fee / availability snapshot (~20k instruments), self-timestamped (#BOF). | **Unbackfillable** (only exists if you snapshot daily — the H7 moat); one broker's book; FTP. |
| **Stooq** (the log references a `stooq_audit`) | Free EOD equities/indices, sometimes deeper history than yfinance. | Coverage/quality varies by ticker; survivorship; verify adjustment conventions before trusting. |
| **Ken French Data Library** ([Dartmouth](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)) (not searched this session — verify URL) | Free factor and portfolio returns (FF factors, momentum, industry). | US/dev-market factors only; portfolio-level (no firm-level signals); monthly/daily. Great for **neutralization benchmarks and factor-spanning tests**. |

---

## What the scan changes about the plan
1. **The Open Source Asset Pricing panel is a free "is this already known / already
   decayed?" oracle.** Any new cross-sectional idea should be checked against it *before*
   spending a trial — and it doubles as a control universe for a regime/vol-timing test.
2. **H1's unlock is purely a source decision** (Sharadar vs. WRDS/Compustat). If a
   survivorship-safe source is reachable, H1 is the lowest-DSR-hurdle live candidate.
3. **The project's two proprietary datasets (H5 revisions, H7 borrow) are the only edges
   here that McLean–Pontiff structurally *cannot* arbitrage**, because they don't exist
   for anyone who didn't start snapshotting. Those deserve disproportionate weight in
   STEP 2.
4. **Hou–Xue–Zhang is the guillotine for STEP 3:** every microcap/equal-weight/small-name
   idea is presumed a cost mirage. Survivors must work value-weighted in liquid names or
   exploit a structural flow.
