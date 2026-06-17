# 06 — H1 verification: the OSAP cheapest-kill test + the (verify) gate (2026-06-16)

**What this is.** The verified layer on top of the deep-research drafts in this
folder (`00_repo_audit.md` … `05_summary.md`). Those drafts recommended H1
(fundamental quality) as the next graded trial (#12), *gated* behind one
zero-cost kill test: pull a clean academic panel and check whether gross
profitability (GP) has already decayed post-publication — if it has, kill H1
before spending a dollar on data (the trial-#2 logic, applied pre-emptively).
This document records the result of actually running that test (and the other
`(verify)` items) via a 20-agent verification workflow, with every number
sourced or computed — **no estimates**.

> **Process honesty flag.** The workflow's own synthesis agent concluded
> "OSAP never run / could_not_compute" — it confused *"no OSAP script exists in
> the repo"* with *"the check was not performed."* It was performed, live, by
> two independent agents (below). The synthesis verdict is **overridden** by the
> underlying computed evidence. (This is exactly why a human reads the dossier.)

---

## 1. The cheapest-kill test RAN — and GP survived it

Two agents independently installed the `openassetpricing` package (Chen &
Zimmermann, Oct-2025 release), pulled the real GP long-short portfolio returns,
and **reconciled to the published 1963–2010 benchmark** (computed 0.30%/mo vs
OSAP-documented 0.31%/mo, t≈2.2–2.5) to prove the pipeline before reporting.

**Gross profitability, value-weighted long-short, monthly returns (computed):**

| construction | full-sample NW t | post-2013 NW t | post-2013 mean | note |
|---|---|---|---|---|
| VW quintile LS (`op`, original Novy-Marx) | 3.88 | **2.87–3.09** | 1.00%/mo | premium ~tripled OOS |
| VW quintile LS, **large-cap** (ME > NYSE 20th pct) | 3.78 | **3.21** (Sharpe ~1.0) | 0.97%/mo | *stronger* in large caps post-2013 |
| VW **extreme-decile** LS | 3.49 | **1.88** | 0.72%/mo | the one cut that weakens |

Pre-2013 → post-2013 mean **rose** (~0.33%/mo → ~0.97–1.00%/mo); robust to
excluding the 2020–22 value rebound (t=3.20). Corroborated by **Novy-Marx &
Medhat (2025)**: the RMW-style profitability factor *strengthened* (31bps→60bps,
Sharpe 1.16, 2007–23), explicitly defying McLean–Pontiff.

**Verdict: the kill test did NOT fire.** GP is alive — there is no free reason
to abandon H1.

## 2. The nuance that actually decides H1's odds

GP is alive *as a factor*. But the **specific cell H1 can implement** — large-cap,
value-weighted, **net of cost**, raw long-short — is the *weakest* version:

- **In-sample, large-cap raw VW GP was always marginal: 0.26%/mo, t=1.88**
  (Novy-Marx 2013, Table 4, largest size quintile). The large-cap result he
  defends is an **FF3 alpha** (0.50%/mo, t=3.90) driven by a −0.51 HML loading —
  *model-dependent*, not a tradable raw return.
- **Hou–Xue–Zhang (2020):** VW NYSE-decile GP earns 0.38%/mo (t=2.62) — fails the
  t≥3 hurdle; and with the **lagged-assets denominator** (the arguably-correct
  one) it **collapses to 0.16%/mo, t=1.04 — insignificant.** Construction- and
  denominator-sensitive.
- OSAP numbers are **gross/paper** (no costs, no risk adjustment). H1's bar is
  net-of-cost.

**Translation:** the factor is real; the large-cap, net-of-cost cut H1 trades is
marginal → the most likely H1 outcome is still a **credible null** at the N=12
hurdle. That is a fine, story-extending result — but H1 is "worth an eyes-open
trial," not "a likely graduation."

## 3. Skeptic panel (adversarial verification, majority vote)

- **"GP decayed post-2013"** → refuted 3/3 (does not survive). Consistent with §1.
- **"Accruals is dead post-2002, drop it"** → refuted 2/3. The blanket claim is
  *overstated* (accruals survives in microcaps/distress/international subsets) —
  but the real reason to drop it is **subsumption by profitability** (§4), not death.
- **"H1 strictly requires PAID data"** → refuted 2/3. The gate is real (free SEC
  is survivorship-blocked), but WRDS-Compustat-via-sponsorship is a *free* path —
  so "must buy Sharadar" is too absolute.

## 4. Spec corrections forced by the verification

1. **Drop the accruals leg; use cash-based operating profitability (CBOP).**
   Accruals is *subsumed* by profitability (Ball, Gerakos, Linnainmaa & Nikolaev
   2016) — `z(GP/A) − z(accruals/A)` double-counts. CBOP (operating profitability
   net of accruals) captures the accruals information cleanly and raises the
   Sharpe more than an accruals+OP pair.
2. **The "~40% financials/REITs excluded" figure is wrong — it is ~21% by count**
   (107/503: Financials 76 + Real Estate 31). ~396 non-financial names → ~40–79
   per quintile (8× the ≥10 bar; ≥38/quintile even under all coverage haircuts).
   Quintiles are very viable. Encode "exclude **both** GICS Financials *and* Real
   Estate" (REITs left Financials in the Sept-2016 GICS reclassification).
3. **Construction must be pre-declared:** value-weighted **quintile** (not decile —
   the decile weakens to t=1.88), and **current-vs-lagged assets** decided up
   front (lagged is the insignificant cut — declare current as primary, report
   lagged as a robustness leg expecting it may be weak).
4. **Data dimension:** Sharadar SF1 must use an **As-Reported** dimension
   (ARQ/ART, keyed on `DATEKEY` = filing date); MRQ/MRT reintroduces restatement
   look-ahead. WRDS alternative = **Compustat PIT/Snapshot**, not vanilla
   (restated) Compustat Fundamentals.
5. **The flow-annualization bug is confirmed:** the harness mixes 10-Q quarterly
   flows with point-in-time stock Assets (understates ratios; the machinery gate
   bypasses it because the synthetic uses GP/A as a level). Annualize flows
   (TTM/ART or 10-K-only) before dividing by Assets — fix before any graded run.

## 5. The data gate (verified)

- **Sharadar SF1:** reachable by an undergrad (Non-Professional tier, no
  institutional gatekeeping); **delisting-inclusive: yes**; **filing-date-PIT:
  yes** *conditional on the ARQ/ART dimension*. **Exact price: COULD NOT VERIFY**
  (hidden behind login on every pricing surface — the agent refused to invent it;
  confirm before any spend). No free academic tier; the free "sample" is annual-only.
- **WRDS / Compustat PIT-Snapshot:** equal-or-better data quality, **free to the
  student if the institution subscribes and a faculty member sponsors** — the ASU
  sponsorship emails (Wahal/Aragon + others) are exactly this path.

## 6. Repo-grounding corrections (the drafts' own arithmetic)

- `scripts/graduation_hurdle.py` **only computes N=9 and N=10** — it cannot
  produce the N=12 figures the summary cites. Computed by inverting the repo's own
  `quantlab.metrics` (reproduces the published N=10 table byte-for-byte):
  - **N=12 daily (~15-yr, n_obs 3378): hurdle ≈ 0.90** net annual SR (symmetric;
    0.93 carry-like). ✓ matches the draft's "~0.90".
  - **N=12 weekly: ≈ 0.86, NOT 0.83.** The 0.83 figure is the **N=10** value — a
    stale trial count. Correct the summary.
  - **"+1 trial ≈ +0.01 SR"** ✓ confirmed (N11→12: +0.0116 daily / +0.0110 weekly).
- N=11, trials #8–#11, and H1=PROPOSED-blocked all confirmed against the repo.

## 7. Recommendation

The kill test did not fire → **H1 stays the best graduation candidate and a
worthwhile trial #12** — but go in honest:

- **If WRDS sponsorship lands (free):** run H1 #12 with the tightened spec (CBOP,
  VW large-cap, net-of-cost, ARQ data, annualization fixed). Likely a credible
  null, possibly a marginal pass — worth it either way at zero data cost.
- **If it requires real $ (Sharadar paid):** the modest odds make the spend
  genuinely questionable. The honest exhibit can be written *for free* — "GP is
  alive as a factor, but the large-cap, net-of-cost cut we could trade is
  marginal (t=1.88 in-sample; insignificant on lagged assets) — consistent with
  our cost-mortality thesis." That extends the story without spending.

Not a free kill; not a slam dunk. A worthwhile, eyes-open trial whose base rate
remains "credible null." The tightened registration is drafted as the
**2026-06-16 PRE-DATA amendment** in `writeup/preregistered_hypotheses.md` (H1),
ready the moment the data question resolves.

## 8. Provenance & caveats
- Source: 20-agent verification workflow `h1-greenlight-verification`
  (run `wf_a271336c-76a`), 2026-06-16. All numbers computed live or sourced; OSAP
  scratch files were written to a scratch dir, no tracked repo files touched.
- **Verified:** OSAP GP returns (computed, reconciled to the published benchmark);
  Novy-Marx 2013 Table 4 (extracted from the PDF); the non-financial counts
  (computed from `data_cache/sp500_current.parquet`); the N=12 hurdles
  (inverted from `quantlab.metrics`, reproducing the published N=10 table).
- **Sourced-but-not-from-primary-PDF (tagged):** Hou–Xue–Zhang Gpa/Gla numbers
  (working-paper text + RePEc) and Ball-et-al. CBOP subsumption (abstract/RePEc;
  primary PDF 404'd). McLean–Pontiff GP-specific magnitude: **could not verify**
  (only their cross-sectional averages).
- **Could not verify:** the exact current Sharadar SF1 price.
- The deep-research drafts `00`–`05` in this folder carry their own `(verify)`
  tags (e.g. the loan-fee/13D magnitudes); treat them as drafts — this document
  (`06`) is the verified layer.

### Sources
- Open Source Asset Pricing (Chen & Zimmermann) — https://www.openassetpricing.com/data/ ; SignalDoc + `openassetpricing` pip package.
- Chen & Zimmermann (2021), *Open Source Cross-Sectional Asset Pricing*, FEDS 2021-037.
- Novy-Marx (2013), *The Other Side of Value*, JFE 108(1):1–28 (Table 2, Table 4, Sec 2.3).
- Novy-Marx & Medhat (2025), *Profitability Retrospective*, NBER WP 33601.
- Hou, Xue & Zhang (2020), *Replicating Anomalies*, RFS 33(5):2019–2133 (NBER w23394; Sec 3.2.4).
- Ball, Gerakos, Linnainmaa & Nikolaev (2016), *Accruals, cash flows, and operating profitability*, JFE 121(1):28–45.
- Green, Hand & Soliman (2011), *Going, Going, Gone? The Apparent Demise of the Accruals Anomaly*, Mgmt Science 57(5):797–816.
- McLean & Pontiff (2016), *Does Academic Research Destroy Stock Return Predictability?*, J. Finance 71(1):5–32.
- Sharadar SF1 docs — https://www.sharadar.com/ ; Nasdaq Data Link.
