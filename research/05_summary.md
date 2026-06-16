# 05 — Handoff: ranked recommendation

**One page. What to do next, what to fact-check first, and the cheapest way to kill the top idea.**

### First, the correction that reframes everything
The brief said **N = 7**. The live record says **N = 11** — trials #8 (H2 carry, real but
DSR-failing), #9 (H8 deletions, null), #10 (H9 tail carry, null), #11 (H6 CEF reversion,
overturned by an entry-lag artifact) all ran 06-13→06-15. **The next trial is #12.** Its
DSR≥0.95 hurdle is **≈0.90 net annual SR** for a ~15-yr daily strategy (≈0.83 weekly);
the +1-trial cost is only ~0.01 SR (sample length dominates). All STEP-4 success criteria
use N=12. (Details + every other code/log discrepancy in `00_repo_audit.md` §5.)

### The recommendation
**Pursue H1 quality (draft DA) as the next graded trial (#12) — but gate it behind one
data decision and one zero-trial kill test, and run the CEF-convergence census (DB) in
parallel as the cheap hedge.**

**Why DA is #1.** It is the only candidate that is *structurally* suited to this project's
hard-won lesson. Trials #2–7 died of **cost mortality** (turnover × 10 bps on a weak edge);
quality rebalances every 63 days, so costs are benign. It has the **lowest DSR hurdle**
(~0.90 at 15 yr), the harness is **already built, tested, and data-gated**
(`run_fundamentals.py`), and the prior — gross profitability — is the **best-surviving
anomaly class in the literature** (Novy-Marx 2013). The outcome is win/win for a
credibility-first portfolio: either the project's **first graduation**, or its **cleanest
null** ("free-data-fidelity quality is arbitraged in large caps too"), extending the
headline survivorship/decay story to fundamentals.

**Why not the others, first.**
- **DB (CEF dated-convergence)** is the most *novel* idea and fixes exactly what killed #11
  (it anchors to a public terminal-NAV date, so a 1-week lag can't break it). But its
  capacity is **$25k–$150k** — a structural-premium *demonstration*, not a business. Run
  its **zero-trial census now**; it either dies for free or earns trial #13.
- **DC (borrow Δfee)** and **DD (revision intensity)** are the project's two genuine,
  unbackfillable data moats — but neither is runnable until **≥60 cycles (~Sept 2026)**.
  Design now (drafts DC/DD), run later.

### `(verify)` items to fact-check BEFORE registering DA — none are optional
1. **Survivorship-safe fundamentals source + cost.** Confirm Sharadar SF1 (or WRDS/Compustat)
   is reachable and its price (free tier? academic? the WRDS sponsorship emails out to
   Wahal/Aragon). This is the literal gate — H1 cannot creditably run on the free SEC source
   (73–75% coverage). *(verify Sharadar pricing/access; verify it is both delisted-inclusive
   AND filing-date point-in-time, not period-end.)*
2. **Is GP/A still alive in LARGE caps, value-weighted, post-publication?** Novy-Marx's
   premium must hold value-weighted in large caps (not just EW small caps) and not have
   decayed since 2013. *(verify against the original Table and the Open Source Asset Pricing
   panel's post-2013 large-cap GP/A long-short.)*
3. **Accruals leg is probably dead post-2002** (Green–Hand–Soliman). *(verify; if dead, drop
   it or run it only as a separately-reported hedged claim — never blended into GP/A.)*
4. **Non-financials coverage.** GP/A excludes ~40% of the index (financials/REITs have no
   COGS). *(verify enough non-financial large-cap names remain for stable quintiles.)*

### The single cheapest experiment that could kill the top idea (DA)
**Before spending a dollar on data or a trial:** pull the **Open Source Asset Pricing**
free panel (Chen–Zimmermann, `openassetpricing.com`) and check whether **gross
profitability's value-weighted, large-cap long-short return is already ≈0 in the
post-2013 (post-publication) period.** If the most-cited quality signal has already decayed
to insignificance in large caps on a clean academic dataset, H1 will almost certainly be a
null on your noisier free-adjacent data — **kill it before buying Sharadar.** Zero trials,
zero dollars, one afternoon. (Mirror of trial #2's logic: let a cheap, clean dataset
falsify the prior before the expensive run.)

### Recommended order of operations (all owner-sign-off-gated)
1. **Now, zero-trial:** OSAP GP/A decay check (kills or greenlights DA) **+** the CEF
   dated-convergence census (kills or greenlights DB).
2. **If DA greenlit:** resolve the data source → run `run_fundamentals.py --source
   compustat/sharadar` as **trial #12** (machinery + data gates first).
3. **Background:** keep H5/H7 collection running untouched; the DC/DD Stage-2 specs are
   ready for the ~Sept 2026 ≥60-cycle gate.

---

## Adversarial review of THIS session (where my enthusiasm may have leaked past the evidence)
*(Required by the working style. I am my own most cynical referee here.)*

1. **DB may be dressed-up #11.** I argued the public terminal date defeats the entry-lag
   artifact — but I have **not** verified that a PIT-identifiable set of ≥50 forward-dated
   term/liquidation funds even exists (the census is *completed* 2021–26 deaths). If
   forward-dated terminal events are rare or not cleanly PIT, DB collapses to a thin,
   underpowered study. I flagged this as the census kill test, but my ranking of DB at #2
   assumes the census passes — that's optimism. **Treat DB as "promising IF the census
   delivers ≥50 PIT events," not as a near-certainty.**
2. **DA's payoff framing ("win/win") is a rhetorical comfort.** A null is only a "win" for a
   *portfolio narrative*; it is still a null, and I'm recommending spending real money
   (Sharadar) and trial #12 on something whose most likely outcome — given McLean–Pontiff and
   the project's seven prior nulls — is another null. The cheapest-kill OSAP check exists
   precisely because I don't fully trust my own optimism here.
3. **I leaned on the literature's reported magnitudes.** Every empirical number in
   `01_literature_scan.md` is search-snippet-sourced and tagged `(verify)`; I did not read the
   full papers. The loan-fee "4.01%/month" and the 13D "7–8%" especially deserve scrutiny
   before any weight is placed on them. If a referee asked "did you read Engelberg et al.
   in full?" the honest answer is no — I verified the papers *exist* and said so.
4. **The DSR-hurdle arithmetic is mine, not the repo's.** I reproduced
   `graduation_hurdle.py`'s N=10 numbers exactly (good evidence it's right), but I
   reimplemented the formula in the sandbox rather than running the repo's own script
   against its environment. A discrepancy in `var_sr` conventions would shift the hurdles.
5. **"Capacity is HIGH for DA" is asserted, not measured.** I did not run the `--capacity`
   sweep (it would be a no-new-trial infra run, but still a real-data run I'm not authorized
   to execute). $100M+ is a plausible-large-cap claim, not a computed one. Tagged as such.
6. **I may be over-weighting the data-moat ideas (DC/DD) because they flatter the project's
   self-image** ("the one edge McLean–Pontiff can't touch"). The honest counter is that the
   loan-fee signal is gross-of-fee and the revision signal is, by the project's own prior, a
   risk control — both may net to nothing. I tried to encode that skepticism into their kill
   tests, but my prose about "moats" is warmer than the evidence warrants.

Net: the strongest, most defensible single action is the **zero-trial OSAP GP/A decay check**
— it costs nothing and could save a data purchase and a trial. Everything warmer than that is
a prior, labeled as one.
