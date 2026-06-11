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
- Status: PROPOSED
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

---

Nothing in this file has been run. N remains 7. Each run requires owner
sign-off, increments N by exactly 1, and gets logged in
`research_log.md` regardless of outcome.
