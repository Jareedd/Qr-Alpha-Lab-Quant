# 00 — Repo Audit: what qr-alpha-lab actually does, as the code enforces it

**Author:** research collaborator (session 2026-06-16)
**Scope of this file:** STEP 0 of the research engagement. Ground-truth the pipeline
against its own logs *before* any web research or ideation. Read the code, not just
the prose; flag every place the log/brief and the code disagree.

**Method / provenance (so you can trust or distrust each claim):**
- Read in full by me: `research_log.md`, `writeup/preregistered_hypotheses.md`,
  `ROADMAP.md`, `CLAUDE.md`, `requirements.txt`, `README.md`, and the source of
  `metrics.py`, `backtest.py`, `models.py`, `features.py`, `validation.py`,
  `synthetic.py`, `regime.py`.
- Verified by a delegated read with line-citations (then sanity-checked against the
  docs by me): `risk.py`, `risk_model.py`, `universe.py`, `live.py`, `registry.py`,
  the data-layer modules (`perp_data`, `cef_data`, `cef_deaths`, `fundamentals_data`,
  `cik_history`, `borrow`), `scripts/run_pipeline.py`, `.github/workflows/ci.yml`,
  and the test suite filenames + key tests.
- **Not personally executed:** I ran nothing on real data and nothing that spends a
  trial (per the boundaries). I did read `results/*.json` values. Where I report a
  number it comes from a log row or a results artifact, cited as such.

---

## 1. What the pipeline actually does (end to end)

A cross-sectional, walk-forward, falsification-first equity research stack, with three
later "arms" bolted on (crypto-perp carry, CEF reversion, event studies) and an
unused-but-built execution engine. The core data flow:

1. **Universe** (`universe.py`): point-in-time S&P 500 membership reconstructed by
   scraping Wikipedia's "List of S&P 500 companies" current table + *changes* table
   (`WIKI_URL`, `pd.read_html`), then walking the changes backward from today's
   members — for each change before the cursor, `members.discard(added)` /
   `members.add(removed)`. Produces dated membership intervals + an as-of-today GICS
   sector map. A legacy `--data yfinance` mode (today's members only, fully biased) is
   kept on purpose as the survivorship contrast.
2. **Data** (`data.py`): yfinance loader, parquet cache keyed on an MD5 of the sorted
   ticker list (the cache-key-collision bug was found and fixed, log 2026-06-10).
3. **Features** (`features.py`): five cross-sectionally z-scored price features —
   `mom_12_1`, `mom_6_1`, `rev_1m`, `vol_3m`, `pct_52w_high`. All use `.shift()` of past
   prices; returns use `pct_change(fill_method=None)` so a dead name yields NaN, never a
   phantom 0%. Z-scores are taken over index members only (member-masked).
4. **Labels** (`features.py::build_labels`): forward `horizon`-day return, z-scored;
   `residualize=True` subtracts `beta_t * mkt_fwd` with **past-only** rolling betas.
5. **Validation** (`validation.py`): expanding-window walk-forward with an embargo
   (`train = [start, test_start − embargo)`, `embargo_days=21 ≥ horizon`). This is the
   purged-CV discipline; standard k-fold is correctly rejected as invalid for
   overlapping labels.
6. **Models** (`models.py`): Ridge (α=10) default; `ridge_cv` does nested per-roll α
   selection on the *training window only*; GBR and a small MLP exist to complete a
   model-class ablation, not as hail-marys.
7. **Backtest** (`backtest.py`): dollar-neutral decile (top/bottom quantile) long-short,
   equal-weight within leg; `daily_w = weights.ffill().shift(1)` so weights chosen at t
   earn from t+1 (no lookahead); linear cost = `Σ|Δw| · bps/1e4`; one-way annualized
   turnover reported.
8. **Neutralization** (`risk.py`): `neutralize_predictions_by_sector` (GICS demean) and
   `beta_neutralize_weights` (Gram-Schmidt projection to `Σw=0` and `w·β=0`), plus a
   `risk_report` that *measures* realized beta / sector tilts rather than asserting them.
9. **Evaluation** (`metrics.py`): per-date Spearman rank IC, Newey-West HAC t-stat
   (Bartlett kernel, lags = horizon), Sharpe, max-DD, PSR, and the **Deflated Sharpe
   Ratio** benchmarked against the expected max of N noise trials.
10. **Capacity** (`impact.py`): square-root impact (k=1) on trailing dollar-ADV; sweeps
    cost drag by AUM. Not in the headline backtest by design.
11. **Falsification harness** (`synthetic.py`): planted / noise worlds (+ regime, perp,
    CEF, quality worlds for each arm), each with a *paired true-null* defined explicitly.
12. **Live** (`live.py` + `scripts/`, `live.yml`): daily Alpaca **paper** cycle, trains
    on fully-labeled history only, logs the full prediction cross-section *before* orders,
    shadow-logs a momentum control arm, and writes a data-revision fingerprint. Monitored
    read-only by `monitor.py`.
13. **Engine** (`sizing.py`, `risk_model.py`, `combine.py`, `limits.py`, `execution.py`,
    `engine.py`): a complete signals→orders portfolio engine that sizes on the *lower
    confidence bound* of the Sharpe (fractional-Kelly-under-uncertainty), so a near-zero
    edge sizes to ~zero. Built, tested, **never fed a real graduated signal** (there isn't
    one yet).

The culture is the product. Per `CLAUDE.md`: *credibility beats performance*; if a
result improves, the first hypothesis is a bug or leak. The deliverable the project is
optimizing for is a defensible research note, not a P&L.

---

## 2. The laws as the CODE enforces them (prose claim → code reality)

| # | CLAUDE.md law (prose) | What the code actually does | Verdict |
|---|---|---|---|
| 1 | No leakage; features past-only; embargo ≥ horizon | `validation.WalkForwardSplitter` train window ends `embargo_days` before test; `features` use `.shift()`/`pct_change(fill_method=None)`; `backtest` does `ffill().shift(1)`; residual labels use past-only `rolling_beta`. A same-day-exploit test + a foresight counter-test pin it. | **Enforced.** |
| 2 | Falsification gate: planted DSR>0.95, noise rejected | `scripts/run_pipeline.py` `--fail-if-dsr-below` / `--fail-if-dsr-above` `sys.exit` on violation; `ci.yml` runs `planted --fail-if-dsr-below 0.95` and `noise --n-trials 20 --fail-if-dsr-above 0.5` on every push. Artifacts: planted DSR 0.9919, noise 0.00043. | **Enforced (in CI).** |
| 3 | Count every trial; the log's N feeds `--n-trials`; never reset | `metrics.deflated_sharpe_ratio(returns, n_trials=...)` consumes N. **But `--n-trials` is a manual CLI argument** — *nothing reads "N = 11" out of `research_log.md`.* Two runner scripts hardcode N as a literal default (`run_cef_reversion.py: N_TRIALS_DEFAULT = 11`, `run_fundamentals.py: default=12`). `registry.py` *does* refuse unregistered real-data runs. | **Partially enforced** — see §5 discrepancy D2. The count is human discipline, not a code-read. |
| 4 | Costs in every result; turnover headline | `backtest.run_backtest` always nets costs off gross; `summary()` always emits `annual_turnover`, `skew`, `dsr`. | **Enforced.** |
| 5 | Beat equal-weight and 12-1 momentum, net, OOS | `baselines.py`; `run_pipeline` emits `beats_mom_baseline` + `baseline_mom_ic` every run. | **Enforced.** |
| 6 | Never weaken a failing test | Cannot be proven from a snapshot, but the log repeatedly *adds* tests after bugs and the suite carries known-answer/equivalence pins. No evidence of weakening. | **Consistent with the record.** |
| 7 | Never fabricate real data; synthetic only in `synthetic.py` | All synthetic generators + the two scenario tools (`inject_delisting_returns`, `inject_post_event_drift`) live in `synthetic.py` and tag artifacts (`_dlret±NN`). | **Enforced.** |
| 8 | Reproducibility from config + seed | Seeds throughout; cross-machine drift measured (~1e-7 from yfinance re-adjustment, ~1e-14 from BLAS) rather than assumed. | **Enforced, with measured caveats.** |

**The DSR machinery, precisely (matters for STEP 4).** `deflated_sharpe_ratio(net,
n_trials)` defaults `var_sr = 1/n_obs`, computes `sr_star = expected_max_sharpe(N,
var_sr, n_obs)` (a Gumbel max-of-N: `sd·[(1−γ)·z₁ + γ·z₂]`, `z₁=Φ⁻¹(1−1/N)`,
`z₂=Φ⁻¹(1−1/(Ne))`), and returns `PSR(net, sr_benchmark=sr_star)`. PSR uses the
**per-period** SR (no annualization) and adjusts for skew/kurtosis — so the DSR is
annualization-agnostic, and negative skew (the carry trades) genuinely lowers it.
`n_obs` is the count of return observations: **daily for the equity/perp arms, weekly
(≈ #weeks) for the CEF arm** — which is why CEF faced a higher Sharpe hurdle.

---

## 3. The trial ledger and current hypothesis status (this is the real state)

**`research_log.md` header: `N = 11`.** Verified internally consistent: every numbered
trial row has a matching `results/metrics_*.json` (`n_trials` field matches the trial
number; DSR matches). Synthetic/infra rows correctly do **not** increment N.

| Trial | Date | Hypothesis | Headline OOS result | Verdict |
|---|---|---|---|---|
| #1 | 06-10 | 5 price features, *biased* yfinance universe | IC 0.033, net SR **0.82**, DSR 0.998 @N=1 | Survivorship illusion — not trusted |
| #2 | 06-10 | same, *PIT* S&P 500 | IC 0.0052 (t_NW **0.54**), net SR **−0.01**, DSR 0.29 | **The alpha was survivorship bias.** Project's best exhibit |
| #3 | 06-10 | + sector/beta neutralization | net SR −0.38, DSR 0.01 | Nothing hiding under factor exposure |
| #4 | 06-10 | 63d horizon (turnover cut) | IC −0.0278 (t_NW −1.95), net SR −0.35 | Sign-flip; **explicitly refused** (no salvage) |
| #5 | 06-10 | residualized labels | IC +0.0225 (t_NW +1.91), net SR **−0.77** | IC ≠ P&L — measurable IC that loses money |
| #6 | 06-10 | GBR nonlinear | net SR −0.12, DSR 0.04 | Costs eat the thin edge |
| #7 | 06-10 | MLP | net SR −0.28, DSR 0.008 | Model-class ablation complete: all null |
| #8 | 06-13 | **H2** crypto-perp funding carry | IC t_NW **−3.54** (right-signed), net SR **+0.87**, DSR **0.865**, skew −1.87, maxDD −74% | **First non-null — refused on DSR.** Decayed 2.28→~0.4 as basis farms scaled |
| #9 | 06-13 | **H8** discretionary S&P deletions | net SR −0.04, t_NW −0.10, DSR 0.05 | Clean null; matched control rebounds +2.6% (Greenwood–Sammon) |
| #10 | 06-14 | **H9** perp-tail carry (ADV 31–150) | IC t_NW **−3.62**, net SR **−0.13**, DSR 0.024 | Real signal, unmonetizable: funding +1.23 / price −0.85, 20bps finishes it |
| #11 | 06-15 | **H6** CEF discount reversion | net SR 1.11, DSR **0.999** nominal — **OVERTURNED** | Entry-lag sweep collapses 1.11→0.10 at 1-wk lag (bid-ask-bounce artifact); shuffle seed-fragile. Does not graduate |

**Hypothesis registry status** (`preregistered_hypotheses.md`):
- **H1** (quality: GP/A + accruals/A): PROPOSED, **blocked on a survivorship-safe data
  source**. Free SEC XBRL gives ~73–75% coverage (current-only ticker→CIK map drops dead
  names); GP/A caps ~59% (financials/REITs have no CoGS). Harness fully built + gated
  (`fundamentals.py`, `fundamentals_data.py`, `make_quality_panel`), refuses to run on the
  free source. Needs Compustat/CRSP. `accruals/A`-only is ~93% coverable, sector-agnostic.
- **H2** perp carry: RUN (#8), criteria not met (DSR only).
- **H3** fixed dispersion-gated momentum: PROPOSED, **trap check**, weak prior.
- **H4** causal-HMM vol-regime gate: PROPOSED, machinery built + falsification-validated;
  **requires a paired artifact control** because the vol-regime IC artifact (below) would
  otherwise masquerade as exactly this hypothesis.
- **H5** vendor data-revision intensity: PROPOSED, two-stage, collection running since
  2026-06-11; Stage-2 at ≥60 cycles (~Sept 2026). Honest prior: a *risk* signal, not alpha.
- **H6** CEF reversion: RUN (#11), overturned by implementability.
- **H7** borrow-fee snapshots: REGISTERED collection-only; first snapshot 2026-06-12
  (498/500 names; fee p50/p90/p99 = 1.2%/17.3%/163.7%). Stage-2 at ≥60 cycles.
- **H8** deletions: RUN (#9), null.
- **H9** tail carry: RUN (#10), null.

**Net state: 11 trials, zero graduated strategies.** The two "closest" are both
instructive deaths — H2 (real premium, fails DSR + crash-skewed) and H6 (looked like a
pass, killed by an entry-lag/implementability diagnostic the frozen criteria forgot to
include).

**The vol-regime IC artifact (cited repeatedly downstream — STEP 4 must respect it).**
Documented in `synthetic.py:63–72` and the 2026-06-12 log entry: on **signal-free** data,
residualized-label momentum diagnostics show ~+0.13 IC in stressed (high-vol) states when
betas are dispersed (beta-estimation error × 2.5× vol), and ~+0.06 even with uniform betas
(label-machinery × regime interaction). On real data this looks *exactly* like "momentum
works conditionally on volatility" (H3/H4). Consequence: any regime-conditional claim must
pass a paired label-shuffled artifact control registered before the run.

---

## 4. Documented limitations (assembled from README + log + code)

1. **Survivorship is reduced, not eliminated.** PIT universe still drops dead names with
   no Yahoo history (`sp500_pit_coverage.json` quantifies it: 661/810 priceable ≈ 81.6%;
   149 unpriceable).
2. **Delisting returns missing**; bounded not imputed (`--delisting-return`, Shumway worst
   case −30%). Measured effect on trial #2 ≈ ΔSR +0.006.
3. **Sector data is as-of-today** (Wikipedia); departed names → UNKNOWN bucket; PIT GICS
   needs paid data.
4. **Betas are estimated, not known**; realized residual beta drifts to ~0.05 mean (p95
   0.23) between rebalances — measured per run.
5. **Free daily data only**; vendor retroactively re-adjusts history (the revision monitor
   exists because of this). Headline costs are linear; sqrt impact lives only in `--capacity`.
6. **Live-IC record gaps**: cycle #1 logged weights only; control arm + revision fingerprint
   start 2026-06-11; first live-vs-backtest IC claim needs >23 matured 21d cycles (~2026-07
   onward).
7. **H1 cannot be run creditably on free data today** (the survivorship hole, reprised in
   fundamentals).

---

## 5. DISCREPANCIES — flagged loudly (the point of STEP 0)

### D1 — **The brief says N = 7; the live record says N = 11.** (Material.)
The engagement brief states "N = 7 trials." That was true through the 2026-06-12 log
entries. Since then **four real-data trials have been run and logged**: #8 (H2, 06-13),
#9 (H8, 06-13), #10 (H9, 06-14), #11 (H6, 06-15). Both `research_log.md:3` and the bottom
line of `preregistered_hypotheses.md` now read **N = 11**, and all four artifacts exist in
`results/`. **The brief's N is stale.**

*Why it matters:* STEP 4 asks for the DSR hurdle "at the incremented N." The next trial is
**#12**, so its DSR benchmark must be `expected_max_sharpe(N=12, …)`, **not** N=8. The
`graduation_candidates_2026-06-14.md` hurdles were computed at N=10; at N=12 they are very
slightly higher (the max-of-N term grows ~like √(2 ln N), so N=10→12 lifts the per-period
`sr_star` only ~5–8%). I will compute exact N=12 hurdles for STEP 4. **I will honor the
*spirit* of the boundary regardless: design only, run nothing, write nothing into
`preregistered_hypotheses.md`.**

### D2 — **Law #3 is only half-mechanized.** (Material for honesty claims.)
`CLAUDE.md:17` promises "If the log says 87 trials, the DSR uses 87." In code, `--n-trials`
is a **hand-entered CLI argument**; nothing parses the log's N. `run_cef_reversion.py`
hardcodes `N_TRIALS_DEFAULT = 11` and `run_fundamentals.py` hardcodes `default=12` —
literals that must be manually bumped or the DSR silently uses a stale N. What *is*
mechanized is the *registration* gate (`registry.require_runnable_registration` refuses any
real-data run not naming a PROPOSED hypothesis). So the honest framing for an interviewer is
"trial **registration** is code-enforced; the trial **count** feeding the DSR is enforced by
discipline + a hardcoded default, not by reading the log." A small script that reads `N =`
from `research_log.md` and passes it would close the gap. (Design note only — I am not
modifying code.)

### D3 — **Artifact filenames are keyed by H-number, but the ledger counts by trial-number.**
H8 → trial #9, H9 → trial #10, H6 → trial #11. So `results/metrics_h8_events.json` is
*trial #9*, `metrics_h6_reversion.json` is *trial #11*. A reader who maps "h8" → "trial #8"
will mis-attribute results. The log calls this out (2026-06-14 integrity row); the filenames
were not renamed. Cosmetic but a real foot-gun for the write-up.

### D4 — **README "Known limitations" / counts are stale.** (Cosmetic.)
`README.md` still says "58 tests" and "six logged trials … best DSR 0.04" and its limitations
section describes only the equity pipeline — no mention of the H2/H6/H8/H9 trials, the regime
artifact, the engine, or the ~189-test suite the log now cites. The log is the current source
of truth; the README trails it. (The log itself flags `research_note_draft.md` as awaiting an
owner rewrite.)

### D5 — **The test suite is not 100% green in every environment.** (Minor, disclosed.)
The log's recent "tests green" figures (e.g. 176/189) explicitly **exclude 3 pre-existing
dashboard failures** attributed to a matplotlib 3.9.2 `print_png(width=)` API change. I did
not run pytest (read-only), so I take the log at its word — but "all tests green" is true only
modulo those 3 environment-specific dashboard failures. Worth stating plainly in the note.

### D6 — **PII committed in source.** (Housekeeping.)
The SEC fair-access User-Agent string embeds a personal email:
`_UA = "qr-alpha-lab research Jared@how.co"` in `cef_deaths.py`, `fundamentals_data.py`, and
`cik_history.py`. SEC asks for a contact UA, so *a* contact is correct — but if this repo is
the public portfolio artifact, consider a role address. (Not a methodology issue.)

### Investigated and found NOT to be discrepancies (recorded for honesty)
- **Alpaca vs `requirements.txt`:** I suspected the live path needed `alpaca-py`/`requests`
  not listed in `requirements.txt`. False alarm — `live.py` is a stdlib-`urllib`-only REST
  wrapper (`class AlpacaPaper`) with a hard paper-only guard (`if "paper" not in self.base:
  raise RuntimeError`), pinned by `test_live.py`. The minimal `requirements.txt` is correct,
  and "no new dependency for the broker client" is a deliberate, defensible choice.
- **CEF weekly Sharpe annualization:** `metrics.sharpe()` defaults to √252, which *would*
  mis-annualize a weekly series — but `run_cef_reversion.py` uses a dedicated
  `_weekly_summary` passing `periods=52`, and the DSR is annualization-agnostic anyway.
  Correct.
- **"neutralization" module:** the brief lists a `neutralization` source file; there is none.
  Neutralization lives in `risk.py` (inline, equity backtest) and `risk_model.py`
  (engine-level). Naming mismatch in the brief, not a code gap.

---

## 6. What this means for the rest of the session (STEP 1–5)

1. The cheap, picked-over corners are **already honestly exhausted**: price-only large-cap
   equity (trials #2–7), index-deletion events (#9), and two cuts of perp carry (#8, #10).
   New ideas must not relitigate these.
2. The project's genuine, **non-relitigable assets** are (a) its *survivorship-aware* PIT
   plumbing, (b) its *paired-control* falsification discipline, and (c) two **proprietary,
   unbackfillable datasets it is already accruing** — the data-revision fingerprints (H5) and
   the daily borrow snapshots (H7) — plus the CEF dead-fund census. STEP 2 should bias hard
   toward edges that exploit (c) and toward structurally-protected, capacity-constrained
   corners institutions can't bother with — the H6 *thesis* (no-arb structural premium) was
   right even though that *implementation* died on microstructure.
3. Two scars must be encoded as permanent screens: **free data's missing dead names**
   (survivorship), and **borrow cost on any short leg**. A third, newly sharp from trial #11:
   **any reversion/cross-sectional-mean idea must register an entry-lag / implementability
   gate as a pass/fail criterion**, not a post-hoc diagnostic.
4. DSR hurdles for STEP 4 use **N = 12** for the next trial. Prefer zero-trial routes
   (synthetic gate, descriptive Stage-1) wherever possible — every real run is a scarce,
   sign-off-gated increment.
5. H1's blocker is a *data-access* problem, not an idea problem. If a survivorship-safe
   fundamentals source can be reached cheaply (STEP 1 must check), H1 is the lowest-hurdle
   live candidate (slow rebalance → benign costs/skew, 15yr sample → ~0.9 net-SR bar).
