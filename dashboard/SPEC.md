# Master prompt: the qr-alpha-lab research-integrity dashboard

> Hand this file, verbatim, to the agent/session that builds the dashboard.
> It is self-contained: no other conversation context is required.

---

You are building a **one-page, read-only Streamlit dashboard** for
qr-alpha-lab, a quantitative research project whose explicit philosophy is
**credibility beats performance**. Read `CLAUDE.md` at the repo root first
and treat it as law. The project's headline finding is a defensible NULL
result (no edge in price-only features on an honest universe) plus a live
paper-trading experiment whose purpose is comparing live IC to backtest IC.
The dashboard's job is to make the project's *integrity machinery* visible
at a glance — it is a monitoring instrument and an interview demo, not a
trading UI and not a marketing page.

## Non-negotiable constraints

1. **Strictly read-only.** The dashboard reads `results/`, `results/live/`,
   `research_log.md`, and `README.md`. It never writes to any of them, never
   imports anything that submits orders (`quantlab.live` is off-limits —
   reuse `quantlab.monitor` instead), and never feeds anything back into the
   strategy. If you find yourself adding a button that *does* something,
   stop.
2. **Every number traces to an artifact.** Each panel states its source file
   (small caption under the panel: e.g. `source: results/live/summary_*.json`).
   Never recompute a "better" version of a logged number; display what the
   artifact says. If a number requires computation (e.g. marks from prices),
   reuse the existing functions in `quantlab.monitor` — do not reimplement.
3. **Honest empty states are a feature.** Today the live experiment has ~2
   cycles, 0 matured IC readings, and possibly 0 revision fingerprints. Every
   panel must render something truthful and informative when its data is
   missing or immature ("first live IC matures ~2026-07-10 — 14 trading days
   to go"), never an error, never a fake placeholder chart.
4. **Dependencies:** Streamlit + what the repo already has (pandas, numpy,
   matplotlib). Put `streamlit` in a NEW `requirements-dashboard.txt`; do NOT
   touch `requirements.txt` (the trading path must stay lean — a dashboard
   dep must never be able to break the nightly cycle).
5. **No network calls by default.** Price fetches for mark-to-market run only
   behind an explicit "fetch prices" button, using `quantlab.data.load_prices`
   (which caches). Everything else works fully offline.
6. **Boring, correct code.** One page. `dashboard/app.py` plus at most one
   helper module `dashboard/loaders.py` for pure parsing functions. Pure
   functions get tests (`tests/test_dashboard_loaders.py`) — especially the
   research-log parser. Match the repo's style: typed where it helps,
   docstrings that state assumptions, no cleverness.

## Layout (top to bottom)

### 0. Header strip — "is the machine healthy?"
- **Falsification gate status**: parse `results/metrics_planted_ridge.json`
  (must show DSR > 0.95 → green "planted signal recovered") and
  `results/metrics_noise_ridge.json` (PSR/DSR low → green "noise rejected").
  If either file is missing or violates its bound, show a loud red banner.
  This is the project's differentiator; it goes first.
- **Cycle freshness alarm**: if the newest `results/live/weights_*.csv` is
  older than 1 trading day (use `pandas.bdate_range`, ignore weekends), red
  banner: "no cycle logged for N trading days — investigate before anything
  else". Otherwise a quiet green tick with the last cycle date.
- **Global trial count N**: parsed from the bold line at the top of
  `research_log.md`. Display prominently. This number never goes down.

### 1. Live experiment monitor — the daily panel
Reuse `quantlab.monitor.load_live_records`, `cycle_continuity`,
`realized_live_ic`, `realized_book_returns`, `live_vs_backtest`.
- **Continuity strip**: one small square per weekday since the first cycle,
  green = logged, grey = missing (label: NYSE holidays are not modeled and
  appear grey — that is documented behavior, not a bug).
- **Cycle facts** (latest `summary_*.json`): as-of date, names in book,
  orders sent/failed, broker equity, and equity delta vs previous cycle.
- **Maturity countdown**: cycles logged with predictions, cycles matured
  (a cycle matures `horizon=21` trading days after its as-of date), and the
  date the FIRST live IC becomes measurable. Show the t_NW validity rule
  plainly: "no live-vs-backtest claim before 23+ matured cycles".
- **Live IC vs backtest IC** (only once ≥1 cycle matured): per-cycle live IC
  dots, horizontal dashed line at the backtest's `mean_rank_ic` from
  `results/metrics_sp500_ridge_both_residlabel.json`, and — in the same
  axes, different marker — the momentum **control arm** IC (column
  `baseline_mom_12_1` in `predictions_*.csv`, via
  `realized_live_ic(..., col="baseline_mom_12_1")`). Caption explains the
  control arm in one sentence: it separates "model decayed" from "period
  hostile to everything".
- **Book snapshot**: top 10 longs and shorts from the latest
  `weights_*.csv`, gross/net exposure, and a one-line reminder that the
  broker's equity curve is authoritative, marks are a cross-check.

### 2. Research-integrity ledger — the panel nobody else has
- Parse the trial table from `research_log.md` (rows whose `#` column is a
  number are trials; `—` rows are infra). The parser is a pure function in
  `loaders.py` with tests — markdown table rows, two table blocks, em-dashes
  and bold markers must all survive parsing. If a row fails to parse, show
  it raw rather than dropping it silently.
- **The DSR-vs-N chart** (centerpiece): x = trial number 1..N, y = each
  trial's logged DSR, with the deflation hurdle visualized — recompute the
  expected-max-Sharpe benchmark per N using the SAME function the pipeline
  uses (import from `quantlab.metrics`; do not re-derive). The visual story:
  the bar rises as N grows; one early point (trial #1, biased universe)
  sits high and is annotated "survivorship bias — killed by trial #2".
- **Trial table**: date, hypothesis (truncated, expandable), net SR, IC,
  DSR, turnover, conclusion. Color nothing green; nulls are results too.
- **Pre-registered queue**: list H1–H5 from `writeup/preregistered_hypotheses.md`
  with status only (PROPOSED/RUN/ABANDONED) — visible proof that future
  trials are declared before they run.

### 3. Data-revision monitor — "the vendor rewrites the past"
- From `results/live/revisions_*.json` (may be empty today): time series of
  `frac_price_cells_changed` and `n_return_cells_changed` per cycle, and the
  latest cycle's top affected tickers.
- Caption (one sentence): price-cell changes are mostly benign re-adjustments;
  RETURN-cell changes silently alter features and labels — that is why
  "point-in-time" includes the data-values dimension.
- Empty state: "collection started 2026-06-11; first comparison lands after
  two consecutive snapshots exist".

### 4. Execution reality — costs and capacity
- From `results/capacity_sp500_ridge_both_residlabel.json`: the cost-drag
  curve (annual drag vs AUM, log-x). Annotate: "any future strategy with
  this turnover profile needs ≥1.4%/yr gross alpha at $1M, ≥5%/yr at $100M
  just to exist".
- Per-trial turnover from the ledger beside it — turnover is a headline
  metric in this project, not a footnote.

### 5. Footer — deliberate honesty
- Render the README's "Known limitations" section verbatim (parse the
  section by its heading). The project's stated position is that naming
  limitations is a feature; put them on the dashboard, not behind it.
- Repo commit hash the dashboard is reading from (`git rev-parse --short HEAD`
  via subprocess, read-only), and a line: "every number on this page is
  regenerable from a config + seed; see research_log.md".

## Genuinely-useful extras (include these, resist all others)
- **"What changed last night"** expander: diff of the two most recent
  `weights_*.csv` (names entered/exited the book, biggest weight changes) +
  the latest revision fingerprint summary. This answers the owner's actual
  daily question in one glance.
- **Interview mode** (`?interview=1` query param or sidebar toggle): hides
  the operational panels (freshness alarm, what-changed) and shows only the
  story panels (gate, DSR-vs-N, live-vs-backtest IC, capacity, limitations)
  — the 5-minute demo ordering.
- **Artifact browser** expander: list `results/*.json` + equity PNGs, click
  to view raw. No editing.

## Anti-goals (do not build)
- No auth, no deployment config, no Docker, no database.
- No strategy parameters, no "run backtest" buttons, no model toggles.
- No plotly/altair/bokeh unless matplotlib genuinely cannot do the job.
- No P&L-first hero chart. The gate and the ledger lead; P&L is panel 1's
  smallest element.
- No optimistic rounding, no green-washing: if live IC comes in negative
  when it matures, this dashboard displays it with exactly the same
  prominence.

## Definition of done
1. `pip install -r requirements-dashboard.txt && streamlit run dashboard/app.py`
   works offline on a fresh clone with today's artifacts (2 cycles, 0
   matured ICs, 0–2 revision files) and every panel shows an honest state.
2. `pytest tests/` green, including new loader tests (research-log parser:
   ≥1 trial row, ≥1 infra row, the bold-N header line; limitations-section
   extractor; weights-diff function).
3. No diff to `requirements.txt`, no import of `quantlab.live`, no writes
   outside the Streamlit session.
4. A screenshot of the rendered page checked into `dashboard/` (the README
   gains a one-line pointer + screenshot embed under a new "Dashboard"
   subsection).
5. A research_log.md infra row: what was built, the read-only guarantee, and
   the test count. Full suite re-run noted. (No falsification-gate re-run
   needed UNLESS anything under `src/quantlab/` changed — which it must not.)
6. Commit message explains what was learned, not just what was added. Do not
   push without the owner's approval.

## Session ritual reminder (from CLAUDE.md)
Read `research_log.md` and `ROADMAP.md` before starting; state the phase
(this is stretch-goal work — Phase 7 write-up takes priority if both are
open); append the log entry and run the full test suite before committing.
