"""qr-alpha-lab dashboard: the project's integrity machinery, on one page.

    pip install -r requirements-dashboard.txt
    streamlit run dashboard/app.py

Read-only by construction: this file reads results/, results/live/,
research_log.md, README.md and writeup/ — it writes nothing, imports
nothing that can submit an order (quantlab.live is deliberately not
imported), and offers no control that changes the strategy. The only
network call (price fetch for marks/live IC) hides behind an explicit
button and goes through the same cached loader the research code uses.

Ordering is the project's philosophy as a layout: falsification gate and
trial ledger lead; P&L is a footnote of panel 1.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, _HERE)

import loaders  # noqa: E402  (dashboard/loaders.py)
from quantlab import metrics, monitor  # noqa: E402

HORIZON = 21
# Env overrides exist ONLY so tests can point the page at synthetic
# artifacts (the data-present branches must not first execute in
# production); they change which files are READ, nothing else.
LIVE_DIR = os.environ.get("QRLAB_LIVE_DIR", os.path.join(ROOT, "results", "live"))
RESULTS = os.environ.get("QRLAB_RESULTS_DIR", os.path.join(ROOT, "results"))
BACKTEST_METRICS = os.path.join(RESULTS, "metrics_sp500_ridge_both_residlabel.json")

st.set_page_config(page_title="qr-alpha-lab", layout="wide")


# ---------------------------------------------------------------------------
# Read-only loaders (thin IO wrappers; all parsing logic lives in loaders.py)
# ---------------------------------------------------------------------------

def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def _read_json(path: str) -> dict | None:
    raw = _read(path)
    try:
        return json.loads(raw) if raw else None
    except json.JSONDecodeError:
        return None


def _src(caption: str) -> None:
    st.caption(f"source: {caption}")


@st.cache_data(show_spinner="fetching prices via cached yfinance loader…")
def _fetch_prices(tickers: tuple[str, ...], start: str) -> pd.DataFrame:
    from quantlab.data import load_prices

    return load_prices(list(tickers), start=start, min_coverage=0.0)


# ---------------------------------------------------------------------------
# Shared state assembled once (all offline)
# ---------------------------------------------------------------------------

log_md = _read(os.path.join(ROOT, "research_log.md")) or ""
readme_md = _read(os.path.join(ROOT, "README.md")) or ""
hypo_md = _read(os.path.join(ROOT, "writeup", "preregistered_hypotheses.md")) or ""

n_trials = loaders.parse_trial_count(log_md)
log_rows = loaders.parse_research_log(log_md)
trials = [r for r in log_rows if r.get("kind") == "trial"]

weights_by_date: dict = {}
preds_by_date: dict = {}
if os.path.isdir(LIVE_DIR):
    weights_by_date, preds_by_date = monitor.load_live_records(LIVE_DIR)

today = pd.Timestamp.today().normalize()
interview = st.sidebar.toggle(
    "Interview mode",
    value=st.query_params.get("interview") == "1",
    help="Story panels only: gate, ledger, live-vs-backtest IC, capacity, limitations.",
)
st.sidebar.caption(
    "Read-only dashboard. It cannot trade, cannot re-run anything, and "
    "cannot touch the research artifacts it displays."
)

st.title("qr-alpha-lab — research integrity at a glance")

# ---------------------------------------------------------------------------
# 0. Header strip: is the machine healthy?
# ---------------------------------------------------------------------------

gate = loaders.gate_verdict(
    _read_json(os.path.join(RESULTS, "metrics_planted_ridge.json")),
    _read_json(os.path.join(RESULTS, "metrics_noise_ridge.json")),
)
c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    if gate["ok"]:
        st.success(
            f"Falsification gate: planted signal recovered "
            f"(DSR {gate['planted_dsr']:.4f} > 0.95) · noise rejected "
            f"(DSR {gate['noise_dsr']:.4f} < 0.5)"
        )
    else:
        st.error(
            "FALSIFICATION GATE NOT GREEN — planted "
            f"{'ok' if gate['planted_ok'] else 'FAILED/MISSING'}, noise "
            f"{'ok' if gate['noise_ok'] else 'FAILED/MISSING'}. "
            "Stop trusting every other number on this page."
        )
    _src("results/metrics_planted_ridge.json · metrics_noise_ridge.json")

with c2:
    if not interview:
        if weights_by_date:
            latest_cycle = max(weights_by_date)
            stale = loaders.cycle_staleness(latest_cycle, today)
            if stale <= 1:
                st.success(f"Live cycle fresh — last logged {latest_cycle.date()}")
            else:
                st.error(
                    f"No cycle logged for {stale} trading days "
                    f"(last {latest_cycle.date()}) — investigate before "
                    "anything else. (NYSE holidays are not modeled and can "
                    "account for one of those days.)"
                )
        elif preds_by_date:
            st.warning(
                "Prediction logs exist but NO weights records found — a "
                "cycle may have died between its two writes; check "
                "results/live/ before the next cycle runs."
            )
        else:
            st.warning("No live cycles logged yet.")
        _src("results/live/weights_*.csv")

with c3:
    st.metric("Trials (N)", n_trials if n_trials is not None else "?")
    if n_trials is None:
        st.error("Trial-count line missing from research_log.md")
    st.caption("never resets; feeds the DSR")

st.divider()

# ---------------------------------------------------------------------------
# 1. Live experiment monitor
# ---------------------------------------------------------------------------

st.header("Live paper-trading experiment")
backtest_stats = _read_json(BACKTEST_METRICS)

if not weights_by_date and not preds_by_date:
    st.info("No live records yet — the daily cycle writes results/live/.")
else:
    facts = loaders.maturity_facts(sorted(preds_by_date), today, HORIZON)

    if not interview and weights_by_date:
        cont = monitor.cycle_continuity(sorted(weights_by_date), today)
        fig, ax = plt.subplots(figsize=(10, 0.6))
        colors = ["#2e7d32" if x else "#bdbdbd" for x in cont["logged"]]
        ax.bar(range(len(cont)), [1] * len(cont), color=colors, width=0.9)
        ax.set_yticks([])
        ticks = range(0, len(cont), max(1, len(cont) // 12))
        ax.set_xticks(list(ticks))
        ax.set_xticklabels(
            [str(cont["date"].iloc[i].date()) for i in ticks], fontsize=7
        )
        ax.set_title("cycle continuity (green = logged; grey = missing — "
                     "NYSE holidays are not modeled and appear grey)", fontsize=9)
        st.pyplot(fig)
        plt.close(fig)
        _src("results/live/weights_*.csv dates vs pandas bdate_range")

        summaries = sorted(glob.glob(os.path.join(LIVE_DIR, "summary_*.json")))
        latest = _read_json(summaries[-1]) if summaries else None
        prev = _read_json(summaries[-2]) if len(summaries) > 1 else None
        if latest:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("as-of", latest.get("asof", "?"))
            k2.metric("names in book", latest.get("n_names", "?"))
            # Display what the artifact says: a dry-run summary has no
            # orders keys at all — rendering '0 / 0' would fabricate two
            # numbers indistinguishable from a real zero-order cycle.
            sent = latest.get("orders_sent")
            k3.metric(
                "orders sent / failed",
                f"{sent} / {latest.get('n_failed', '?')}" if sent is not None
                else ("— (dry run, not submitted)"
                      if latest.get("submitted") is False else "—"),
            )
            eq, prev_eq = latest.get("equity"), (prev or {}).get("equity")
            k4.metric(
                "broker equity (paper)",
                f"${eq:,.0f}" if eq else "n/a",
                delta=f"{eq - prev_eq:+,.0f}" if eq and prev_eq else None,
            )
            _src("results/live/summary_*.json")

    m1, m2, m3 = st.columns(3)
    m1.metric("cycles with prediction logs", facts["n_logged"])
    m2.metric("matured cycles (21 trading days)", facts["n_matured"])
    first = facts["first_measurable"]
    m3.metric(
        "first live IC measurable",
        str(first.date()) if first is not None else "n/a",
        delta=(f"{len(pd.bdate_range(today, first)) - 1} trading days to go"
               if first is not None and first > today else None),
        delta_color="off",
    )
    st.caption(
        f"Validity rule, stated up front: no live-vs-backtest claim before "
        f"{facts['min_cycles_for_tstat']}+ matured cycles (Newey–West needs "
        f"them); early ICs are single noisy draws. Maturity dates here use "
        f"weekday counting (holidays not modeled); the IC computation itself "
        f"uses actual trading days."
    )

    if facts["n_matured"] == 0:
        bt_ic = (f"{backtest_stats['mean_rank_ic']:+.4f}"
                 if backtest_stats else "n/a")
        st.info(
            f"Live IC not yet measurable — 0 matured cycles. The backtest "
            f"anchor it will be compared against is mean rank IC {bt_ic} "
            f"(source: {os.path.basename(BACKTEST_METRICS)})."
        )
    st.write("**Live IC vs backtest IC + public-price marks** — fetch prices "
             "to compute (explicit, cached, the page's only network call). "
             "Marks are measurable from the first cycle; IC needs maturity.")
    if st.button("fetch prices and compute live IC + marks"):
        tickers = tuple(sorted(set().union(
            *(set(w.index) for w in weights_by_date.values()),
            *(set(p.index) for p in preds_by_date.values()),
        )))
        first_rec = min(list(weights_by_date) + list(preds_by_date))
        start = (first_rec - pd.Timedelta(days=600)).date().isoformat()
        prices = _fetch_prices(tickers, start)
        bundle = loaders.compute_live_vs_backtest(
            preds_by_date, weights_by_date, prices, backtest_stats, HORIZON
        )
        live_ic, base_ic = bundle["live_ic"], bundle["baseline_ic"]
        if len(live_ic):
            fig, ax = plt.subplots(figsize=(10, 3.5))
            live_ic.plot(ax=ax, marker="o", lw=0.8, label="model live IC")
            if len(base_ic):
                base_ic.plot(ax=ax, marker="x", lw=0.6,
                             label="momentum control arm live IC")
            if backtest_stats:
                ax.axhline(backtest_stats["mean_rank_ic"], ls="--", c="gray",
                           label=f"backtest mean IC "
                                 f"{backtest_stats['mean_rank_ic']:+.4f}")
            ax.axhline(0, c="k", lw=0.5)
            ax.legend(fontsize=8)
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                "The control arm separates 'the model decayed' from 'the "
                "period was hostile to everything'."
            )
            comp = bundle["comparison"]
            if comp is not None:
                n_meas = comp["n_cycles_measurable"]
                st.write(
                    f"live mean IC **{comp['live_mean_ic']:+.4f}** "
                    f"(t_NW {comp['live_ic_tstat_nw']:.2f}) vs backtest "
                    f"**{comp['backtest_mean_ic']:+.4f}** "
                    f"(t_NW {comp['backtest_ic_tstat_nw']:.2f}) over "
                    f"{n_meas} matured cycles"
                )
                if n_meas < facts["min_cycles_for_tstat"]:
                    st.warning(
                        f"do not interpret yet: {n_meas} matured cycles < "
                        f"{facts['min_cycles_for_tstat']} needed for a valid "
                        "Newey–West t-stat — early ICs are single noisy draws"
                    )
        else:
            st.info("No matured cycles — IC chart omitted (marks below).")
        book_pnl = bundle["book_pnl"]
        if len(book_pnl):
            cum = float((1 + book_pnl).prod() - 1)
            st.write(
                f"public-price mark of the logged books: {len(book_pnl)} "
                f"trading days, cumulative **{cum:+.2%}** (gross, no costs) "
                "— a cross-check on the broker's equity curve, never a "
                "performance claim; fills, costs and borrow live at the broker."
            )
        else:
            st.write("No markable days yet (a book logged at t earns from t+1).")
        _src("results/live/predictions_*.csv · weights_*.csv via "
             "quantlab.monitor (realized_live_ic, realized_book_returns, "
             "live_vs_backtest)")

    if not interview and weights_by_date:
        latest_w = weights_by_date[max(weights_by_date)]
        longs = latest_w.nlargest(10)
        shorts = latest_w.nsmallest(10)
        b1, b2, b3 = st.columns([1, 1, 1])
        b1.write("**top longs**")
        b1.dataframe(longs.rename("weight").round(4), height=240)
        b2.write("**top shorts**")
        b2.dataframe(shorts.rename("weight").round(4), height=240)
        with b3:
            st.metric("gross exposure", f"{latest_w.abs().sum():.3f}")
            st.metric("net exposure", f"{latest_w.sum():+.4f}")
            st.caption(
                "Smallest element of this panel on purpose: early paper P&L "
                "is noise. The broker's equity curve is authoritative; "
                "public-price marks are a cross-check only."
            )
        _src("results/live/weights_*.csv")

        with st.expander("What changed last night"):
            dates = sorted(weights_by_date)
            if len(dates) >= 2:
                d = loaders.diff_weights(
                    weights_by_date[dates[-2]], weights_by_date[dates[-1]]
                )
                st.write(f"**entered** ({len(d['entered'])}): "
                         f"{', '.join(d['entered']) or '—'}")
                st.write(f"**exited** ({len(d['exited'])}): "
                         f"{', '.join(d['exited']) or '—'}")
                if d["biggest_changes"]:
                    st.dataframe(pd.DataFrame(d["biggest_changes"]))
            else:
                st.write("Need two logged books to diff — one cycle so far.")
            rev_files = sorted(glob.glob(os.path.join(LIVE_DIR, "revisions_*.json")))
            if rev_files:
                rev = (_read_json(rev_files[-1]) or {}) | {
                    "file": os.path.basename(rev_files[-1])
                }
                if all(k in rev for k in loaders.REVISION_KEYS):
                    st.write(
                        f"latest data-revision fingerprint "
                        f"(vs {rev.get('compared_to')}): "
                        f"{rev['n_price_cells_changed']:,} price cells / "
                        f"{rev['n_return_cells_changed']:,} RETURN cells changed"
                    )
                else:
                    st.warning(
                        f"latest revisions artifact {rev['file']} is "
                        "unreadable/incomplete — no numbers shown rather "
                        "than fabricated zeros"
                    )
            _src("results/live/weights_*.csv · revisions_*.json")

st.divider()

# ---------------------------------------------------------------------------
# 2. Research-integrity ledger
# ---------------------------------------------------------------------------

st.header("Research-integrity ledger")
st.write(
    "Every strategy variant ever evaluated on real data is one row and one "
    "increment of N — the deflated Sharpe ratio is computed against all of "
    "them, not just the survivors."
)

if trials:
    # Sample size for the hurdle comes from an artifact, not a magic number:
    # the deployed config's metrics JSON records its OOS length (n_days).
    n_obs = int(backtest_stats["n_days"]) if backtest_stats else None
    xs = [t["trial_no"] for t in trials]
    dsrs = [t["dsr"] for t in trials]
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()
    if n_obs:
        hurdle = [
            np.sqrt(252) * metrics.expected_max_sharpe(n, 1.0 / n_obs, n_obs)
            for n in xs
        ]
        ax2.step(xs, hurdle, where="post", color="#b71c1c", lw=1.2,
                 label="E[max SR of N noise trials] (ann., right axis)")
        ax2.set_ylim(0, max(hurdle) * 1.6)  # explicit: axes are NOT comparable
    for t, d in zip(trials, dsrs):
        if d is None:
            continue
        ax1.scatter(t["trial_no"], d, s=60,
                    color="#1a237e" if d < 0.95 else "#f57f17", zorder=3)
    ax1.axhline(0.95, ls=":", c="gray", lw=1, label="DSR 0.95 bar")
    if trials and trials[0]["trial_no"] == 1 and dsrs[0] is not None:
        ax1.annotate("trial #1: survivorship bias —\nkilled by trial #2",
                     xy=(1, dsrs[0]), xytext=(1.4, max(0.55, dsrs[0] - 0.25)),
                     fontsize=8, arrowprops={"arrowstyle": "->", "lw": 0.8})
    ax1.set_xlabel("trial number (chronological)")
    ax1.set_ylabel("logged DSR (probability)")
    ax2.set_ylabel("luck hurdle: annualized E[max SR | N]")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xticks(xs)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    ax1.set_title("the cost of every trial: the luck hurdle rises with N; "
                  "all post-bias-fix trials sit at the floor", fontsize=10)
    st.pyplot(fig)
    plt.close(fig)
    if n_obs:
        st.caption(
            f"Hurdle recomputed with quantlab.metrics.expected_max_sharpe "
            f"(same function the pipeline uses), var_sr = 1/{n_obs}, "
            f"n_obs = {n_obs} = n_days from "
            f"{os.path.basename(BACKTEST_METRICS)} (the deployed config; "
            f"earlier trials' samples differ slightly — the hurdle's trend, "
            f"not its level, is the story). Left axis is a probability, "
            f"right axis an annualized Sharpe: different units, deliberately "
            f"not comparable; any vertical alignment is coincidence. DSR "
            f"points are the values logged at each trial's own N — history, "
            f"not recomputed."
        )
    else:
        st.caption(
            f"Luck hurdle omitted: {os.path.basename(BACKTEST_METRICS)} not "
            "found, so there is no artifact-backed sample size to compute it "
            "from. DSR points are the logged values."
        )
    _src("research_log.md trial rows + quantlab.metrics.expected_max_sharpe")

    table = pd.DataFrame(
        [{"#": t["trial_no"], "date": t["date"],
          "hypothesis": (t["hypothesis"][:90] + "…"
                         if len(t["hypothesis"]) > 90 else t["hypothesis"]),
          "net SR": t["net_sr"], "IC": t["ic"], "DSR": t["dsr"],
          "turnover/yr": t["turnover"],
          "conclusion": (t["conclusion"][:110] + "…"
                         if len(t["conclusion"]) > 110 else t["conclusion"])}
         for t in trials]
    ).set_index("#")
    st.dataframe(table, use_container_width=True)
    st.caption("Nulls are results too — nothing here is colored green.")
    with st.expander("full hypothesis / conclusion text per trial"):
        for t in trials:
            st.markdown(f"**#{t['trial_no']} ({t['date']})** — {t['hypothesis']}")
            st.markdown(f"*conclusion:* {t['conclusion']}")
    unparsed = [r for r in log_rows if not r.get("parsed")]
    if unparsed:
        with st.expander(f"{len(unparsed)} log rows could not be parsed "
                         "(shown raw, never dropped)"):
            for r in unparsed:
                st.code(r["raw"])
else:
    st.warning("No trial rows parsed from research_log.md — the raw file is "
               "the source of truth; check its table format.")
_src("research_log.md")

hypos = loaders.parse_hypotheses(hypo_md)
if hypos:
    st.write("**Pre-registered queue** — declared before they run, never edited after:")
    st.dataframe(
        pd.DataFrame(hypos).set_index("name"), use_container_width=True, height=200
    )
else:
    st.info("Pre-registered queue: no registrations parsed from "
            "writeup/preregistered_hypotheses.md (file missing or empty) — "
            "future trials must be registered there before they run.")
_src("writeup/preregistered_hypotheses.md")

st.divider()

# ---------------------------------------------------------------------------
# 3. Data-revision monitor
# ---------------------------------------------------------------------------

st.header("Data revisions — the vendor rewrites the past")
rev_files = sorted(glob.glob(os.path.join(LIVE_DIR, "revisions_*.json")))
if rev_files:
    records = [(_read_json(p) or {}) | {"file": os.path.basename(p)}
               for p in rev_files]
    revs, skipped = loaders.split_revision_records(records)
    if skipped:
        st.warning(
            f"{len(skipped)} revisions artifact(s) unreadable/incomplete and "
            f"EXCLUDED (never rendered as zeros): {', '.join(skipped)}"
        )
    if revs:
        df = pd.DataFrame(revs)
        fig, (axa, axb) = plt.subplots(1, 2, figsize=(10, 3))
        axa.plot(range(len(df)), df["frac_price_cells_changed"] * 100, marker="o")
        axa.set_title("price cells changed (%)", fontsize=9)
        axb.plot(range(len(df)), df["n_return_cells_changed"], marker="o",
                 color="#b71c1c")
        axb.set_title("RETURN cells changed (count)", fontsize=9)
        for ax in (axa, axb):
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels([f["file"][10:20] for f in revs], fontsize=7,
                               rotation=45)
        st.pyplot(fig)
        plt.close(fig)
        last = revs[-1]
        if last.get("top_affected_tickers"):
            st.dataframe(pd.DataFrame(last["top_affected_tickers"]))
    else:
        st.info("Revision files exist but none is readable — see the "
                "exclusion list above; the raw files are the record.")
else:
    st.info(
        "Collection started 2026-06-11; the first comparison lands once two "
        "consecutive daily snapshots exist. Nothing is shown because nothing "
        "has been measured — that is the honest state."
    )
st.caption(
    "Price-cell changes are mostly benign whole-history re-adjustments; "
    "RETURN-cell changes silently alter features and labels — point-in-time "
    "has a data-values dimension, not just a universe dimension."
)
_src("results/live/revisions_*.json")

st.divider()

# ---------------------------------------------------------------------------
# 4. Execution reality
# ---------------------------------------------------------------------------

st.header("Execution reality — costs and capacity")
cap = _read_json(os.path.join(RESULTS, "capacity_sp500_ridge_both_residlabel.json"))
e1, e2 = st.columns([3, 2])
with e1:
    curve = pd.DataFrame(cap["curve"]) if cap and cap.get("curve") else pd.DataFrame()
    aum_col = next((c for c in curve.columns if "aum" in c.lower()), None)
    # Both columns must be recognized, or we say so — silently plotting a
    # non-AUM column on the x-axis would be a fake chart, not a fallback.
    if aum_col is not None and "ann_cost_drag" in curve.columns:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.semilogx(curve[aum_col], curve["ann_cost_drag"] * 100, marker="o")
        ax.set_xlabel("AUM ($)")
        ax.set_ylabel("annual cost drag (%)")
        ax.set_title(f"square-root impact sweep "
                     f"(ADV coverage {cap.get('adv_coverage', float('nan')):.1%})",
                     fontsize=9)
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Any future strategy with this turnover profile needs ≥1.4%/yr "
            "of true gross alpha at $1M, ≥5%/yr at $100M, just to exist."
        )
    elif cap:
        st.info("Capacity artifact present but its schema is unrecognized "
                "(need 'aum'-named and 'ann_cost_drag' columns) — refusing "
                "to guess which column is which.")
    else:
        st.info("Capacity artifact not found — run the --capacity sweep to "
                "produce it.")
    _src("results/capacity_sp500_ridge_both_residlabel.json")
with e2:
    if trials:
        tdf = pd.DataFrame(
            [{"trial": t["trial_no"], "turnover/yr": t["turnover"]} for t in trials]
        ).set_index("trial")
        st.dataframe(tdf, use_container_width=True)
        st.caption("Turnover is a headline metric here: it is what kills "
                   "weak edges at real costs.")
        _src("research_log.md trial rows")

# ---------------------------------------------------------------------------
# 5. Artifact browser + footer
# ---------------------------------------------------------------------------

if not interview:
    with st.expander("Artifact browser (results/, read-only)"):
        artifacts = sorted(
            glob.glob(os.path.join(RESULTS, "*.json"))
            + glob.glob(os.path.join(RESULTS, "*.png"))
        )
        choice = st.selectbox(
            "artifact", [os.path.basename(p) for p in artifacts] or ["(none)"]
        )
        path = os.path.join(RESULTS, choice)
        if choice.endswith(".json") and os.path.exists(path):
            st.json(_read_json(path) or {})
        elif choice.endswith(".png") and os.path.exists(path):
            st.image(path)

st.divider()
st.subheader("Known limitations (verbatim from the README — naming them is the point)")
limitations = loaders.extract_md_section(readme_md, "Known limitations")
if limitations:
    st.markdown(limitations)
else:
    st.error("README 'Known limitations' section not found — that section "
             "is part of the project's contract; restore it.")
_src("README.md · 'Known limitations'")

try:
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, timeout=5, check=False,
    ).stdout.strip() or "unknown"
except OSError:
    commit = "unknown"
st.caption(
    f"reading repo state at commit `{commit}` · every number on this page is "
    "regenerable from a config + seed; see research_log.md · this dashboard "
    "is read-only and cannot feed back into the strategy"
)
