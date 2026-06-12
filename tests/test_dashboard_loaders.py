"""Dashboard parsing functions (dashboard/loaders.py) — pure, no Streamlit.

The research-log parser is the one that matters: if it silently drops or
misreads a trial row, the dashboard understates the project's trial
history, which is the one number this project must never get wrong.
Tested against BOTH a synthetic fixture (known answers) and the real
research_log.md (must parse every actual trial).
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dashboard"))

import loaders  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")

FIXTURE = """# Research Log — test

**Global trial count (feeds `--n-trials` for the DSR): N = 7**

| # | Date | Type | Hypothesis / change | Config | OOS result | Conclusion |
|---|------|------|---------------------|--------|------------|------------|
| — | 2026-06-10 | infra | Vectorize hot paths | defaults | planted PASS | behavior-preserving |
| 1 | 2026-06-10 | **trial** | Default config on real data | yfinance | IC 0.0333 (t=7.77*), net SR 0.82, DSR 0.998 @ N=1, turnover 3.81×/yr | survivorship bias suspected |

| 2 | 2026-06-10 | **trial** | PIT universe | sp500 | IC 0.0052 (t_NW **0.54**), **net SR −0.01**, DSR 0.29, turnover 7.26×/yr | alpha was survivorship bias |
| 3 | 2026-06-10 | **trial** | Neutralization | sp500 | IC unchanged 0.0052, net SR −0.38, DSR 0.01. Vol fell 10.0%→6.5% | machinery validated: p95 |β| 0.32→0.03 with signal intact |
| 5 | 2026-06-10 | **trial** | Residual labels | sp500 | IC vs residual label **+0.0225**, gross SR **−0.61**, net −0.77, DSR ≈ 0, turnover 3.46× | IC and P&L differ |
| broken row with too few cells |
"""


def test_parse_trial_count():
    assert loaders.parse_trial_count(FIXTURE) == 7
    assert loaders.parse_trial_count("no header here") is None


def test_parse_research_log_known_answers():
    rows = loaders.parse_research_log(FIXTURE)
    trials = [r for r in rows if r["kind"] == "trial"]
    infra = [r for r in rows if r["kind"] == "infra"]
    assert len(trials) == 4 and len(infra) == 1  # both table blocks read

    t1 = next(t for t in trials if t["trial_no"] == 1)
    assert t1["net_sr"] == pytest.approx(0.82)
    assert t1["ic"] == pytest.approx(0.0333)
    assert t1["dsr"] == pytest.approx(0.998)
    assert t1["turnover"] == pytest.approx(3.81)

    # Unicode minus and bold markers must survive: net SR −0.01 -> -0.01.
    t2 = next(t for t in trials if t["trial_no"] == 2)
    assert t2["net_sr"] == pytest.approx(-0.01)
    assert t2["dsr"] == pytest.approx(0.29)

    # 'DSR 0.01.' (sentence-ending period) must parse as 0.01, and a literal
    # '|' INSIDE a cell ('p95 |β| 0.32') must not truncate the conclusion --
    # both real-log formats that silently corrupted before being pinned.
    t3 = next(t for t in trials if t["trial_no"] == 3)
    assert t3["dsr"] == pytest.approx(0.01)
    assert t3["conclusion"].endswith("0.32→0.03 with signal intact")

    # Trial #5's real formats, all of which once parsed to None while the
    # values sat in the log: '+'-signed IC with words between ('IC vs
    # residual label +0.0225'), bare 'net −0.77', 'turnover 3.46×' (no /yr).
    t5 = next(t for t in trials if t["trial_no"] == 5)
    assert t5["ic"] == pytest.approx(0.0225)
    assert t5["net_sr"] == pytest.approx(-0.77)
    assert t5["turnover"] == pytest.approx(3.46)
    assert t5["dsr"] == pytest.approx(0.0)  # 'DSR ≈ 0'

    # Malformed rows are returned raw and flagged, never silently dropped.
    broken = [r for r in rows if not r["parsed"]]
    assert len(broken) == 1 and "too few cells" in broken[0]["raw"]
    assert all(r["parsed"] for r in rows if r["kind"] != "unknown")


def test_parser_against_the_real_research_log():
    with open(os.path.join(ROOT, "research_log.md"), encoding="utf-8") as f:
        md = f.read()
    n = loaders.parse_trial_count(md)
    rows = loaders.parse_research_log(md)
    trials = [r for r in rows if r["kind"] == "trial"]
    assert n is not None and n >= 7
    # Law #3 invariant: the trial row and the N bump are one edit, so the
    # max trial number must equal the declared N. If this fails, either a
    # trial was logged without bumping N (understated N inflates every DSR)
    # or vice versa -- fix the LOG, not this test.
    assert max(t["trial_no"] for t in trials) == n, (
        "research_log.md trial numbering disagrees with the declared N "
        "(CLAUDE.md law #3) -- reconcile the log before anything else"
    )
    # Known history (trials 1-7): every headline metric is in those rows'
    # prose and must parse -- blank dashboard cells for logged values are
    # silent corruption. Scoped to <= 7 on purpose: a FUTURE trial with a
    # new prose format should extend the parser + this test together, not
    # redden the suite for wording reasons.
    for t in (t for t in trials if t["trial_no"] <= 7):
        for field in ("dsr", "ic", "net_sr", "turnover"):
            assert t[field] is not None, f"trial {t['trial_no']}: {field} unparsed"
    # Trial #3's conclusion contains literal '|' chars; the sentinel value
    # sits PAST them and must survive (truncation regression).
    t3 = next(t for t in trials if t["trial_no"] == 3)
    assert "0.32" in t3["conclusion"]
    assert len([r for r in rows if r["kind"] == "infra"]) >= 5


def test_extract_md_section():
    md = "# T\n\n## Known limitations (deliberate honesty)\n\nbody line\n\n## References\n\nrefs"
    out = loaders.extract_md_section(md, "Known limitations")
    assert out == "body line"
    assert loaders.extract_md_section(md, "Nonexistent") is None


def test_parse_hypotheses():
    md = (
        "### H1: quality tilts\n- Status: PROPOSED — blocked\n\n"
        "### H2: crypto carry\n- Status: PROPOSED\n\n"
        "### H3: regimes\nno status line here\n"
    )
    out = loaders.parse_hypotheses(md)
    assert [h["name"] for h in out] == ["H1", "H2", "H3"]
    assert out[0]["status"] == "PROPOSED"
    assert out[2]["status"] == "UNKNOWN"


def test_diff_weights_known_answer():
    prev = pd.Series({"AAA": 0.02, "BBB": -0.02, "CCC": 0.01})
    cur = pd.Series({"AAA": 0.025, "CCC": -0.03, "DDD": 0.02})
    d = loaders.diff_weights(prev, cur, top=1)
    assert d["entered"] == ["DDD"] and d["exited"] == ["BBB"]
    assert d["biggest_changes"] == [{"ticker": "CCC", "from": 0.01, "to": -0.03}]


def test_gate_verdict_bounds_and_missing_files():
    ok = loaders.gate_verdict({"dsr": 0.99}, {"dsr": 0.08})
    assert ok["ok"] and ok["planted_ok"] and ok["noise_ok"]
    assert not loaders.gate_verdict(None, {"dsr": 0.08})["ok"]          # missing = loud
    assert not loaders.gate_verdict({"dsr": 0.90}, {"dsr": 0.08})["ok"]  # planted fail
    assert not loaders.gate_verdict({"dsr": 0.99}, {"dsr": 0.60})["ok"]  # noise fail


def test_cycle_staleness_ignores_weekends():
    fri = pd.Timestamp("2026-06-05")  # Friday
    mon = pd.Timestamp("2026-06-08")  # Monday
    assert loaders.cycle_staleness(fri, fri) == 0
    assert loaders.cycle_staleness(fri, mon) == 1  # weekend skipped
    assert loaders.cycle_staleness(fri, pd.Timestamp("2026-06-10")) == 3


def test_split_revision_records_excludes_incomplete_files():
    good = {k: 1 for k in loaders.REVISION_KEYS} | {"file": "revisions_a.json"}
    bad = {"file": "revisions_b.json"}  # truncated write / schema drift
    valid, skipped = loaders.split_revision_records([good, bad])
    assert valid == [good]
    assert skipped == ["revisions_b.json"]
    assert loaders.split_revision_records([]) == ([], [])


def test_compute_live_vs_backtest_pins_the_monitor_contract():
    # The fetch button's whole computation, run on synthetic prices: pins
    # the producer/consumer key contract (live IC + control-arm column +
    # book marks + NW comparison) so the data-present branch does not
    # execute for the first time in production when cycles mature.
    import numpy as np

    rng = np.random.default_rng(3)
    # 420 days: the label residualizes against rolling 252d betas, so the
    # as-of date needs a full beta warm-up of history behind it (the same
    # reason live_report fetches 600 calendar days back).
    dates = pd.bdate_range("2024-06-03", periods=420)
    tickers = [f"T{i:02d}" for i in range(40)]
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (420, 40)), axis=0)),
        index=dates, columns=tickers,
    )
    asof = dates[-60]  # > 21 trading days of future prices -> matured
    preds = pd.DataFrame(
        {"pred_raw": rng.normal(size=40),
         "pred_sector_neutral": rng.normal(size=40),
         "baseline_mom_12_1": rng.normal(size=40)},
        index=pd.Index(tickers, name="ticker"),
    )
    weights = pd.Series(
        [0.05] * 10 + [-0.05] * 10 + [0.0] * 20, index=tickers, name="weight"
    )
    bt_stats = {"mean_rank_ic": 0.0225, "ic_tstat_newey_west": 1.91}

    out = loaders.compute_live_vs_backtest(
        {asof: preds}, {asof: weights}, prices, bt_stats, horizon=21
    )
    assert len(out["live_ic"]) == 1 and np.isfinite(out["live_ic"].iloc[0])
    assert len(out["baseline_ic"]) == 1
    assert len(out["book_pnl"]) > 0  # marks exist without any maturity
    comp = out["comparison"]
    assert comp["n_cycles_measurable"] == 1
    assert comp["backtest_mean_ic"] == 0.0225
    # 1 matured cycle << 23: t_NW must be NaN, never a fake number.
    assert np.isnan(comp["live_ic_tstat_nw"])

    # Without the backtest artifact the comparison is None, not a guess.
    assert loaders.compute_live_vs_backtest(
        {asof: preds}, {}, prices, None, horizon=21
    )["comparison"] is None


def test_maturity_facts_counts_trading_days():
    d0 = pd.Timestamp("2026-06-11")
    facts = loaders.maturity_facts([d0], today=pd.Timestamp("2026-06-12"), horizon=21)
    assert facts["n_logged"] == 1 and facts["n_matured"] == 0
    assert facts["first_measurable"] == pd.bdate_range(d0, periods=22)[-1]
    # horizon + 2 = the smallest n where metrics.newey_west_tstat returns a
    # number (it NaNs below lags + 2) -- the spec's '23+'.
    assert facts["min_cycles_for_tstat"] == 23
    # at maturity day it counts
    facts2 = loaders.maturity_facts(
        [d0], today=facts["first_measurable"], horizon=21
    )
    assert facts2["n_matured"] == 1
    assert loaders.maturity_facts([], pd.Timestamp("2026-06-12"))["n_logged"] == 0
