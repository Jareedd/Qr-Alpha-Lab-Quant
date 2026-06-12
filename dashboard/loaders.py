"""Pure parsing functions for the dashboard (no Streamlit, no IO side effects).

Everything here takes text/Series in and returns plain data out, so it is
testable without a browser. Parsing philosophy, stated once: the research
log and README are human-written markdown — parsers must degrade honestly.
A row that cannot be parsed is returned with ``parsed=False`` and its raw
text intact so the UI can SHOW it rather than silently drop it (a dropped
trial row would understate N's history, the one thing this project must
never do).
"""

from __future__ import annotations

import re

import pandas as pd

# Markdown noise we strip before extracting numbers: bold markers, unicode
# minus/multiplication signs, non-breaking spaces.
_CLEAN = str.maketrans({"−": "-", "×": "x", " ": " "})


def _clean(text: str) -> str:
    return text.replace("**", "").translate(_CLEAN)


def parse_trial_count(log_md: str) -> int | None:
    """The bold 'Global trial count ... N = 7' line at the top of the log.

    Returns None if the line is missing (the UI shows a loud warning then —
    a research log without a trial count is itself a finding).
    """
    m = re.search(r"Global trial count[^\n]*?N\s*=\s*(\d+)", log_md)
    return int(m.group(1)) if m else None


def parse_research_log(log_md: str) -> list[dict]:
    """All rows from the log's markdown tables (there are several blocks).

    A row is a trial if its first cell is an integer; '—' rows are infra.
    Returns dicts with keys: kind ('trial'|'infra'), trial_no (int|None),
    date, hypothesis, config, result, conclusion, parsed (bool), raw (str),
    and best-effort floats net_sr, ic, dsr, turnover extracted from the
    result cell (None where absent — absence is shown, not faked).
    """
    rows: list[dict] = []
    for line in log_md.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        # maxsplit=6: the 7th cell (conclusion) may contain literal '|'
        # characters (e.g. 'p95 |β| 0.32' in trial #3) — an unbounded split
        # would silently truncate it mid-sentence while claiming parsed=True.
        cells = [c.strip() for c in line.strip().strip("|").split("|", 6)]
        if cells[0] in ("#", "---") or set(cells[0]) <= {"-"}:
            continue  # header or separator
        if len(cells) < 7:
            # Malformed table row: shown raw by the UI, never dropped.
            rows.append({"kind": "unknown", "parsed": False, "raw": line})
            continue
        first = _clean(cells[0]).strip()
        if first == "—" or first == "-":
            kind, trial_no = "infra", None
        elif first.isdigit():
            kind, trial_no = "trial", int(first)
        else:
            rows.append({"kind": "unknown", "parsed": False, "raw": line})
            continue
        result = _clean(cells[5])
        rows.append(
            {
                "kind": kind,
                "trial_no": trial_no,
                "date": cells[1],
                "hypothesis": _clean(cells[3]),
                "config": _clean(cells[4]),
                "result": result,
                "conclusion": _clean(cells[6]),
                "parsed": True,
                "raw": line,
                # Patterns shaped by the real log's prose (each variant is a
                # row that silently rendered blank before being pinned):
                # 'net SR −0.01' but also bare 'net −0.77' (trial #5);
                # 'IC 0.0052' but also 'IC vs residual label +0.0225';
                # 'turnover 7.26×/yr' but also 'turnover 3.46×' (no /yr).
                "net_sr": _first_float(result, rf"\bnet(?:\s+SR)?\s*({_NUM})"),
                "ic": _first_float(result, rf"\bIC\b[^,;.(]*?({_NUM})"),
                "dsr": _parse_dsr(result),
                "turnover": _first_float(result, rf"turnover\s*({_NUM})"),
            }
        )
    return rows


# A float that cannot swallow a sentence-ending period ('DSR 0.01. Vol…'
# must capture '0.01') and accepts an explicit '+' ('IC +0.0225') — both
# real-log formats that silently parsed to None before being pinned.
_NUM = r"[+-]?\d+(?:\.\d+)?"


def _first_float(text: str, pattern: str) -> float | None:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    try:
        return float(m.group(1)) if m else None
    except ValueError:
        return None


def _parse_dsr(result: str) -> float | None:
    """DSR appears as 'DSR 0.29', 'DSR 0.998 @ N=1' or 'DSR ≈ 0'."""
    m = re.search(rf"DSR\s*≈?\s*({_NUM})", result)
    try:
        return float(m.group(1)) if m else None
    except ValueError:
        return None


def extract_md_section(md: str, heading: str) -> str | None:
    """Body of the markdown section whose heading contains ``heading``
    (case-insensitive), up to the next heading of same-or-higher level.
    Returns None when absent — the caller must say so, not improvise."""
    pattern = rf"^(#{{1,6}})\s*[^\n]*{re.escape(heading)}[^\n]*$"
    m = re.search(pattern, md, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return None
    level = len(m.group(1))
    rest = md[m.end():]
    nxt = re.search(rf"^#{{1,{level}}}\s", rest, flags=re.MULTILINE)
    return (rest[: nxt.start()] if nxt else rest).strip()


def parse_hypotheses(md: str) -> list[dict]:
    """(name, title, status) for each '### H<n>: ...' registration block.

    Status is the first '- Status:' line in the block, first word kept
    (PROPOSED / RUN / ABANDONED); 'UNKNOWN' if the line is missing.
    """
    out: list[dict] = []
    blocks = re.split(r"^###\s+", md, flags=re.MULTILINE)[1:]
    for block in blocks:
        head, _, body = block.partition("\n")
        m = re.match(r"(H\d+)\s*:\s*(.+)", head.strip())
        if not m:
            continue
        sm = re.search(r"-\s*Status:\s*([A-Z]+)", body)
        out.append(
            {
                "name": m.group(1),
                "title": m.group(2).strip(),
                "status": sm.group(1) if sm else "UNKNOWN",
            }
        )
    return out


def diff_weights(prev: pd.Series, cur: pd.Series, top: int = 8) -> dict:
    """What changed between two logged books (ticker -> weight Series).

    Returns entered/exited ticker lists and the ``top`` largest absolute
    weight changes among names present in both. Pure description of two
    artifacts — no judgement about whether the changes were good.
    """
    prev_names, cur_names = set(prev.index), set(cur.index)
    common = sorted(prev_names & cur_names)
    delta = (cur[common] - prev[common]).sort_values(key=abs, ascending=False)
    return {
        "entered": sorted(cur_names - prev_names),
        "exited": sorted(prev_names - cur_names),
        "biggest_changes": [
            {"ticker": t, "from": float(prev[t]), "to": float(cur[t])}
            for t in delta.index[:top]
        ],
    }


def gate_verdict(planted: dict | None, noise: dict | None) -> dict:
    """Falsification-gate verdict from the two metrics dicts.

    Bounds mirror the CI gate flags: planted DSR must exceed 0.95; noise
    DSR must stay below 0.5. Missing artifact => not ok (loud, not quiet).
    """
    p_ok = planted is not None and planted.get("dsr", 0.0) > 0.95
    n_ok = noise is not None and noise.get("dsr", 1.0) < 0.5
    return {
        "ok": p_ok and n_ok,
        "planted_ok": p_ok,
        "noise_ok": n_ok,
        "planted_dsr": planted.get("dsr") if planted else None,
        "noise_dsr": noise.get("dsr") if noise else None,
    }


def cycle_staleness(latest_cycle: pd.Timestamp, today: pd.Timestamp) -> int:
    """Whole trading days elapsed since the latest logged cycle (0 = fresh).

    Weekends are ignored via bdate_range; NYSE holidays are NOT modeled
    (project-wide convention), so a holiday Monday shows as 1 day stale —
    the caption must say so rather than pretend precision.
    """
    days = pd.bdate_range(latest_cycle.normalize(), today.normalize())
    return max(0, len(days) - 1)


def maturity_facts(
    pred_dates: list[pd.Timestamp], today: pd.Timestamp, horizon: int = 21
) -> dict:
    """How far along the live-IC record is, in honest units.

    A cycle matures ``horizon`` TRADING days after its as-of date (the
    backtest's label convention). 'first_measurable' is the maturity date
    of the OLDEST prediction-bearing cycle. ``min_cycles_for_tstat`` is
    horizon + 2 = the smallest n for which metrics.newey_west_tstat
    returns a number (it NaNs below lags + 2) — the same bound the spec
    quotes as '23+'.
    """
    if not pred_dates:
        return {"n_logged": 0, "n_matured": 0, "first_measurable": None,
                "min_cycles_for_tstat": horizon + 2}
    first = min(pred_dates)
    first_measurable = pd.bdate_range(start=first, periods=horizon + 1)[-1]
    n_matured = sum(
        pd.bdate_range(start=d, periods=horizon + 1)[-1] <= today
        for d in pred_dates
    )
    return {
        "n_logged": len(pred_dates),
        "n_matured": int(n_matured),
        "first_measurable": first_measurable,
        "min_cycles_for_tstat": horizon + 2,
    }


# Keys the revisions panel renders; a fingerprint file missing any of them
# (truncated write, schema drift) is listed as unreadable, never silently
# rendered as zeros and never allowed to crash the page.
REVISION_KEYS = (
    "frac_price_cells_changed",
    "n_price_cells_changed",
    "n_return_cells_changed",
    "n_cells_compared",
)


def split_revision_records(records: list[dict]) -> tuple[list[dict], list[str]]:
    """(valid, skipped_names): records carrying every REVISION_KEYS key vs
    the names of files that don't. Each record needs a 'file' key for the
    skip list."""
    valid = [r for r in records if all(k in r for k in REVISION_KEYS)]
    skipped = [r.get("file", "<unnamed>") for r in records
               if not all(k in r for k in REVISION_KEYS)]
    return valid, skipped


def compute_live_vs_backtest(
    preds_by_date: dict,
    weights_by_date: dict,
    prices: pd.DataFrame,
    backtest_stats: dict | None,
    horizon: int = 21,
) -> dict:
    """Everything the fetch button displays, via quantlab.monitor (the spec's
    contract: reuse, never reimplement). Pure computation on artifacts +
    prices; no IO, so the producer/consumer key contract is pinned by a unit
    test with synthetic prices instead of first executing in production.

    Returns dict with: live_ic, baseline_ic (Series, possibly empty),
    book_pnl (Series — public-price MARKS of the logged books, measurable
    before any IC matures), comparison (dict | None — monitor.live_vs_backtest
    incl. NW t-stats, None when backtest_stats is unavailable).
    """
    from quantlab import monitor

    live_ic = monitor.realized_live_ic(preds_by_date, prices, horizon=horizon)
    with_base = {d: p for d, p in preds_by_date.items()
                 if "baseline_mom_12_1" in p.columns}
    baseline_ic = (
        monitor.realized_live_ic(with_base, prices, horizon=horizon,
                                 col="baseline_mom_12_1")
        if with_base else pd.Series(dtype=float)
    )
    book_pnl = (monitor.realized_book_returns(weights_by_date, prices)
                if weights_by_date else pd.Series(dtype=float))
    comparison = (monitor.live_vs_backtest(live_ic, backtest_stats, horizon=horizon)
                  if backtest_stats else None)
    return {"live_ic": live_ic, "baseline_ic": baseline_ic,
            "book_pnl": book_pnl, "comparison": comparison}
