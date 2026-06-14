"""H8 event-study harness: falsification gate + known-answer logic.

Before the real run may touch the deletion data, the harness must (a)
RECOVER a planted post-event drift on pseudo-events and (b) REJECT when no
drift is planted -- the same planted/noise discipline as the rest of the
project, applied to event time.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import events, metrics
from quantlab.synthetic import inject_post_event_drift, make_panel


def _pseudo_events(panel, n=40, seed=3):
    """Random (date, ticker) pseudo-events in the middle of the panel."""
    rng = np.random.default_rng(seed)
    dates = panel.index[300:-80]
    cols = list(panel.columns)
    return [(dates[rng.integers(0, len(dates))], cols[rng.integers(0, len(cols))])
            for _ in range(n)]


def _run(panel, ev_list):
    dv = panel * 1.0  # uniform "dollar volume" proxy = price (size ~ log price)
    events_df = pd.DataFrame(
        [{"effective_date": d, "ticker": t} for d, t in ev_list]
    )
    return events.event_study(events_df, panel, dv, horizon=60, n_match=10)


def test_event_harness_recovers_planted_drift_and_rejects_none():
    panel = make_panel(n_assets=60, n_days=1500, mode="noise", seed=5)
    ev = _pseudo_events(panel, n=50, seed=3)

    planted = inject_post_event_drift(panel, ev, drift=0.15, horizon=60)
    res_p = _run(planted, ev)
    res_0 = _run(panel, ev)  # same events, NO drift -> the null world

    assert res_p["n_events"] >= 30, "test setup: enough usable events"
    # RECOVER: planted drift shows up as positive mean event excess and a
    # positive daily-portfolio Sharpe...
    assert res_p["event_total_excess"].mean() > 0.05
    assert metrics.sharpe(res_p["daily_portfolio"]) > 0.5
    # ...REJECT (one-sided): with no planted drift, the harness must NOT
    # manufacture a positive rebound. The null's mean excess sits at ~0
    # gross, and net of the per-event cost it is mildly NEGATIVE (a
    # zero-alpha strategy loses the spread) -- which is correct, so the
    # null test is "no positive rebound", not "Sharpe near zero".
    assert res_0["event_total_excess"].mean() < 0.02
    assert metrics.sharpe(res_0["daily_portfolio"]) < 0.3
    # and the planted world must clearly exceed the null (paired control).
    assert (res_p["event_total_excess"].mean()
            - res_0["event_total_excess"].mean()) > 0.05


def test_discretionary_deletions_filters_by_reason():
    changes = pd.DataFrame({
        "date": pd.to_datetime(["2015-03-01", "2018-06-01", "2020-09-01",
                                "2009-01-01"]),
        "added": [None, None, None, None],
        "removed": ["AAA", "BBB", "CCC", "OLD"],
        "reason": ["Market capitalization decline.", "Acquired by MegaCorp.",
                   "Moved to S&P MidCap 400", "Market cap change."],
    })
    out = events.discretionary_deletions(changes, start="2010-01-01")
    # only AAA qualifies: BBB is M&A, CCC is migration, OLD is pre-2010
    assert list(out["ticker"]) == ["AAA"]


def test_matched_controls_picks_nearest_by_size_and_momentum():
    feat = pd.DataFrame({
        "size": [10.0, 10.1, 5.0, 9.9, 2.0],
        "mom": [0.1, 0.11, -0.5, 0.09, 0.8],
    }, index=["TGT", "NEAR1", "FAR1", "NEAR2", "FAR2"])
    ctrls = events.matched_controls("TGT", feat, n=2)
    assert set(ctrls) == {"NEAR1", "NEAR2"}
    assert "TGT" not in ctrls
