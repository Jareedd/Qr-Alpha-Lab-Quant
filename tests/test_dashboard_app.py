"""Smoke-run the dashboard headlessly via streamlit.testing.AppTest.

Asserts the page renders with TODAY'S real artifacts (few cycles, zero
matured ICs, possibly zero revision files) without raising — the spec's
'honest empty states' requirement, executed rather than promised. Skipped
wholesale when streamlit is not installed (it is a dashboard-only dep;
the trading path and CI without requirements-dashboard.txt stay lean).
"""

import json
import os
import sys

import pandas as pd
import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

HERE = os.path.dirname(__file__)
APP = os.path.join(HERE, "..", "dashboard", "app.py")
sys.path.insert(0, os.path.join(HERE, "..", "dashboard"))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import loaders  # noqa: E402


def test_app_renders_offline_with_current_artifacts():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, f"dashboard raised: {at.exception}"
    assert at.title[0].value.startswith("qr-alpha-lab")
    # Not just 'no exception': the gate banner must be GREEN (the committed
    # artifacts pass both bounds) -- a page of red banners renders without
    # exceptions too, and that regression is the one worth catching.
    assert len(at.success) >= 1, "no green banner: gate red or artifacts unread"
    # And N must come from the real log, not the '?' fallback.
    with open(os.path.join(HERE, "..", "research_log.md"), encoding="utf-8") as f:
        n = loaders.parse_trial_count(f.read())
    assert str(at.metric[0].value) == str(n)
    bodies = " ".join(str(getattr(el, "value", "")) for el in at.markdown)
    assert "Known limitations" in bodies or any(
        "Known limitations" in str(getattr(e, "value", "")) for e in at.subheader
    )


def test_interview_mode_renders_too():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    # Flip the sidebar toggle and re-run: still no exception, page intact.
    at.sidebar.toggle[0].set_value(True).run()
    assert not at.exception, f"interview mode raised: {at.exception}"


def test_data_present_branches_render_before_they_ever_run_in_production(
    tmp_path, monkeypatch
):
    """Mature cycles + revision files (one valid, one corrupt) via the
    QRLAB_LIVE_DIR override: the calendar-activated branches (maturity
    metrics with matured cycles, the revisions chart, the corrupt-file
    exclusion path) must render TODAY in a test, not for the first time in
    production the day the real data matures."""
    from quantlab.revisions import compare_price_snapshots

    tickers = [f"T{i:02d}" for i in range(40)]
    asof = "2026-01-05"  # ~5 months ago -> long matured
    pd.Series([0.05] * 10 + [-0.05] * 10 + [0.0] * 20,
              index=pd.Index(tickers, name="ticker"), name="weight"
              ).to_csv(tmp_path / f"weights_{asof}.csv")
    pd.DataFrame({"pred_raw": range(40), "pred_sector_neutral": range(40),
                  "baseline_mom_12_1": range(40)},
                 index=pd.Index(tickers, name="ticker")
                 ).to_csv(tmp_path / f"predictions_{asof}.csv")
    (tmp_path / f"summary_{asof}.json").write_text(
        json.dumps({"asof": asof, "n_names": 20, "submitted": False})
    )
    # A REAL producer fingerprint (pins the producer/consumer key contract)
    # plus a corrupt one (truncated write) that must be excluded loudly.
    dates = pd.bdate_range("2025-12-01", periods=20)
    old = pd.DataFrame(100.0, index=dates, columns=tickers[:5])
    new = old * 1.01
    (tmp_path / "revisions_2026-01-06.json").write_text(
        json.dumps({"compared_to": asof} | compare_price_snapshots(old, new))
    )
    (tmp_path / "revisions_2026-01-07.json").write_text("{}")

    monkeypatch.setenv("QRLAB_LIVE_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, f"data-present branches raised: {at.exception}"
    # The corrupt file is excluded with a loud warning, never zeros.
    warnings = " ".join(str(getattr(w, "value", "")) for w in at.warning)
    assert "revisions_2026-01-07.json" in warnings
    # Matured cycles register (the fetch button renders; it is NOT clicked,
    # so the test stays offline) and the dry-run summary shows no fabricated
    # order counts.
    metrics_text = " ".join(str(m.value) for m in at.metric)
    assert "dry run" in metrics_text
