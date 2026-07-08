"""Known-answer tests for ExecutionManager.route() — TWAP + urgency routing.

Deliberately uses a DUCK-TYPED order (types.SimpleNamespace with plain-string
side/urgency, no core enums) so the tests themselves prove Law #2: the
execution layer needs nothing from core. Pins TWAP slicing, share
conservation (Law #3), aggressive routing, and provenance stamping (Law #5).
"""

import json
import os
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from execution.manager import SESSION_MINUTES, TWAP_INTERVALS, ExecutionManager

FROZEN = datetime(2026, 7, 8, 13, 30, 0, tzinfo=timezone.utc)


def _order(urgency="low", shares=1000, symbol="AAPL", side="buy", agent_id="pead-1"):
    # SimpleNamespace, NOT a core dataclass: proves route() is pure duck-typing.
    return SimpleNamespace(
        symbol=symbol, side=side, shares=shares, urgency=urgency, agent_id=agent_id
    )


def _manager(lines):
    return ExecutionManager(sink=lines.append, now_fn=lambda: FROZEN, session_open=FROZEN)


# -- TWAP (LOW urgency) ------------------------------------------------------


def test_low_urgency_produces_13_twap_slices():
    lines = []
    recs = _manager(lines).route(_order(urgency="low", shares=1300))
    assert len(recs) == TWAP_INTERVALS == 13
    assert all(r["execution_style"] == "twap" for r in recs)
    assert all(r["order_type"] == "limit" for r in recs)
    # evenly divisible -> 100 each
    assert [r["shares"] for r in recs] == [100] * 13
    # one line emitted per slice
    assert len(lines) == 13


def test_twap_schedule_is_every_30_minutes_across_the_session():
    recs = _manager([]).route(_order(urgency="low", shares=1300))
    offsets = [r["scheduled_offset_min"] for r in recs]
    assert offsets == [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
    assert offsets[-1] <= SESSION_MINUTES
    # scheduled_ts advances from the session open by the offset
    assert recs[0]["scheduled_ts"] == FROZEN.isoformat()
    assert recs[1]["scheduled_ts"] == "2026-07-08T14:00:00+00:00"


def test_twap_conserves_shares_when_uneven():
    recs = _manager([]).route(_order(urgency="low", shares=100))
    assert sum(r["shares"] for r in recs) == 100  # Law #3
    assert len(recs) == 13
    # 100 = 9*8 + 4*7, remainder loaded onto the earliest slices
    assert [r["shares"] for r in recs] == [8] * 9 + [7] * 4


def test_twap_fewer_shares_than_intervals_emits_one_lot_slices():
    recs = _manager([]).route(_order(urgency="low", shares=5))
    assert [r["shares"] for r in recs] == [1, 1, 1, 1, 1]
    assert sum(r["shares"] for r in recs) == 5  # no fabricated zero-share children


def test_all_slices_carry_parent_share_count():
    recs = _manager([]).route(_order(urgency="low", shares=100))
    assert all(r["parent_shares"] == 100 for r in recs)
    assert all(r["n_slices"] == 13 for r in recs)
    assert [r["slice_index"] for r in recs] == list(range(13))


# -- HIGH / NORMAL -----------------------------------------------------------


def test_high_urgency_is_single_aggressive_market_order():
    recs = _manager([]).route(_order(urgency="high", shares=1000))
    assert len(recs) == 1
    assert recs[0]["execution_style"] == "aggressive_market"
    assert recs[0]["order_type"] == "market"
    assert recs[0]["shares"] == 1000


def test_normal_urgency_is_single_working_order():
    recs = _manager([]).route(_order(urgency="normal", shares=1000))
    assert len(recs) == 1
    assert recs[0]["execution_style"] == "working"
    assert recs[0]["shares"] == 1000


def test_unknown_urgency_defaults_to_single_working_order():
    recs = _manager([]).route(_order(urgency="frantic", shares=42))
    assert len(recs) == 1 and recs[0]["execution_style"] == "working"
    assert recs[0]["shares"] == 42


def test_zero_or_negative_shares_route_to_nothing():
    assert _manager([]).route(_order(shares=0)) == []


# -- provenance (Law #5) & telemetry (Law #3) --------------------------------


def test_every_emitted_slice_is_stamped_paper_only():
    lines = []
    _manager(lines).route(_order(urgency="low", shares=250))
    for line in lines:
        rec = json.loads(line)
        assert rec["paper_only"] is True
        assert rec["record_type"] == "simulated_execution"


def test_emitted_lines_match_returned_records():
    lines = []
    recs = _manager(lines).route(_order(urgency="low", shares=77))
    assert [json.loads(line) for line in lines] == recs


# -- dependency inversion (Law #2) -------------------------------------------


def test_execution_module_does_not_import_core_or_strategies():
    import ast

    path = os.path.join(os.path.dirname(__file__), "..", "execution", "manager.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    # actual import statements only — docstring prose about "core" is fine
    for mod in imported:
        assert not mod.startswith("core"), f"execution imports {mod}"
        assert not mod.startswith("strategies"), f"execution imports {mod}"


def test_route_works_on_pure_duck_typed_order_without_core():
    # The order here is a SimpleNamespace with string fields — if route()
    # secretly depended on core enums this would raise. It must not.
    recs = _manager([]).route(_order(urgency="high", shares=10))
    assert recs[0]["side"] == "buy" and recs[0]["urgency"] == "high"
