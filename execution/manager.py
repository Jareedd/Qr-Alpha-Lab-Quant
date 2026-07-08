"""ExecutionManager — simulation-only order handler and scheduler.

Ingests authorized orders (minted by core.controller) and emits structured
JSON lines to simulate market interaction. Explicitly NOT a broker client:
no network, no keys, no live path — src/quantlab/live.py remains the only
code that talks to Alpaca (paper-only).

Two entry points:
- execute(order): the atomic "emit one simulated fill" primitive (one order,
  one JSON record). Unchanged, backward-compatible.
- route(order): the urgency-aware scheduler. LOW -> TWAP (13 slices across a
  390-minute session); HIGH -> a single aggressive market order; NORMAL ->
  a single working order. Returns the child slice records it emits.

Constitutional compliance:
- Law #2 (dependency inversion): this module imports NOTHING from core or
  strategies. Orders are consumed by STRUCTURAL DUCK-TYPING — any object with
  symbol / side / shares / urgency / agent_id works, whether it carries enum
  or plain-string fields. There is no back-channel to portfolio logic.
- Law #3 (telemetry invariant): a routed order never loses shares —
  sum(slice.shares) == order.shares, asserted before records are returned.
- Law #5 (provenance): every emitted record is stamped paper_only=True and
  record_type="simulated_execution", so a reader can never mistake a
  simulation for a live fill.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

#: TWAP schedule for a patient (LOW-urgency) order: 13 child slices spread
#: evenly across a 390-minute (6.5-hour) US cash session -> one slice every
#: 30 minutes at offsets 0, 30, ..., 360.
TWAP_INTERVALS = 13
SESSION_MINUTES = 390


class ExecutionManager:
    """Simulation-only execution handler + urgency-aware scheduler."""

    def __init__(
        self,
        sink: Callable[[str], Any] | None = None,
        now_fn: Callable[[], datetime] | None = None,
        *,
        session_open: datetime | None = None,
        twap_intervals: int = TWAP_INTERVALS,
        session_minutes: int = SESSION_MINUTES,
    ) -> None:
        self._sink = sink if sink is not None else print
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self._session_open = session_open
        if twap_intervals < 1:
            raise ValueError("twap_intervals must be >= 1")
        self._twap_intervals = int(twap_intervals)
        self._session_minutes = int(session_minutes)

    # -- atomic primitive (unchanged) --------------------------------------

    def execute(self, order) -> dict:
        """Log ``order`` as one structured JSON line; return the record.

        The record is self-describing: record_type and paper_only are in
        the payload so a log reader can never mistake simulation for a
        live fill.
        """
        record = {
            "record_type": "simulated_execution",
            "paper_only": True,
            "ts_utc": self._now().isoformat(),
            "symbol": order.symbol,
            "side": order.side.value,
            "shares": order.shares,
            "urgency": order.urgency.value,
            "agent_id": order.agent_id,
        }
        self._sink(json.dumps(record, sort_keys=True))
        return record

    # -- duck-typed field access (no core import; Law #2) ------------------

    @staticmethod
    def _as_str(value) -> str:
        """Enum-or-string -> lowercase string. Lets route() accept both a
        core Side/Urgency enum and a plain duck-typed string without ever
        importing the enum types."""
        return (value.value if hasattr(value, "value") else str(value)).lower()

    # -- TWAP schedule -----------------------------------------------------

    def _twap_plan(self, shares: int) -> list[tuple[int, int]]:
        """(offset_minutes, slice_shares) per child, summing to ``shares``.

        Shares split as evenly as possible with the remainder loaded onto the
        earliest slices. If shares < intervals, we emit ``shares`` one-lot
        slices rather than fabricate zero-share children — the sum still
        equals the parent (Law #3)."""
        n = self._twap_intervals
        step = self._session_minutes / n
        if shares <= 0:
            return []
        if shares < n:
            return [(int(round(i * step)), 1) for i in range(shares)]
        base, rem = divmod(shares, n)
        return [(int(round(i * step)), base + (1 if i < rem else 0)) for i in range(n)]

    # -- scheduler ---------------------------------------------------------

    def route(self, order) -> list[dict]:
        """Schedule + emit child slices per the order's urgency.

        LOW -> TWAP (``twap_intervals`` slices across the session), NORMAL ->
        one working order, HIGH -> one aggressive market order. Returns the
        emitted slice records; guarantees the slices conserve the parent's
        share count."""
        shares = int(order.shares)
        if shares <= 0:
            return []

        urgency = self._as_str(getattr(order, "urgency", "normal"))
        if urgency == "low":
            plan, style, order_type = self._twap_plan(shares), "twap", "limit"
        elif urgency == "high":
            plan, style, order_type = [(0, shares)], "aggressive_market", "market"
        else:  # normal / anything unrecognized -> a single working order
            plan, style, order_type = [(0, shares)], "working", "limit"

        anchor = self._session_open or self._now()
        n = len(plan)
        records: list[dict] = []
        for i, (offset_min, slice_shares) in enumerate(plan):
            record = {
                "record_type": "simulated_execution",
                "paper_only": True,
                "ts_utc": self._now().isoformat(),
                "symbol": order.symbol,
                "side": self._as_str(order.side),
                "shares": slice_shares,
                "urgency": urgency,
                "agent_id": order.agent_id,
                "execution_style": style,
                "order_type": order_type,
                "slice_index": i,
                "n_slices": n,
                "scheduled_offset_min": offset_min,
                "scheduled_ts": (anchor + timedelta(minutes=offset_min)).isoformat(),
                "parent_shares": shares,
            }
            self._sink(json.dumps(record, sort_keys=True))
            records.append(record)

        routed = sum(r["shares"] for r in records)
        assert routed == shares, (
            f"telemetry breach: routed {routed} shares != parent {shares} "
            "(Law #3 — no shares may disappear silently)"
        )
        return records
