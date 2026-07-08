"""ExecutionManager — simulation-only sink for authorized orders.

Ingests ExecutionOrder objects (minted only by core.controller) and emits a
structured JSON line per order to simulate market interaction. Explicitly
NOT a broker client: no network, no keys, no live path. Injectable clock
and sink keep tests deterministic.

Duck-typed on purpose: this module does not import core, so the execution
layer cannot grow a back-channel dependency on portfolio logic. Anything
with symbol / side / shares / urgency / agent_id fields is loggable.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable


class ExecutionManager:
    """Dummy execution handler: order in, JSON record out."""

    def __init__(
        self,
        sink: Callable[[str], Any] | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self._sink = sink if sink is not None else print
        self._now = now_fn or (lambda: datetime.now(timezone.utc))

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
