"""H13 — Post-Earnings-Announcement Drift agent (scaffolding, data-gated).

Wraps the registered-but-not-yet-graduated H13 hypothesis in the core
StrategyAgent contract so the logic is ready the instant the real Bloomberg
PIT estimate pull arrives. It reuses the ACTUAL core contracts
(TradeIntent / RiskExposure / Side / Urgency) rather than re-declaring them —
the framework boundary is respected, not duplicated.

NOT DEPLOYABLE YET, by design: H13 is PROPOSED (data-gated), so
PortfolioController.register_agent() will refuse this agent via the registry
gate. That refusal is correct — do not work around it. This module only
defines the strategy; wiring it live waits on trial #14 graduating.

Point-in-time safety (law #1): at each ``asof`` the agent sees ONLY rows with
as_of_date <= asof AND announcement_date <= asof. The drift is traded AFTER
the surprise is public, so both timestamps must be in the past; the surprise
is measured against the PRE-announcement consensus carried in the row, which
is exactly the field free data lacks and Bloomberg PIT provides.
"""

from __future__ import annotations

import os
import sys
from typing import Mapping

import numpy as np
import pandas as pd

# Repo root on the path so the top-level `core` package imports — the same
# convention core/registry_gate.py and the test suite use.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.agent import RiskExposure, Side, StrategyAgent, TradeIntent, Urgency

_REQUIRED_COLUMNS = (
    "ticker",
    "as_of_date",
    "announcement_date",
    "consensus_eps_est",
    "actual_eps",
    "pre_earnings_volatility",
)


class PEADAgent(StrategyAgent):
    """Long positive-surprise names, short negative-surprise names; size by
    inverse pre-earnings volatility.

    A name is eligible only if |SUE| exceeds ``sue_threshold`` (1.5 by
    default), where SUE = (actual_eps - consensus_eps_est) /
    pre_earnings_volatility. Weights are inverse-vol (lower vol -> higher
    conviction), normalized to ``target_gross``.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        agent_id: str = "pead-1",
        hypothesis_id: str = "H13",
        *,
        book_equity: float = 1_000_000.0,
        target_gross: float = 1.0,
        reference_price: float = 100.0,
        sue_threshold: float = 1.5,
    ) -> None:
        super().__init__(agent_id, hypothesis_id)
        missing = [c for c in _REQUIRED_COLUMNS if c not in data.columns]
        if missing:
            raise ValueError(f"PEAD data missing required columns: {missing}")
        if sue_threshold <= 0 or book_equity <= 0 or reference_price <= 0:
            raise ValueError("threshold, book_equity, reference_price must be > 0")

        self._data = data.copy()
        self._data["as_of_date"] = pd.to_datetime(self._data["as_of_date"])
        self._data["announcement_date"] = pd.to_datetime(self._data["announcement_date"])

        self._book_equity = float(book_equity)
        self._target_gross = float(target_gross)
        self._reference_price = float(reference_price)
        self._sue_threshold = float(sue_threshold)

        # Cache of the last rebalance so calculate_risk_exposure() reflects
        # the "currently desired book" (the contract's wording).
        self._last_asof: pd.Timestamp | None = None
        self._last_signals: dict[str, float] = {}
        self._last_book: dict[str, float] | None = None

    @classmethod
    def from_csv(cls, path: str, **kwargs) -> "PEADAgent":
        return cls(pd.read_csv(path), **kwargs)

    # -- point-in-time slice ------------------------------------------------

    def _visible(self, asof: pd.Timestamp) -> pd.DataFrame:
        """Rows knowable at ``asof`` (both timestamps in the past), latest
        announcement per ticker. This is the ONLY place data enters the
        signal, so PIT safety is enforced in one auditable spot."""
        df = self._data
        mask = (df["as_of_date"] <= asof) & (df["announcement_date"] <= asof)
        visible = df.loc[mask].sort_values("announcement_date")
        return visible.groupby("ticker", as_index=False).tail(1)

    # -- single source of truth for a rebalance ----------------------------

    def _rebalance(self, asof) -> tuple[dict[str, float], dict[str, float]]:
        asof_ts = pd.Timestamp(asof)
        v = self._visible(asof_ts)

        sue = (v["actual_eps"] - v["consensus_eps_est"]) / v["pre_earnings_volatility"]
        eligible = v.assign(sue=sue.values)
        eligible = eligible[eligible["sue"].abs() > self._sue_threshold]

        signals = {t: float(s) for t, s in zip(eligible["ticker"], eligible["sue"])}

        if len(eligible):
            # Inverse-vol conviction: lower pre-earnings vol -> larger |weight|.
            inv_vol = 1.0 / eligible["pre_earnings_volatility"].to_numpy()
            raw = np.sign(eligible["sue"].to_numpy()) * inv_vol
            gross = np.abs(raw).sum()
            weights = raw / gross * self._target_gross
            book = {t: float(w) for t, w in zip(eligible["ticker"], weights)}
        else:
            book = {}

        self._last_asof = asof_ts
        self._last_signals = signals
        self._last_book = book
        return signals, book

    # -- StrategyAgent contract ---------------------------------------------

    def get_signal(self, asof) -> Mapping[str, float]:
        """SUE score per eligible ticker at ``asof`` (positive = bullish,
        negative = bearish). Names inside the +/-threshold band are omitted."""
        signals, _ = self._rebalance(asof)
        return signals

    def calculate_risk_exposure(self) -> RiskExposure:
        """Exposure of the currently desired book. Inverse-vol sizing shows
        up here as concentration in the lowest-vol, highest-conviction names.
        Call get_signal() or generate_intent() first to set the book."""
        if self._last_book is None:
            raise RuntimeError(
                "no book computed yet — call get_signal(asof) or "
                "generate_intent(asof) before calculate_risk_exposure()"
            )
        if not self._last_book:
            return RiskExposure(0.0, 0.0, 0.0)
        w = np.array(list(self._last_book.values()))
        return RiskExposure(
            gross_exposure=float(np.abs(w).sum()),
            net_exposure=float(w.sum()),
            largest_position_weight=float(np.abs(w).max()),
        )

    def generate_intent(self, asof) -> tuple[TradeIntent, ...]:
        """Immutable TradeIntents at ``asof``. Target weights are converted to
        integer share counts at a reference price; a name that rounds to 0
        shares is dropped rather than emitted as an invalid intent. PEAD is a
        patient multi-week drift, so urgency is LOW."""
        _, book = self._rebalance(asof)
        intents: list[TradeIntent] = []
        for ticker, weight in book.items():
            shares = int(round(abs(weight) * self._book_equity / self._reference_price))
            if shares <= 0:
                continue
            intents.append(
                TradeIntent(
                    symbol=ticker,
                    side=Side.BUY if weight > 0 else Side.SELL,
                    requested_shares=shares,
                    urgency=Urgency.LOW,
                    agent_id=self.agent_id,
                )
            )
        return tuple(intents)
