"""Strategy-agent contract for the multi-agent architecture.

Assumptions, stated so the owner can defend them in an interview:
- An agent WRAPS a graduated, pre-registered hypothesis; it never originates
  one. Deployment authority is checked by core.registry_gate at registration
  time, not here — this module only defines the contract.
- The point-in-time law (law #1) applies at the contract level: get_signal()
  and generate_intent() take an explicit ``asof`` and may use data with
  timestamp <= asof ONLY. Every concrete agent must state its PIT argument
  in its own docstring.
- TradeIntent is immutable: what a strategy asked for must be un-editable
  downstream, or the controller's authorization trail proves nothing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class Urgency(str, Enum):
    LOW = "low"        # patient — TWAP/VWAP-eligible once execution grows up
    NORMAL = "normal"  # default working order
    HIGH = "high"      # cross the spread; reserved for risk-off exits


@dataclass(frozen=True, slots=True)
class TradeIntent:
    """What a strategy WANTS — not what it is allowed to do.

    ``agent_id`` carries provenance beyond the four spec fields: the
    controller needs to know WHO asked, both for the audit trail and to
    reject intents from agents it never registered.
    """

    symbol: str
    side: Side
    requested_shares: int
    urgency: Urgency
    agent_id: str

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("TradeIntent.symbol must be non-empty")
        if not self.agent_id:
            raise ValueError("TradeIntent.agent_id must be non-empty")
        if self.requested_shares <= 0:
            raise ValueError(
                "TradeIntent.requested_shares must be > 0 — direction lives "
                "in `side`, not in the sign of the share count"
            )


@dataclass(frozen=True, slots=True)
class RiskExposure:
    """Standardized risk snapshot every agent must be able to report.

    Fractions of book equity: gross = sum(|w|), net = sum(w) (0.0 means
    dollar-neutral), largest = max(|w|) concentration.
    """

    gross_exposure: float
    net_exposure: float
    largest_position_weight: float


class StrategyAgent(ABC):
    """Contract every alpha module must satisfy to talk to the controller.

    Subclasses MUST implement get_signal, calculate_risk_exposure and
    generate_intent — Python refuses to instantiate a partial implementation.
    """

    def __init__(self, agent_id: str, hypothesis_id: str) -> None:
        if not agent_id or not hypothesis_id:
            raise ValueError("agent_id and hypothesis_id are required")
        self.agent_id = agent_id
        # The registry entry this agent trades. Deployment is refused unless
        # that entry's status is GRADUATED (core.registry_gate).
        self.hypothesis_id = hypothesis_id

    @abstractmethod
    def get_signal(self, asof) -> Mapping[str, float]:
        """Cross-sectional signal at ``asof`` (symbol -> score), using data
        with timestamp <= asof ONLY."""

    @abstractmethod
    def calculate_risk_exposure(self) -> RiskExposure:
        """Risk snapshot of the agent's currently desired book."""

    @abstractmethod
    def generate_intent(self, asof) -> tuple[TradeIntent, ...]:
        """Immutable TradeIntents at ``asof``.

        A cross-sectional book emits many intents per rebalance, so the
        return is a tuple of frozen TradeIntent — the whole request is
        immutable, per the spec's intent."""
