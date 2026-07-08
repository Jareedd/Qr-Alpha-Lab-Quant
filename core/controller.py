"""PortfolioController — the single authority between alpha and execution.

Every TradeIntent passes through here; nothing reaches the execution layer
without an ExecutionOrder minted by this class. Authorization order:

    1. kill switch   (book-level daily drawdown; sticky once tripped)
    2. agent check   (only intents from registry-gated, registered agents)
    3. regime filter (hardcoded volatility / correlation caps)

PIT note (law #1): the controller never fetches data. The caller supplies a
MarketState computed FROM PAST DATA ONLY (realized vol, realized pairwise
correlation, today's realized drawdown) — the controller has no channel
through which future information could enter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.agent import Side, StrategyAgent, TradeIntent, Urgency
from core.registry_gate import require_deployable


@dataclass(frozen=True, slots=True)
class MarketState:
    """Caller-computed, past-only snapshot the controller judges against."""

    realized_vol_ann: float    # trailing realized book vol, annualized
    avg_pairwise_corr: float   # trailing avg pairwise correlation of holdings
    daily_drawdown: float      # today's realized peak-to-now return, <= 0


@dataclass(frozen=True, slots=True)
class RegimeLimits:
    """Hardcoded caps. Frozen: changing a cap is a config change that
    belongs in git history, not in runtime mutation."""

    max_realized_vol_ann: float = 0.30   # refuse new risk above 30% ann vol
    max_avg_pairwise_corr: float = 0.80  # refuse when everything correlates
    max_daily_drawdown: float = -0.03    # kill switch trips at -3% on the day


@dataclass(frozen=True, slots=True)
class ExecutionOrder:
    """An AUTHORIZED instruction. Only the controller mints these; the
    execution layer must accept nothing else."""

    symbol: str
    side: Side
    shares: int
    urgency: Urgency
    agent_id: str


@dataclass(frozen=True, slots=True)
class RejectedIntent:
    """A refusal with its reason — refusals are results too (house style)."""

    intent: TradeIntent
    reason: str


@dataclass
class PortfolioController:
    """Ingests TradeIntents, applies kill switch -> agent check -> regime
    filter, emits ExecutionOrders. One instance owns one book."""

    limits: RegimeLimits = field(default_factory=RegimeLimits)
    registry_path: str | None = None  # None = the real registry (tests inject)

    def __post_init__(self) -> None:
        self._agents: dict[str, StrategyAgent] = {}
        self._killed: bool = False
        self._kill_reason: str = ""

    # -- agent registration (deployment authority) ------------------------

    def register_agent(self, agent: StrategyAgent) -> None:
        """Admit an agent iff its hypothesis is GRADUATED in the registry."""
        require_deployable(agent.hypothesis_id, self.registry_path)
        if agent.agent_id in self._agents:
            raise ValueError(f"agent_id {agent.agent_id!r} already registered")
        self._agents[agent.agent_id] = agent

    # -- kill switch -------------------------------------------------------

    @property
    def execution_authority(self) -> bool:
        return not self._killed

    @property
    def kill_reason(self) -> str:
        return self._kill_reason

    def check_kill_switch(self, daily_drawdown: float) -> bool:
        """Trip (sticky) when today's drawdown breaches the cap; return the
        tripped state. A calmer later reading does NOT untrip — recovery
        within the same day is luck, not authority."""
        if daily_drawdown <= self.limits.max_daily_drawdown:
            self._killed = True
            self._kill_reason = (
                f"KILL SWITCH: daily drawdown {daily_drawdown:.4f} breaches "
                f"cap {self.limits.max_daily_drawdown:.4f}; execution "
                "authority revoked for ALL agents"
            )
        return self._killed

    def reinstate_authority(self) -> None:
        """Explicitly restore authority. A human act after investigation —
        nothing in the framework calls this."""
        self._killed = False
        self._kill_reason = ""

    # -- authorization pipeline ---------------------------------------------

    def _regime_rejection(self, state: MarketState) -> str | None:
        if state.realized_vol_ann > self.limits.max_realized_vol_ann:
            return (
                f"REGIME: realized vol {state.realized_vol_ann:.3f} > cap "
                f"{self.limits.max_realized_vol_ann:.3f}"
            )
        if state.avg_pairwise_corr > self.limits.max_avg_pairwise_corr:
            return (
                f"REGIME: avg pairwise corr {state.avg_pairwise_corr:.3f} > "
                f"cap {self.limits.max_avg_pairwise_corr:.3f}"
            )
        return None

    def process_intents(
        self, intents: list[TradeIntent], state: MarketState
    ) -> tuple[list[ExecutionOrder], list[RejectedIntent]]:
        """Authorize or reject every intent. Total: len(orders) +
        len(rejections) == len(intents), so nothing disappears silently."""
        self.check_kill_switch(state.daily_drawdown)
        if self._killed:
            return [], [RejectedIntent(i, self._kill_reason) for i in intents]

        regime_reason = self._regime_rejection(state)
        orders: list[ExecutionOrder] = []
        rejections: list[RejectedIntent] = []
        for intent in intents:
            if intent.agent_id not in self._agents:
                rejections.append(
                    RejectedIntent(
                        intent,
                        f"UNREGISTERED AGENT {intent.agent_id!r}: no "
                        "deployment authority (registry gate)",
                    )
                )
            elif regime_reason is not None:
                rejections.append(RejectedIntent(intent, regime_reason))
            else:
                orders.append(
                    ExecutionOrder(
                        symbol=intent.symbol,
                        side=intent.side,
                        shares=intent.requested_shares,
                        urgency=intent.urgency,
                        agent_id=intent.agent_id,
                    )
                )
        return orders, rejections
