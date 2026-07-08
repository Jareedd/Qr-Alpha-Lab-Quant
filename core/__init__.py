"""Multi-agent portfolio architecture: contracts, authority, enforcement.

Layering (one direction only):
    strategies/  -> core.agent      (implement the StrategyAgent contract)
    core.controller                 (authorizes TradeIntent -> ExecutionOrder)
    execution/                      (consumes ExecutionOrder; simulation-only today)

The research laws are enforced in code here, not by convention:
- Deployment authority comes from the registry (core.registry_gate): only a
  GRADUATED hypothesis may be wrapped in an agent. As of 2026-07-08 nothing
  has graduated (N=13, zero graduations), so the framework authorizes zero
  live strategies — that is the honest state of the research.
- The legacy pipeline (src/quantlab/, scripts/) is untouched; live.py remains
  the only production paper-trading path.
"""

from core.agent import RiskExposure, Side, StrategyAgent, TradeIntent, Urgency
from core.controller import (
    ExecutionOrder,
    MarketState,
    PortfolioController,
    RegimeLimits,
    RejectedIntent,
)
from core.registry_gate import require_deployable

__all__ = [
    "ExecutionOrder",
    "MarketState",
    "PortfolioController",
    "RegimeLimits",
    "RejectedIntent",
    "RiskExposure",
    "Side",
    "StrategyAgent",
    "TradeIntent",
    "Urgency",
    "require_deployable",
]
