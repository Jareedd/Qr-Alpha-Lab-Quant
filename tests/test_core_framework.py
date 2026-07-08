"""Known-answer tests for the multi-agent framework (core/, execution/).

Pins the contracts the architecture rests on: intent immutability, ABC
enforcement, registry-gated deployment (including the honest fact that the
REAL registry authorizes zero deployments today), regime refusals, the
sticky kill switch, and the simulation-only JSON execution record.
"""

import dataclasses
import json
import os
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.agent import RiskExposure, Side, StrategyAgent, TradeIntent, Urgency
from core.controller import (
    ExecutionOrder,
    MarketState,
    PortfolioController,
    RegimeLimits,
)
from core.registry_gate import require_deployable
from execution.manager import ExecutionManager

CALM = MarketState(realized_vol_ann=0.15, avg_pairwise_corr=0.30, daily_drawdown=-0.005)


def _intent(agent_id="a1", symbol="AAPL", shares=100):
    return TradeIntent(
        symbol=symbol,
        side=Side.BUY,
        requested_shares=shares,
        urgency=Urgency.NORMAL,
        agent_id=agent_id,
    )


class _StubAgent(StrategyAgent):
    def get_signal(self, asof):
        return {"AAPL": 1.0}

    def calculate_risk_exposure(self):
        return RiskExposure(gross_exposure=1.0, net_exposure=0.0, largest_position_weight=0.05)

    def generate_intent(self, asof):
        return (_intent(agent_id=self.agent_id),)


@pytest.fixture
def graduated_registry(tmp_path):
    """A synthetic registry (fixture only, clearly labeled) with one
    GRADUATED entry, so deployment paths are testable before any real
    hypothesis graduates."""
    p = tmp_path / "registry.md"
    p.write_text(
        "### H99: synthetic fixture hypothesis (test only, never real)\n"
        "- Status: GRADUATED (fixture)\n",
        encoding="utf-8",
    )
    return str(p)


# -- TradeIntent contract ----------------------------------------------------


def test_trade_intent_is_immutable():
    intent = _intent()
    with pytest.raises(dataclasses.FrozenInstanceError):
        intent.requested_shares = 999


def test_trade_intent_validation():
    with pytest.raises(ValueError):
        _intent(shares=0)
    with pytest.raises(ValueError):
        _intent(shares=-100)
    with pytest.raises(ValueError):
        _intent(symbol="")


def test_abc_refuses_partial_implementation():
    class Partial(StrategyAgent):  # missing generate_intent
        def get_signal(self, asof):
            return {}

        def calculate_risk_exposure(self):
            return RiskExposure(0.0, 0.0, 0.0)

    with pytest.raises(TypeError):
        Partial("p1", "H99")


# -- registry gate (deployment authority) ------------------------------------


def test_real_registry_authorizes_zero_deployments_today():
    """The honest pinned fact: N=13, zero graduations -> no hypothesis is
    deployable. If a hypothesis ever graduates, this test fails on purpose
    and must be updated consciously alongside the registry."""
    for n in range(1, 14):
        with pytest.raises(RuntimeError):
            require_deployable(f"H{n}")


def test_registry_gate_authorizes_graduated_and_refuses_unknown(graduated_registry):
    require_deployable("H99", graduated_registry)  # does not raise
    with pytest.raises(RuntimeError, match="not registered"):
        require_deployable("H42", graduated_registry)


def test_register_agent_enforces_registry(graduated_registry):
    refused = PortfolioController()  # real registry: nothing deployable
    with pytest.raises(RuntimeError):
        refused.register_agent(_StubAgent("a1", "H10"))

    admitted = PortfolioController(registry_path=graduated_registry)
    admitted.register_agent(_StubAgent("a1", "H99"))
    with pytest.raises(ValueError, match="already registered"):
        admitted.register_agent(_StubAgent("a1", "H99"))


# -- controller: regime filter and authorization ------------------------------


def _controller(graduated_registry):
    c = PortfolioController(registry_path=graduated_registry)
    c.register_agent(_StubAgent("a1", "H99"))
    return c


def test_calm_regime_authorizes_and_mirrors_intent(graduated_registry):
    c = _controller(graduated_registry)
    orders, rejections = c.process_intents([_intent()], CALM)
    assert rejections == []
    assert orders == [
        ExecutionOrder(symbol="AAPL", side=Side.BUY, shares=100, urgency=Urgency.NORMAL, agent_id="a1")
    ]


def test_vol_breach_rejects_all(graduated_registry):
    c = _controller(graduated_registry)
    hot = MarketState(realized_vol_ann=0.45, avg_pairwise_corr=0.30, daily_drawdown=-0.005)
    orders, rejections = c.process_intents([_intent()], hot)
    assert orders == []
    assert rejections[0].reason.startswith("REGIME: realized vol")


def test_corr_breach_rejects_all(graduated_registry):
    c = _controller(graduated_registry)
    crisis = MarketState(realized_vol_ann=0.15, avg_pairwise_corr=0.95, daily_drawdown=-0.005)
    orders, rejections = c.process_intents([_intent()], crisis)
    assert orders == []
    assert rejections[0].reason.startswith("REGIME: avg pairwise corr")


def test_unregistered_agent_is_rejected_even_in_calm_regime(graduated_registry):
    c = _controller(graduated_registry)
    orders, rejections = c.process_intents([_intent(agent_id="rogue")], CALM)
    assert orders == []
    assert "UNREGISTERED AGENT" in rejections[0].reason


def test_nothing_disappears_silently(graduated_registry):
    c = _controller(graduated_registry)
    intents = [_intent(), _intent(agent_id="rogue"), _intent(symbol="MSFT")]
    orders, rejections = c.process_intents(intents, CALM)
    assert len(orders) + len(rejections) == len(intents)


# -- kill switch ---------------------------------------------------------------


def test_kill_switch_boundary(graduated_registry):
    c = _controller(graduated_registry)
    assert c.check_kill_switch(-0.029) is False  # above cap: no trip
    assert c.execution_authority
    assert c.check_kill_switch(-0.03) is True  # exactly at cap: trips
    assert not c.execution_authority


def test_kill_switch_is_sticky_until_reinstated(graduated_registry):
    c = _controller(graduated_registry)
    crash = MarketState(realized_vol_ann=0.15, avg_pairwise_corr=0.30, daily_drawdown=-0.05)
    orders, rejections = c.process_intents([_intent()], crash)
    assert orders == [] and "KILL SWITCH" in rejections[0].reason

    # a calm later state does NOT restore authority by itself
    orders, rejections = c.process_intents([_intent()], CALM)
    assert orders == [] and "KILL SWITCH" in rejections[0].reason

    c.reinstate_authority()  # explicit human act
    orders, rejections = c.process_intents([_intent()], CALM)
    assert len(orders) == 1 and rejections == []


# -- execution layer -------------------------------------------------------------


def test_execution_manager_emits_structured_json(graduated_registry):
    c = _controller(graduated_registry)
    orders, _ = c.process_intents([_intent()], CALM)

    lines = []
    frozen_now = datetime(2026, 7, 8, 12, 0, 0, tzinfo=timezone.utc)
    mgr = ExecutionManager(sink=lines.append, now_fn=lambda: frozen_now)
    record = mgr.execute(orders[0])

    assert json.loads(lines[0]) == record  # the log line IS the record
    assert record == {
        "record_type": "simulated_execution",
        "paper_only": True,
        "ts_utc": "2026-07-08T12:00:00+00:00",
        "symbol": "AAPL",
        "side": "buy",
        "shares": 100,
        "urgency": "normal",
        "agent_id": "a1",
    }
