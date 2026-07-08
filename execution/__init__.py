"""Execution layer: order routing, slippage models, VWAP/TWAP — eventually.

Today it contains exactly one thing: a SIMULATION-ONLY manager that logs
authorized orders as structured JSON. There is deliberately no live order
path in this package; src/quantlab/live.py remains the only code that talks
to a broker (paper-only), and it stays that way until a strategy graduates.
"""

from execution.manager import ExecutionManager

__all__ = ["ExecutionManager"]
