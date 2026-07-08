"""Alpha agent modules — one per GRADUATED hypothesis.

Empty on purpose. A strategy lands here only when its hypothesis clears the
pre-registered graduation criteria (registry status GRADUATED), at which
point it implements core.agent.StrategyAgent and registers with the
PortfolioController. As of 2026-07-08: N=13 trials, zero graduations, so
there is nothing honest to put here yet.

First candidate: H13 PEAD (data-gated on the Bloomberg PIT analyst-estimate
pull — see writeup/bloomberg_pead_pull.md). If trial #14 graduates it, the
port target is strategies/pead.py wrapping the registered config.
"""
