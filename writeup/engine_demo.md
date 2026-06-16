# The execution/risk engine — the honest "scale it" machinery

**Status: synthetic demonstration. N unchanged (= 11). No market data, no orders submitted.**

This project has run 11 honest trials and graduated **zero** strategies. So the
question "once you *have* an edge, how do you size and run it?" has to be answered
on ground-truth synthetic data — there is no real edge to drive the machinery.
`scripts/engine_demo.py` does exactly that, and `tests/test_engine_demo.py` pins
the result so it cannot silently rot.

## What the engine is

`quantlab.engine.PortfolioEngine` composes the separately-tested pieces into the
pipeline a multi-manager book actually runs:

```
combine signals  →  neutralize factor exposure  →  SIZE (vol-target × confidence)
                 →  LIMITS (position / gross / drawdown)  →  integer-share ORDERS
```

Each stage is a small, source-agnostic, past-only transform with its own
known-answer tests (`combine.py`, `risk_model.py`, `sizing.py`, `limits.py`,
`execution.py`). Order *submission* stays in the frozen live Alpaca path — the
engine only ever produces a plan.

## The property it proves

Leverage is driven by the **lower confidence bound** of the trailing Sharpe
(Lo-2002 standard error), not the point estimate. So:

| Fed... | Avg gross exposure | Net Sharpe (synthetic) |
|---|---|---|
| a real planted edge | **1.70** (commits) | 7.1 (by construction — large planted premium) |
| an identically-built **null** | **0.000** (sizes to zero) | 0.0 |

The null result is the point. An engine that maximizes geometric growth on a
*confident* edge but **refuses to lever an edge it isn't statistically sure of**
is the direct, in-code counterweight to the trial-#11 over-confidence trap: the
machine declining to bet on a signal whose lower-bound Sharpe is not clearly
positive. The same demo also shows:

- **Slow to commit:** gross exposure ramps from ~1.1 (early, still gathering
  evidence) to the 2.0 gross cap (late, confident) — it does not lever on day one.
- **Limits bind:** per-name |w| ≤ 0.10, gross ≤ 2.0, both respected.
- **Neutralization works:** net market-beta exposure 0.068 → 2e-16 after the
  projection `w − L(LᵀL)⁻¹Lᵀw`.
- **Real orders:** 180 integer-share orders at $1M equity, gross within the cap,
  ready for the live client.
- **The combiner down-weights dead signals:** the quality signal's trailing IC
  (0.30) dwarfs a pure-noise signal's (−0.05) — exactly the regime this project
  lives in, where most candidate signals have ~no IC.

## How a graduated strategy plugs in

The day any strategy clears its pre-registered gate (H1 via CRSP is the leading
candidate), it enters this engine as one more entry in the `signals` dict —
combine → neutralize → size → limit → orders — with no change to the engine
itself. The infrastructure is ready; the edge is the missing piece, and finding
one honestly is the rest of the project.

*Reproduce:* `python scripts/engine_demo.py --strict` (writes
`results/engine_demo.json` + `results/engine_demo_ramp.png`).
