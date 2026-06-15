# Agent budget policy — don't blow usage, still get the work done

**Why this exists.** A deep-research fan-out spent ~3.2M tokens across 108 agents,
and a later ideation workflow hit the **org monthly spend limit** mid-run (its
agents then failed). Root cause: every fan-out agent inherited the expensive
main-loop model (Opus). These rules make that impossible by default.

## The five rules (apply to EVERY workflow)

1. **Cheap workers, capable synthesis (the #1 lever).** Every fan-out worker —
   search, fetch, extract, verify, score — runs on **`haiku`** (`opts.model:
   'haiku'`). Only the single final **synthesis** uses the capable default model.
   Haiku is ~10–20× cheaper per token; a 20-worker + 1-synth run on this split
   costs a small fraction of the same run on Opus.

2. **Hard agent ceiling.** A single workflow spawns **≤ 18 agents by default,
   ≤ 24 ever** without explicit user opt-in. A budget setting can only *lower*
   the cap, never raise it. Track a counter and gate every `agent()` call on it.

3. **Live budget guard.** If the turn has a `+Nk` budget, stop spawning once
   `budget.remaining() < ~40k` and return **partial results** (`.filter(Boolean)`)
   rather than failing the run. Log `budget.spent()` after each phase.

4. **No multiplicative nesting.** Never `parallel(... parallel(...))` that
   multiplies counts (e.g. 5 dimensions × 3 verifiers = 15 is fine; 20 × 5 = 100
   is not). Dedupe before any expensive stage; only verify load-bearing claims.

5. **Estimate before launching; fall back when capped.** Rough cost ≈
   `agents × ~avg_tokens × model_rate`. If a task needs more than the hard cap or
   a large share of remaining budget, **scale down or ask first**. If the org
   spend limit is already hit, workflows/subagents WILL fail — **fall back to
   direct local compute** (Bash/Python), which costs no API spend (that is how
   the trials + screens in this repo were finished after the limit hit).

## How to use

- **Research questions →** invoke the managed harness:
  `Workflow({ name: 'managed-research', args: '<question>' })`, or
  `args: { question, maxAgents, workerModel, synthModel }` to tune (caps still win).
- **Custom workflows →** copy the `AGENT MANAGER` block from
  `managed-research.js` (the `canSpawn()` guard + `model: 'haiku'` on workers +
  capable model only on the synthesis) into the new script.

## Quick cost intuition

| run shape | rough relative cost |
|---|---|
| 100 workers, all Opus (the blowup) | 100× |
| 18 workers Opus + 1 Opus synth | ~19× |
| **18 workers haiku + 1 capable synth (this policy)** | **~2–3×** |
| direct local compute (no agents) | ~0 API spend |

The goal is not "never fan out" — it's "fan out cheaply, cap it hard, and never
let a single run threaten the org limit."
