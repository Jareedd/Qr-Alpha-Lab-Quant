export const meta = {
  name: 'managed-research',
  description: 'Budget-disciplined research fan-out: capped HAIKU workers + ONE capable synthesis. Designed so a fan-out can never blow org usage — invoke with args = the question (or {question, maxAgents, workerModel}).',
  phases: [
    { title: 'Search', detail: 'capped haiku searchers, one per angle' },
    { title: 'Verify', detail: 'haiku adversarial check of load-bearing claims' },
    { title: 'Synthesize', detail: 'ONE capable-model cited report' },
  ],
}

// ============================================================================
// AGENT MANAGER — the guardrails. The blowup (3.2M tokens, 100+ agents) happened
// because every fan-out agent inherited the expensive main-loop model. The fix:
//   1. ALL fan-out workers run on the cheap tier (haiku) — ~10-20x cheaper.
//   2. Only the SINGLE final synthesis uses a capable model.
//   3. A HARD agent ceiling that no budget setting can exceed.
//   4. A live budget-remaining guard: stop spawning before an overrun, return
//      partial results gracefully instead of failing the whole run.
// Tune via args, but the hard cap always wins.
// ============================================================================
const A = (typeof args === 'object' && args) ? args : {}
const QUESTION = typeof args === 'string' ? args : (A.question || '')
const WORKER_MODEL = A.workerModel || 'haiku'   // cheap tier for every fan-out worker
const SYNTH_MODEL = A.synthModel || undefined    // inherit capable model for the one synthesis
const HARD_AGENT_CAP = Math.min(A.maxAgents || 18, 24)   // absolute ceiling — budget can only LOWER it
const SAFETY_MARGIN = 40_000                      // stop spawning when remaining < this (if a budget is set)
const MAX_ANGLES = 5
const MAX_VERIFY = 10

if (!QUESTION) { log('ERROR: no question provided in args'); return { error: 'no question' } }

let spawned = 0
const canSpawn = () => spawned < HARD_AGENT_CAP &&
  (!budget.total || budget.remaining() > SAFETY_MARGIN)
const tick = (n = 1) => { spawned += n }
const spentk = () => Math.round(budget.spent() / 1000)
const status = () => log(`[manager] agents ${spawned}/${HARD_AGENT_CAP} · spent ~${spentk()}k` +
  (budget.total ? ` / ${Math.round(budget.total / 1000)}k cap · ${Math.round(budget.remaining() / 1000)}k left` : ' · no hard budget set'))

// Budget-aware fleet sizing: if the user set a "+Nk" budget, scale workers to it;
// else use the default angle count. Never exceed the hard cap.
const angleBudget = budget.total ? Math.max(2, Math.floor(budget.total / 120_000)) : MAX_ANGLES
const N_ANGLES = Math.min(MAX_ANGLES, angleBudget, HARD_AGENT_CAP - 1)  // reserve 1 for synthesis

const CLAIMS_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: {
          claim: { type: 'string' },
          evidence: { type: 'string' },
          source: { type: 'string' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
        required: ['claim', 'evidence', 'source', 'confidence'],
      },
    },
  },
  required: ['findings'],
}
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    verdict: { type: 'string', enum: ['confirmed', 'refuted', 'uncertain'] },
    reason: { type: 'string' },
  },
  required: ['verdict', 'reason'],
}

// ---- Phase 1: decompose + cheap parallel search (haiku) ----
phase('Search')
log(`[manager] question: ${QUESTION.slice(0, 120)}`)
log(`[manager] plan: ${N_ANGLES} haiku searchers + haiku verify + 1 capable synthesis (hard cap ${HARD_AGENT_CAP})`)

const angles = Array.from({ length: N_ANGLES }, (_, i) => i)
const searches = await parallel(angles.map(i => () => {
  if (!canSpawn()) return Promise.resolve(null)
  tick()
  return agent(
    `You are search angle ${i + 1} of ${N_ANGLES} researching: "${QUESTION}". ` +
    `Use web search. Return 3-6 well-sourced, falsifiable findings from YOUR distinct angle ` +
    `(angle ${i + 1}: ${['primary/landscape', 'skeptical/contrarian', 'quantitative/benchmarks', 'practitioner/adoption', 'edge-cases/risks'][i % 5]}). Be concise.`,
    { label: `search:a${i + 1}`, phase: 'Search', model: WORKER_MODEL, schema: CLAIMS_SCHEMA })
})).then(r => r.filter(Boolean))
status()

const allFindings = searches.flatMap(s => s.findings || [])
// dedupe near-identical claims cheaply (first ~80 chars), keep highest confidence
const seen = new Map()
for (const f of allFindings) {
  const k = f.claim.slice(0, 80).toLowerCase()
  if (!seen.has(k) || f.confidence === 'high') seen.set(k, f)
}
const unique = [...seen.values()]
log(`[manager] ${allFindings.length} findings -> ${unique.length} unique`)

// ---- Phase 2: verify only the load-bearing claims (haiku), budget-capped ----
phase('Verify')
const toVerify = unique
  .filter(f => f.confidence !== 'low')
  .slice(0, Math.min(MAX_VERIFY, Math.max(0, HARD_AGENT_CAP - spawned - 1)))  // always reserve 1 for synthesis
const verified = await parallel(toVerify.map(f => () => {
  if (!canSpawn()) return Promise.resolve({ ...f, status: 'unchecked' })
  tick()
  return agent(
    `Adversarially fact-check this claim about "${QUESTION}". Try to REFUTE it; default to "uncertain" if you cannot confirm from a real source.\nCLAIM: ${f.claim}\nEVIDENCE: ${f.evidence}\nSOURCE: ${f.source}`,
    { label: `verify`, phase: 'Verify', model: WORKER_MODEL, schema: VERDICT_SCHEMA })
    .then(v => ({ ...f, status: v ? v.verdict : 'unchecked', vreason: v?.reason }))
    .catch(() => ({ ...f, status: 'unchecked' }))
})).then(r => r.filter(Boolean))
status()

const confirmed = verified.filter(v => v.status === 'confirmed' || v.status === 'unchecked')
const refuted = verified.filter(v => v.status === 'refuted')

// ---- Phase 3: ONE capable-model synthesis ----
phase('Synthesize')
tick()
const dossier = verified.map(v => `[${v.status}] ${v.claim} (src: ${v.source})`).join('\n')
const report = await agent(
  `Synthesize a tight, cited report answering: "${QUESTION}".\n\n` +
  `Verified findings (status = adversarial check):\n${dossier}\n\n` +
  `Lead with the bottom line. Separate CONFIRMED from UNCERTAIN. Note anything REFUTED and exclude it from claims. Cite sources inline. Be honest about gaps.`,
  { label: 'synthesize', phase: 'Synthesize', model: SYNTH_MODEL })
status()

log(`[manager] DONE — ${spawned} agents, ~${spentk()}k tokens. ` +
  (spawned >= HARD_AGENT_CAP ? 'hit the hard agent cap (partial coverage).' :
    (budget.total && budget.remaining() <= SAFETY_MARGIN ? 'stopped early on budget margin (partial coverage).' : 'within budget.')))

return {
  report,
  agentsUsed: spawned,
  tokensSpentK: spentk(),
  findingsConfirmed: confirmed.length,
  findingsRefuted: refuted.length,
  cappedEarly: spawned >= HARD_AGENT_CAP || (budget.total && budget.remaining() <= SAFETY_MARGIN),
}
