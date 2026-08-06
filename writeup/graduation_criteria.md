# Graduation criteria & live decision rules — pre-registered 2026-08-06

Frozen now, while there is nothing to graduate (13 trials, zero
graduations; the live experiment has 17 matured cycles and no valid
t-stat), so no result can bias the bar. Any later change is a dated
amendment with a logged reason, and never applies retroactively to a
decision already pending.

**Disclosure at freeze:** 17 matured live ICs were visible when this file
was written (model mean −0.0174, momentum control −0.2350, both
uninterpretable at n=17). Freezing acknowledgment thresholds while the
early data points *against* the model is the conservative direction; the
e-process below still starts at the first matured cycle.

## A. A sleeve may be DISCUSSED for real capital only when ALL of:

1. **Record**: ≥ 120 matured live cycles (≈ 6 months at daily cadence)
   with prediction logs committed BEFORE orders on every cycle and zero
   unexplained gaps (NYSE holidays exempt; `monitor.cycle_continuity` is
   the arbiter).
2. **Live edge**: the model-vs-control e-process (§B) ≥ **20**.
3. **No decay signature**: live mean IC inside the stationary-block-
   bootstrap **80%** interval of the backtest's per-date IC mean
   (Politis–Romano, mean block length = the 21-day horizon, ≥ 2,000
   resamples, on the deployed config's OOS IC series).
4. **Neutrality holds in practice**: realized rolling-63-cycle beta of
   the live book vs SPY at public marks: **|mean| ≤ 0.05 and p95 |β| ≤
   0.15** (measured by the `live_beta_diagnosis` machinery; the 2026-08-06
   diagnosis shows the current construction fails this — see H14).
5. **Costs are real**: net-of-measured-impact Sharpe > 0 with **DSR ≥
   0.95 at the cumulative registered trial count N on that date**. Impact
   uses the own-fill-calibrated k (§5.2 of the research menu) once the
   fill ledger exists; until then the 10 bps/side assumption must be
   stated next to every number it touches.
6. **Adversarial sign-off**: a written red-team review filed in the repo
   addressing all six standing lenses — leakage, artifact, power, costs,
   multiple testing, reproducibility — with no unresolved objection.

Anything less, and the public answer to "should this trade real money?"
remains **no** — in the README, like everything else.

## B. The live e-process (anytime-valid; frozen)

**Data.** Matured cycles only (as-of + 21 trading days). Per-cycle
Spearman IC of the committed predictions against the realized
residualized label (`monitor.realized_live_ic`): model = `pred_raw`,
control = `baseline_mom_12_1`, from the same committed prediction files.

**Dependence control.** Consecutive 21-day-horizon ICs overlap ~20/21 and
are strongly dependent; the e-process updates ONLY on non-overlapping
cycles: the first matured cycle, then the first matured cycle at least 21
trading days after the previous update, in as-of order (≈ one update per
month). All other cycles are displayed, never bet on. This costs power
and buys validity — the honest trade.

**Construction** (test supermartingale, betting on signs; magnitude is
deliberately discarded — conservative and fat-tail robust):

- Edge process: D_k = IC_model − IC_control at update k.
  Backtest-null process: D_k = IC_model.
  Decay process: D_k = 0.0225 − IC_model (0.0225 = the deployed config's
  frozen OOS mean IC, `results/metrics_sp500_ridge_both_residlabel.json`).
- s_k = sign(D_k) ∈ {−1, +1}; D_k = 0 → skip, no bet.
- E_T(λ) = Π_{k≤T} (1 + λ·s_k). Under H0 (median D ≤ 0), each factor has
  expectation ≤ 1, so E is a supermartingale and Ville's inequality gives
  anytime-valid error control.
- Mixture over the bet size: Ē_T = mean of E_T(λ) over the frozen grid
  λ ∈ {0.05, 0.10, …, 0.50} (10 points, uniform).

**Frozen thresholds and responses** (act on these and nothing else):

| e-process | Ē ≥ 20 means | frozen response |
|---|---|---|
| edge (vs control) | live edge over momentum control acknowledged (anytime-valid p ≤ 0.05) | log entry + graduation criterion 2 satisfied |
| backtest-null | live IC positive vs zero | log entry only |
| decay | live IC below the backtest's mean — decay declared | log entry + investigation |

No threshold crossing ever triggers an automatic model, allocation, or
config change (the live config stays frozen mid-experiment; the same rule
as BOCPD alerts). Peeking is free by construction — that is the point —
but the update SCHEDULE, grid, and thresholds may only change by dated
amendment before the affected cycles mature.

## C. Status at freeze (2026-08-06)

Zero graduations. The meta-allocator idles by design until a sleeve
passes §A. Criterion 4 is currently FAILED by the live construction
(realized SPY beta +0.53, ex-ante w·β_SPY positive 38/38 cycles) — the
registered fix is H14 in `writeup/preregistered_hypotheses.md`; the
diagnosis is `writeup/live_vol_diagnosis_2026-08-06.md`. Criterion 5's
own-impact calibration does not exist yet (fill ledger unbuilt). The
e-process machinery is registered here before its implementation lands;
the implementation must reproduce this file's definitions exactly and
carry known-answer tests before the first update is computed.
