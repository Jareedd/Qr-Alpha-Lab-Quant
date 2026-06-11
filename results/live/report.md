# Live paper-trading monitor — as of 2026-06-10

## Cycle continuity
- cycles logged: **1** (2026-06-10 → latest 2026-06-10)
- prediction logs: **0** of 1 cycles (weights-only cycles predate prediction logging and cannot yield live IC)
- weekdays in window with NO log: **0** — record is gap-free

## Live IC vs backtest IC
- measurable cycles: **0** of 0 logged (a cycle matures 21 trading days after its as-of date)
- live mean rank IC: *not yet measurable*
- backtest mean rank IC (same config, 2010→2026 OOS): **+0.0225** (t_NW = 1.91)
- **do not interpret yet**: t_NW needs > 23 matured cycles; early ICs are single noisy draws

## Realized book P&L (public-price marks, gross, no costs)
- no marked days yet (first book earns from its t+1 open)

## Standing limitations
- live IC residualizes vs the equal-weight mean of logged names, not the full PIT universe (close, not identical, market proxy)
- yfinance marks are split/dividend-adjusted closes; broker fills will differ
- this monitor is read-only: it never feeds back into the strategy
