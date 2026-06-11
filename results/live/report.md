# Live paper-trading monitor — as of 2026-06-11

## Cycle continuity
- cycles logged: **1** (2026-06-10 → latest 2026-06-10)
- prediction logs: **0** of 1 cycles (weights-only cycles predate prediction logging and cannot yield live IC)
- weekdays in window with NO log: **1** — 2026-06-11  *(NYSE holidays are not modeled and appear here; anything else is a missed cycle and must be explained)*

## Live IC vs backtest IC
- *(offline run: prices not fetched, IC not computed)*

## Realized book P&L (public-price marks, gross, no costs)
- no marked days yet (first book earns from its t+1 open)

## Standing limitations
- live IC residualizes vs the equal-weight mean of logged names, not the full PIT universe (close, not identical, market proxy)
- yfinance marks are split/dividend-adjusted closes; broker fills will differ
- this monitor is read-only: it never feeds back into the strategy
