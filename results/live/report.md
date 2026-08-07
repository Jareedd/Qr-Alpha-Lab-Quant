# Live paper-trading monitor — as of 2026-08-07

## Cycle continuity
- cycles logged: **39** (2026-06-10 → latest 2026-08-06)
- prediction logs: **38** of 39 cycles (weights-only cycles predate prediction logging and cannot yield live IC)
- weekdays in window with NO log: **4** — 2026-06-16, 2026-06-19, 2026-07-03, 2026-08-07  *(NYSE holidays are not modeled and appear here; anything else is a missed cycle and must be explained)*

## Live IC vs backtest IC
- measurable cycles: **17** of 38 logged (a cycle matures 21 trading days after its as-of date)
- live mean rank IC: **-0.0175** (t_NW = nan)
- backtest mean rank IC (same config, 2010→2026 OOS): **+0.0225** (t_NW = 1.91)
- **do not interpret yet**: t_NW needs > 23 matured cycles; early ICs are single noisy draws

### Control arm (12-1 momentum baseline, shadow-logged — no orders)
- baseline live mean rank IC: **-0.2352** over 17 matured cycles
- purpose: if the model's live IC sags vs backtest, the baseline's own live-vs-backtest gap separates 'model decayed' from 'period was hostile to everything'

## Data revisions (vendor rewriting the shared past)
- snapshot pairs compared: **37**; latest (2026-08-05 → cycle): 6,491 of 1,311,818 shared price cells changed (0.4948%), **232 return cells** changed (max |Δreturn| 4.33e-03)
- price-level changes are mostly benign re-adjustments; *return* changes alter features/labels — they are why backtest and live model literally saw different versions of the same past

## Realized book P&L (public-price marks, gross, no costs)
- 39 trading days marked; cumulative -0.17%, ann. vol 19.80%
- cross-check only: fills, costs and shorts-availability live at the broker; the Alpaca equity curve is authoritative

## Standing limitations
- live IC residualizes vs the equal-weight mean of logged names, not the full PIT universe (close, not identical, market proxy)
- yfinance marks are split/dividend-adjusted closes; broker fills will differ
- this monitor is read-only: it never feeds back into the strategy
