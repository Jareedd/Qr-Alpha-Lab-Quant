# Live paper-trading monitor — as of 2026-06-25

## Cycle continuity
- cycles logged: **10** (2026-06-10 → latest 2026-06-25)
- prediction logs: **9** of 10 cycles (weights-only cycles predate prediction logging and cannot yield live IC)
- weekdays in window with NO log: **2** — 2026-06-16, 2026-06-19  *(NYSE holidays are not modeled and appear here; anything else is a missed cycle and must be explained)*

## Live IC vs backtest IC
- measurable cycles: **0** of 9 logged (a cycle matures 21 trading days after its as-of date)
- live mean rank IC: *not yet measurable*
- backtest mean rank IC (same config, 2010→2026 OOS): **+0.0225** (t_NW = 1.91)
- **do not interpret yet**: t_NW needs > 23 matured cycles; early ICs are single noisy draws

### Control arm (12-1 momentum baseline, shadow-logged — no orders)
- baseline live IC: *not yet measurable*
- purpose: if the model's live IC sags vs backtest, the baseline's own live-vs-backtest gap separates 'model decayed' from 'period was hostile to everything'

## Data revisions (vendor rewriting the shared past)
- snapshot pairs compared: **8**; latest (2026-06-24 → cycle): 6,396 of 1,302,197 shared price cells changed (0.4912%), **287 return cells** changed (max |Δreturn| 1.84e+01)
- price-level changes are mostly benign re-adjustments; *return* changes alter features/labels — they are why backtest and live model literally saw different versions of the same past

## Realized book P&L (public-price marks, gross, no costs)
- 7 trading days marked; cumulative +4.30%, ann. vol 10.50%
- cross-check only: fills, costs and shorts-availability live at the broker; the Alpaca equity curve is authoritative

## Standing limitations
- live IC residualizes vs the equal-weight mean of logged names, not the full PIT universe (close, not identical, market proxy)
- yfinance marks are split/dividend-adjusted closes; broker fills will differ
- this monitor is read-only: it never feeds back into the strategy
