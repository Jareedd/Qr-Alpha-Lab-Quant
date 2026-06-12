# H2 design note — crypto-perp funding carry (pre-data, 2026-06-12)

Status: design complete, registration amended (see
`preregistered_hypotheses.md` H2, amendment dated 2026-06-12). **No exchange
data has been downloaded as of this document.** The run is trial #8, gated on
owner sign-off.

## 1. Steelman of the null (why this SHOULD fail)

- **It is the most visible premium in crypto.** Funding rates are printed on
  every exchange UI and aggregated by free dashboards; "short the payers" is
  the most retail-legible systematic trade in the asset class. Harvey–Liu–Zhu
  logic says visible premia attract capital until they are fee-sized.
- **It is harvested at industrial scale.** Delta-neutral basis/carry farms
  (the Ethena-style structure) hold billions in exactly this trade. The
  marginal carry seller today is a funded institution, not a hobbyist.
- **Crypto decays faster than equities.** The McLean–Pontiff post-publication
  decay that took years in equities plays out in months in crypto; published
  perp-carry backtests largely predate 2022's institutionalization.
- **The seller's risk is structural, not statistical.** Short-the-crowded-long
  collects pennies until a squeeze collects them back: the strategy is short
  a convexity the funding rate is paying for. Even a positive mean must
  survive negative skew (the DSR's PSR core penalizes skew explicitly) and
  single-name blowups (kill criterion: survive excluding the top-3 earners).
- **Fees are first-order at h=7d.** Taker fees + spread on both legs of a
  weekly-rebalanced book are a few-bps-per-day drag against a premium
  measured in bps per day. The null says: net of realistic costs, nothing is
  left. Trials #2–7 died exactly this death in equities.

## 2. The cheapest test that could kill it

One line, no ML: rank the PIT universe by trailing 7-day funding, short the
top quartile, long the bottom, equal weight, weekly rebalance, on
funding-inclusive total returns, net of taker fees + spread. If THAT shows
nothing on the honest universe, the elaborate version is dead and trial #8
is spent honestly. This baseline-first test IS the registered config — the
ML variant only earns a registration if the baseline survives.

## 3. The label (the part that silently invalidates everything if wrong)

A perp position's daily P&L has two legs: mark-price return and funding
transfer. For a LONG with funding rate F_t (positive = longs pay shorts):

    total_return_long(t) = mark_return(t) − F_t

The carry premium, if it exists, lives in the funding leg. A price-only
label measures the wrong object — demonstrated, not asserted, in the
synthetic lab: on the planted-carry world the same book scores SR ≈ +0.9 to
+1.6 funding-inclusive and SR ≈ −1.9 to −2.7 price-only
(`tests/test_synthetic_carry.py::test_price_only_label_measures_the_wrong_object`).
The null world (`priced_carry`) encodes the efficient-market counterfactual:
funding fully offset by mark drift, total returns unpredictable from funding.

## 4. Timestamp conventions (law 2: decided before they can bite)

- Binance perp funding settles every 8h at 00:00 / 08:00 / 16:00 UTC; the
  rate applied at settlement is fixed ahead of it (published as the
  prevailing rate during the window).
- **Daily cycle convention:** features at day t use only funding SETTLED at
  or before t 00:00 UTC plus klines through t's close; the trade is assumed
  at the next day's first price. Funding accrued to the label is every
  settlement in the holding window (t_entry, t_exit].
- Trailing-funding feature = mean of the last 21 settlements (7 days × 3) —
  fully realized transfers only, never the "predicted next rate" the UI
  shows (that quantity is partially forward-looking by construction).

## 5. Point-in-time universe (perps die too)

- Source: Binance public dumps (data.binance.vision) enumerate every
  contract that ever traded, including delisted ones (LUNA, FTT, dozens of
  low-caps) with full terminal history — the delisting-return hole equities
  have does NOT exist here; dead contracts keep printing to their final day.
- Universe at t: top-30 perps by trailing 30d dollar volume among contracts
  LISTED at t (listing date = first kline in the dumps; delisting = last).
  No knowledge of future listing/delisting/volume enters the ranking.
- Residual honesty item: the dumps are the exchange's own record; if a
  contract's history were retroactively pruned we could not see it. Stated
  as a limitation, judged minor (delisted contracts demonstrably persist).

## 6. Cost model (ported, parameters set now)

- Taker fee 5.0 bps per side (Binance USDT-perp VIP0 4.5 bps, rounded
  against ourselves), spread 2 bps per side for liquid top-30 names,
  square-root impact reusing `quantlab.impact` with perp dollar-ADV from the
  dumps. Headline result at $1M notional; capacity sweep as in Phase 5.
- Weekly rebalance, quartile book (top-30 universe → ~7 names/side; deciles
  at n=30 are 3-name lottery tickets).

## 7. Paired controls for the real run (registered up front)

1. **Shuffled-funding control:** rebuild the book on cross-sectionally
   permuted funding ranks (same dates, same names, same costs). Must show
   ~nothing; if it "earns", the harvest is structure, not carry.
2. **Decomposition reporting:** funding income vs price drag reported
   separately — a result whose entire premium is one regime/period/name
   fails the spirit even if the mean passes.
3. **Synthetic gate:** `make_perp_panel` planted/priced worlds must pass
   their tests in the same environment immediately before the real run.

## 8. What failure means

Carry consumed by fees/crowding at our fidelity → the write-up gains "the
pipeline generalizes; the free lunch does not" with crypto evidence — the
same sentence trials #2–7 earned in equities, in a second asset class. That
is a publishable section either way, which is the only reason to spend
trial #8 at all.
