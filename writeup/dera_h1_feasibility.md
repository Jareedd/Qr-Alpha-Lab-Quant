# Free survivorship-safe H1 via SEC DERA — Stage-1 feasibility

**Status: Stage-1 feasibility (zero-trial). Loader built and unit-tested
(`src/quantlab/dera.py`, `tests/test_dera.py`); the full multi-quarter
reconstruction is a dedicated follow-up. N unchanged (= 11).**

## The question

H1 (quality fundamentals) is blocked on survivorship: SEC's
`company_tickers.json` is current-only, so ~27% of the PIT S&P universe — the
dead/renamed names — is unmapped, and a free historical ticker→CIK recovery
lifts survivorship-safe coverage only **73% → 75%** with reassignment risk
(measured 2026-06-15, `cik_history.py`). Can the SEC **DERA Financial Statement
Data Sets** — bulk quarterly archives of every filer's numeric facts, keyed on
CIK and filing date, back to 2009 — get H1 to a creditable free run without
waiting on CRSP/Compustat?

## What DERA cleanly solves: fundamentals-by-CIK, survivorship-safe, PIT

DERA `sub.txt` (submissions) + `num.txt` (numeric facts) give, for **every filer
that ever filed** (dead companies included — filings persist under their CIK),
the reported value of any us-gaap tag (Assets, NetIncomeLoss, CFO, Revenue,
CoGS…), stamped by **filing date**. `dera.pit_value(cik, tag, asof)` returns the
freshest figure a filer disclosed on or before `asof` — the same
survivorship-safe, point-in-time primitive `FreeSECSource.field_series` provides,
but sourced in bulk and without the current-only ticker dependency on the
fundamentals side. The unit tests confirm a *dead* filer (Activision, CIK 718877)
resolves by CIK exactly as a live one does. **This side of H1 is closed by DERA.**

## What DERA does NOT solve: the dead-ticker → CIK crosswalk (the real bottleneck)

The H1 universe is defined by **ticker** (PIT S&P membership from Wikipedia's
changes table). To attach DERA fundamentals to a name's returns we need its CIK.
DERA does not provide that bridge for dead names:

- `num.txt` is **numeric only** — `dei:TradingSymbol` is a *text* fact and is
  absent from the standard datasets.
- `sub.txt` carries the company **name**, not a ticker.

So mapping a dead **ticker** → CIK still requires a name/ticker bridge, and our
PIT universe carries **only tickers, no company names** (established 2026-06-15).
DERA therefore *reinforces* the standing finding rather than overturning it:
**fundamentals coverage is no longer the binding constraint — the dead-ticker→CIK
crosswalk is**, and free sources cap it near 75%.

## What would actually close it

1. **A name-bearing historical S&P membership list.** With dead-name *company
   names*, DERA's `sub.txt` (a complete CIK↔name index across all filers) becomes
   a free name→CIK crosswalk, and survivorship-safe coverage could rise well past
   75%. The missing input is a names column the current free universe lacks.
2. **CRSP** (the `permno ↔ ticker ↔ cik` linkage table) — the institutional
   answer, which is what H1's `CompustatSource` slot waits on.

## Honest verdict

DERA is a real, free improvement to the *fundamentals* half of H1 and is now
loadable and tested. But "free survivorship-safe H1" is gated not on fundamentals
coverage — which DERA closes — but on the dead-ticker→CIK crosswalk, which it does
not. The next free increment is a name-bearing PIT membership source to feed
DERA's CIK↔name index; absent that, CRSP remains the clean unlock. Either way the
loader is built, so the day the crosswalk lands H1 can read PIT fundamentals from
bulk DERA without per-CIK API calls.
