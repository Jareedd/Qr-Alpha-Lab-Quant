# Name->CIK Crosswalk Recovery Spot-Check (H1 fundamentals source)

**Date:** 2026-06-24
**Input:** `results/h1_name_crosswalk_audit.json` (309 dead names; 290 recovered a `name_cik`).
**Method:** For every recovered row, load the cached SEC submissions metadata
`data_cache/fundamentals/sub_<name_cik>.json` and compare the Wikipedia removed-name
against the SEC entity's **current name + full `formerNames` history** (with date
ranges) and last 10-K filing date. A recovery is **CORRECT** if the wiki name equals
the SEC current name OR any former name, counting acquisition/rename **continuations**
as correct (e.g. Sara Lee -> Hillshire, AK Steel -> Cleveland-Cliffs/CLF, Time
Warner -> WarnerMedia). No source/code was modified; no network calls; no graded
trial run (N stays 11).

## Headline verdict

| Bucket | Count | Share |
|---|---|---|
| Recoveries with a `name_cik` | 290 | 100% |
| **Clearly-correct** | **286** | **98.6%** |
| Correct-but-notable-drift (footnote) | 2 (ADT, TYC) | 0.7% |
| **Suspicious / likely-wrong** | **2 (AAL, GHC)** | **0.7%** |

**Overall trustworthiness: HIGH.** Resolving by *name* (not ticker) eliminates the
dominant leak mode — a recycled ticker pointing at an unrelated SPAC/successor. Of the
20 rows where the name route actively *disagreed* with the baseline ticker->CIK, 18
are the name route correctly preferring the true historical operating entity over a
recycled-ticker shell (most baseline CIKs in those disagreements have **no cached
submissions file at all** — they are exactly the reassigned tickers the name route is
designed to avoid). Only **one** disagreement is the name route getting it wrong (GHC).

Verification was anchored on SEC's own `formerNames` date-ranged history, which is the
authoritative chain for rename/merger continuations, so continuation calls are not
guesswork. A random 18-row sample of the "clearly-correct" bucket was hand-checked and
all 18 matched (current or documented former name, with last-10-K dates consistent with
each firm's known delisting/acquisition).

## Suspicious list (action required before the graded run)

| Ticker | Wiki name | Recovered CIK -> SEC name | Last 10-K | Problem |
|---|---|---|---|---|
| **GHC** | Graham Holdings | 0000716314 -> **GRAHAM CORP** (ticker GHM) | 2026-06-08 | **Wrong entity — true namesake collision.** "Graham Holdings" resolved to *Graham Corporation*, a small industrial pump/machinery maker (SIC: General Industrial Machinery), still alive under ticker GHM. The correct company is **Graham Holdings Co, CIK 0000104889, ticker GHC** (the ex-Washington Post Co). Worse: the baseline ticker->CIK route had it **right** (104889) and the name route **overrode** it with the wrong CIK. This would inject an unrelated industrial firm's fundamentals under GHC. |
| **AAL** | American Airlines Group | 0000004515 -> **AMERICAN AIRLINES, INC.** (no listed ticker) | 2026-02-18 | **Subsidiary-vs-parent mismatch.** Resolved to the operating *subsidiary* American Airlines, Inc. (files its own 10-Ks, but is not the listed S&P constituent). The listed constituent is the parent **American Airlines Group Inc, CIK 0000006201, ticker AAL** (ex-AMR Corp) — which is the baseline CIK. Pulling 4515 yields subsidiary-level, not consolidated listed-entity, financials. |

## Watch / footnote (correct continuation, but notable drift — confirm intent)

| Ticker | Wiki name | Recovered CIK -> SEC name | Note |
|---|---|---|---|
| ADT | ADT | 0000833444 -> Johnson Controls (JCI), via former name "ADT LIMITED" (1995-1997) -> Tyco -> JCI | Long chain. The *modern* ADT Inc (2018 IPO, ticker ADT, CIK 1703056 = baseline) is a **different relisted entity**. Whether ADT->Tyco->JCI is right depends on which ADT era the H1 window covers. Same CIK now serves both the ADT and TYC rows. |
| TYC | Tyco International | 0000833444 -> Johnson Controls (JCI) | Correct merger continuation; flagged only because it shares the CIK with the ADT row above. |

## Why the other "disagreements" are fine (not suspicious)

The 20 name-vs-baseline disagreements were each inspected. Aside from GHC and AAL, all
are the name route correctly selecting the genuine historical operating entity:

- **Continuations confirmed via SEC `formerNames`:** AA/Alcoa (-> Arconic -> Howmet,
  former "ALCOA INC"), SLE/Sara Lee (-> Hillshire Brands, former "Sara Lee Corp"),
  S/Sprint Nextel (-> Sprint LLC, former "SPRINT NEXTEL CORP"), EP/El Paso (-> El Paso
  LLC, former "EL PASO CORP/DE"), MI/Marshall & Ilsley, MMI/Motorola Mobility,
  SE/Spectra Energy, POM/Pepco Holdings, CPWR/Compuware, LIFE/Life Technologies,
  KG/King Pharmaceuticals, FTI/FMC Technologies, TE/TECO Energy, SII/Smith
  International, STI/SunTrust, APC/Anadarko, DV/DeVry (-> Adtalem -> Covista, the real
  DeVry lineage).
- **Baseline CIK was the recycled-ticker shell** (no cached submissions file exists for
  the baseline CIK) in: AAL-baseline aside, KG, MI, EP, TE, S, and others — i.e. the
  name route is doing exactly its job of dodging ticker reassignment.

## Trustworthiness read for the H1 build

- **~98.6% clean** on the 290 recoveries, with the single hard error (GHC) being a
  classic dead/live **namesake collision** — precisely the residual risk the audit's
  own note flagged ("dead/live NAMESAKE collision; dead_by date gate is the next
  refinement"). GHC is a useful concrete motivating example for wiring the `dead_by`
  index-removal gate into `NameCikResolver.operating_cik(name, dead_by=...)`: Graham
  Holdings was removed from the S&P long ago, and the colliding Graham Corp is still
  live, so the dead_by gate would disambiguate it.
- **AAL** is a different failure class (subsidiary vs listed parent), not fixed by the
  dead_by gate; it needs a "prefer the entity that is/was the listed issuer" tiebreak,
  or simply trusting the baseline ticker->CIK when it exists and points at a listed
  parent.
- No fabricated data, no code touched, no trial graded. Recommend: before the graded
  run, hard-override GHC -> 0000104889 and AAL -> 0000006201 (or implement the dead_by
  gate + listed-issuer tiebreak), then the source is safe to compose into
  `pit_feature_panels`.

## Reproducibility

Spot-check artifact written to `results/spotcheck_results.json` (per-row: ticker, wiki
name, name_cik, baseline_cik, SEC current name, formerNames, last 10-K date, last filing
date, current tickers, SIC, token-overlap score). Regenerable from the audit JSON + the
cached `sub_<CIK>.json` files with no network.
