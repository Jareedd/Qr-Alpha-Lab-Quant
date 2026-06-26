# Bloomberg data pull — H13 PEAD study (bounded, license-respecting)

**Goal:** one small CSV of historical earnings *surprises* (actual vs the
**pre-announcement consensus**) for the S&P 500. That consensus is the single
thing free data cannot give us — everything else (prices, returns) we already
have from Tiingo. So the Bloomberg trip is small and targeted.

> **⚠ License + access discipline (read first).** The terminal is licensed for
> *individual, non-systematic* research use. This pull is a **bounded, one-time
> research extract** (~500 names × ~10 yrs of quarterly earnings = a small dataset),
> which is fine. Do **NOT** attempt to harvest a full price/fundamentals universe
> or automate bulk extraction — that violates the terminal agreement and can get
> your ASU access revoked. Watch the **data-limit indicator** and stop if you near
> the monthly cap. This is the same integrity line the project holds everywhere.

## Where
Noble Library, 2nd floor, **room 276** (book a slot via ASU's Bloomberg booking
page) or the **W. P. Carey FAR Lab** if you have lab access. Bring your ASU login
to save the Excel file to your OneDrive/email afterwards.

## What to pull (per S&P 500 ticker)
For every quarterly earnings announcement, ~2015→today:

| column (CSV header) | what it is | why |
|---|---|---|
| `ticker` | e.g. `AAPL` | join key |
| `ann_date` | announcement date (YYYY-MM-DD) | the PIT event date (signal enters AFTER this) |
| `period` | fiscal quarter, e.g. `2023Q1` | dedup / sanity |
| `actual_eps` | reported EPS | numerator |
| `est_eps` | **consensus estimate at the announcement** | the PIT consensus free data lacks |
| `surprise_pct` | Bloomberg's reported surprise % | PIT by construction (Bloomberg uses the at-the-time consensus) |
| `num_est` | # of contributing analysts | optional — SUE denominator / quality filter |
| `std_est` | std-dev of estimates | optional — SUE = (actual−est)/std_est |

The harness needs **`ticker, ann_date, actual_eps, est_eps`** at minimum;
`surprise_pct`/`std_est`/`num_est` make a better signal (SUE) if they fit the limit.

## How (two ways — use whichever exports cleanest)
1. **Excel add-in (preferred, bounded).** On the terminal, open Excel with the
   Bloomberg add-in. Use `FLDS <GO>` to confirm the exact field mnemonics by
   searching **"earnings surprise"**, **"EPS actual"**, **"EPS estimate"** — Bloomberg
   field names drift, so verify them live. Then a bulk-data pull per ticker, e.g.
   `=BDS("AAPL US Equity","<earnings-surprise-history field>")` returns the dated
   history. Paste the universe down column A (the ~500 tickers — bring the list
   from `results/` or `universe`), fill the formula across, let it populate, then
   **paste-special → values** and save as CSV. Keep the field count small to stay
   under the data limit.
2. **`ERN <GO>` per ticker (fallback).** Type `<ticker> US Equity ERN <GO>` →
   the **Earnings** tab shows quarterly Reported / Estimate / Surprise% with dates →
   export to Excel. Slower (one ticker at a time) but unambiguous and PIT-correct.

## Staying within the limit
~500 names × ~40 quarters × ~5 fields ≈ **~100k data points**. University
terminals typically allow a few hundred thousand data points/month, so this should
fit — but **watch the indicator**. If you get close: pull **~250 names** or **5
years** first (enough for a first graded run), or split across two appointments.
A smaller, clean pull beats a large, throttled one.

## Output
Save as **`data_cache/bloomberg/pead_surprises.csv`** with the headers above, then:

```
python scripts/run_pead.py            # registration-gated; runs trial #14
```

The harness is built and **offline-tested on a synthetic PEAD world** before you
ever go to the lab — so if the CSV has the right columns, the graded run is one
command. No Bloomberg data ever enters the repo as a redistributable dataset
(`data_cache/` is git-ignored); only the resulting metrics/figures are committed,
exactly like every other trial.
