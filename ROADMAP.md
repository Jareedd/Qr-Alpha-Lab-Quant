# 12-Week Semester Roadmap

Goal: turn the working skeleton into an interview-dominating research project. Each phase ends with something demonstrable. Keep a `research_log.md` from day one — record every variant you try (this is your honest `--n-trials` count, and discussing that discipline is itself interview gold).

## Weeks 1–2 — Own the foundation
Run everything, read every module until you can rewrite it from memory. Whiteboard-derive the DSR formula and the embargo logic — these are direct interview questions ("how do you know your backtest isn't overfit?"). Get real data flowing locally via yfinance; commit the repo to GitHub with CI running the test suite (GitHub Actions, one YAML file).

## Weeks 3–4 — Kill survivorship bias, add benchmarks
Build a point-in-time-ish universe (e.g., scrape historical S&P 500 membership from Wikipedia revisions, or use the free Sharadar sample). Add benchmark comparisons: your signal vs. naive equal-weight, vs. pure 12-1 momentum rank. If ML doesn't beat the one-line momentum baseline net of costs, say so in the write-up — that finding is *more* credible to a QR interviewer than a too-good backtest.

## Weeks 5–6 — Risk neutralization
Add sector and beta neutralization to portfolio construction (regress out, or constrain weights). Compare neutralized vs. raw long-short: factor exposure usually explains most of a naive signal's return. Add a simple risk report per run: rolling beta, sector tilts, factor correlation.

## Weeks 7–8 — Better labels and features
Try: residualized returns as labels (vs. market), multiple horizons (5/21/63d), feature interactions via the GBR model, feature importance stability across walk-forward windows (unstable importance = overfitting tell). Log every experiment; let the DSR trial count grow honestly.

## Weeks 9–10 — Execution realism
Add a square-root market-impact term to costs; model rebalance at next open vs. close; measure capacity (at what AUM do costs eat the edge?). Write the capacity analysis up — capacity is the question this whole project's research report (see parent folder) shows the industry actually cares about.

## Week 11 — Live paper trading
Deploy the best honest config to Alpaca paper trading (free API) with a daily cron job and a one-page monitoring dashboard. Even 4–6 weeks of live paper results separates you from 99% of candidates: live IC vs. backtest IC is the ultimate out-of-sample test.

## Week 12 — The write-up
Produce a 6–10 page research note (the artifact recruiters actually read): question, data, methodology, results with DSR, what failed, capacity, live results. Structure it like an AQR white paper. Post the repo + note on GitHub; put the planted/noise falsification test in the README header — it is your differentiator.

## Stretch goals
Numerai submission using the same pipeline discipline; IMC Prosperity (next edition) as a team; re-run the whole study on crypto perps via free exchange data (no survivorship issues, 24/7 data, capacity analysis is fun there).

## Interview talking points this project buys you
Why k-fold CV fails on financial data (embargo/purging); why your Sharpe is deflated and by how much; why turnover is the strategy-killer; what the planted-noise test caught during development; why beating a momentum baseline is hard; what you'd need to make the backtest institutional-grade (point-in-time data, impact model, borrow costs).
