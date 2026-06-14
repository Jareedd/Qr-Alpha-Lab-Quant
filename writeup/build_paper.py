"""Render the qr-alpha-lab research note as an ASU-themed academic working
paper (PDF), in the restrained empirical-finance register of Greenwood &
Sammon (2025). Reproducible: every number traces to research_log.md /
results/. Run: python writeup/build_paper.py

Honest framing baked in: the cover marks it an UNDERGRADUATE WORKING PAPER,
independent research, with the public code/data/trial-log. ASU theming is
visual only and implies no institutional endorsement.
"""

from __future__ import annotations

import os

from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

MAROON = HexColor("#8C1D40")   # ASU maroon
GOLD = HexColor("#FFC627")     # ASU gold
INK = HexColor("#1A1A1A")
GRAY = HexColor("#6B6B6B")
GOLD_TINT = HexColor("#FBF1D6")  # very light gold for alternating rows
RULE = HexColor("#C8A14B")

OUT = os.path.join(os.path.dirname(__file__), "Galvez_2026_falsification_first.pdf")
SHORT = "Does Anything Survive an Honest Backtest?"

ss = getSampleStyleSheet()


def style(name, **kw):
    base = kw.pop("parent", ss["Normal"])
    return ParagraphStyle(name, parent=base, **kw)


TITLE = style("t", fontName="Times-Bold", fontSize=18, leading=22,
              textColor=MAROON, alignment=TA_CENTER, spaceAfter=4)
AUTH = style("a", fontName="Times-Roman", fontSize=12, leading=15,
             textColor=INK, alignment=TA_CENTER, spaceBefore=6)
AFF = style("af", fontName="Times-Italic", fontSize=10, leading=13,
            textColor=GRAY, alignment=TA_CENTER)
TAG = style("tg", fontName="Helvetica", fontSize=8, leading=11,
            textColor=MAROON, alignment=TA_CENTER, spaceBefore=4)
ABSH = style("abh", fontName="Helvetica-Bold", fontSize=9, leading=12,
             textColor=MAROON, alignment=TA_CENTER, spaceBefore=10)
ABS = style("abs", fontName="Times-Roman", fontSize=9.5, leading=13.5,
            textColor=INK, alignment=TA_JUSTIFY, leftIndent=26, rightIndent=26)
META = style("meta", fontName="Times-Roman", fontSize=8.5, leading=12,
             textColor=GRAY, alignment=TA_CENTER, leftIndent=26, rightIndent=26,
             spaceBefore=4)
H = style("h", fontName="Helvetica-Bold", fontSize=11.5, leading=14,
          textColor=MAROON, spaceBefore=14, spaceAfter=4)
H2 = style("h2", fontName="Helvetica-Bold", fontSize=9.5, leading=12,
           textColor=INK, spaceBefore=8, spaceAfter=2)
BODY = style("b", fontName="Times-Roman", fontSize=10, leading=13.8,
             textColor=INK, alignment=TA_JUSTIFY, spaceAfter=6)
CAP = style("cap", fontName="Helvetica-Oblique", fontSize=8, leading=11,
            textColor=GRAY, spaceBefore=2, spaceAfter=4)
CELL = style("cell", fontName="Times-Roman", fontSize=8.5, leading=11,
             textColor=INK)
CELLH = style("cellh", fontName="Helvetica-Bold", fontSize=8.5, leading=11,
              textColor=white)
REF = style("ref", fontName="Times-Roman", fontSize=8.5, leading=12,
            textColor=INK, alignment=TA_JUSTIFY, leftIndent=14,
            firstLineIndent=-14, spaceAfter=3)
FOOT = style("fn", fontName="Times-Roman", fontSize=7.5, leading=10,
             textColor=GRAY, alignment=TA_JUSTIFY)


def decorate(canvas, doc):
    canvas.saveState()
    w, h = letter
    first = doc.page == 1
    if first:
        # ASU letterhead band
        canvas.setFillColor(MAROON)
        canvas.rect(0, h - 34, w, 34, fill=1, stroke=0)
        canvas.setFillColor(GOLD)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(54, h - 22, "ARIZONA STATE UNIVERSITY")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(w - 54, h - 22, "qr-alpha-lab  ·  Undergraduate Research")
        canvas.setFillColor(GOLD)
        canvas.rect(0, h - 37, w, 2.2, fill=1, stroke=0)
    else:
        canvas.setStrokeColor(MAROON)
        canvas.setLineWidth(0.8)
        canvas.line(54, h - 40, w - 54, h - 40)
        canvas.setFillColor(MAROON)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(54, h - 36, "GALVEZ (2026)")
        canvas.drawRightString(w - 54, h - 36, SHORT)
    # footer
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.6)
    canvas.line(54, 44, w - 54, 44)
    canvas.setFillColor(GRAY)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(54, 33, "Galvez (2026) · qr-alpha-lab")
    canvas.drawCentredString(w / 2, 33, str(doc.page))
    canvas.drawRightString(w - 54, 33, "Working paper — not peer reviewed")
    canvas.restoreState()


def p(text, st=BODY):
    return Paragraph(text, st)


def make_table(data, widths, header=True):
    rows = [[Paragraph(c, CELLH if (header and i == 0) else CELL) for c in row]
            for i, row in enumerate(data)]
    t = Table(rows, colWidths=widths, hAlign="CENTER")
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), MAROON),
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, MAROON),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, GOLD),
        ("LINEBELOW", (0, -1), (-1, -1), 1.0, MAROON),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    for r in range(1, len(data)):
        if r % 2 == 0:
            cmds.append(("BACKGROUND", (0, r), (-1, r), GOLD_TINT))
    t.setStyle(TableStyle(cmds))
    return t


def build():
    doc = BaseDocTemplate(OUT, pagesize=letter, topMargin=0.92 * inch,
                          bottomMargin=0.72 * inch, leftMargin=0.75 * inch,
                          rightMargin=0.75 * inch, title=SHORT,
                          author="Jared Galvez")
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame],
                                        onPage=decorate)])
    S = []

    # ---- title block ----
    S.append(Spacer(1, 8))
    S.append(p("Does Anything Survive an Honest Backtest?", TITLE))
    S.append(p("Falsification-First Evidence on Cross-Sectional Anomalies "
               "in Equities and Crypto", style("sub", parent=TITLE,
               fontSize=12.5, leading=16, spaceAfter=2)))
    S.append(p("Jared Galvez<super><font size=7>*</font></super>", AUTH))
    S.append(p("Arizona State University — B.S. Data Science (Junior)", AFF))
    S.append(p("UNDERGRADUATE WORKING PAPER  ·  JUNE 2026", TAG))

    # gold rule
    rule = Table([[""]], colWidths=[doc.width * 0.5], rowHeights=[2])
    rule.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), GOLD),
                              ("LINEBELOW", (0, 0), (-1, -1), 0, GOLD)]))
    rule.hAlign = "CENTER"
    S.append(Spacer(1, 6)); S.append(rule); S.append(Spacer(1, 6))

    # ---- abstract ----
    S.append(p("ABSTRACT", ABSH))
    S.append(p(
        "I ask whether the standard published price signals still pay, in the "
        "markets a free-data researcher can reach, once every removable form of "
        "backtest inflation is removed. Across nine pre-registered trials the "
        "answer is largely no, and the manner of the failure is the "
        "contribution. On a survivorship-biased universe the pipeline yields a "
        "net Sharpe of 0.82; on a point-in-time universe the identical code "
        "yields -0.01 — the apparent alpha was the bias. Seven equity trials, "
        "spanning horizons, labels, neutralization, and three model classes, "
        "never clear a Deflated Sharpe of 0.04. Extending the same machinery to "
        "crypto-perpetual funding carry recovers a real, strongly significant "
        "premium (t = -3.5, net Sharpe 0.87) that nonetheless fails the "
        "pre-registered evidence bar (Deflated Sharpe 0.865), is severely "
        "left-skewed, and has decayed from a Sharpe of 2.3 to roughly 0.4 as the "
        "trade institutionalized. A ninth trial shows the post-effective S&P 500 "
        "deletion rebound is matched small-loser mean reversion, not an index "
        "effect. The deliverable is a research process that detects a genuine "
        "premium where one exists and declines to claim it where one does not.",
        ABS))
    S.append(p("<b>Keywords:</b> market efficiency; cross-sectional anomalies; "
               "survivorship bias; deflated Sharpe ratio; funding-rate carry; "
               "research integrity. &nbsp;&nbsp; <b>JEL:</b> C58, G11, G12, G14.",
               META))
    S.append(Spacer(1, 6))

    # ---- 1 ----
    S.append(p("1.&nbsp;&nbsp;Introduction", H))
    S.append(p(
        "Can the most widely published price-only cross-sectional signals — 12-1 "
        "and 6-1 momentum, short-term reversal, low volatility, distance from the "
        "52-week high — be traded profitably in the S&amp;P 500 today, net of "
        "costs, after one removes every backtest inflation available to a "
        "free-data researcher? The published record predicts not, and I confirm "
        "it. The harder question is whether a null can be demonstrated "
        "convincingly enough to survive adversarial scrutiny, and whether the "
        "same machinery, confronted with a market in which a premium genuinely "
        "exists, both finds it and is refused by its own pre-registered standard."))
    S.append(p(
        "I make three contributions. First, I reproduce McLean and Pontiff (2016) "
        "in-house: correcting a single source of bias — the survivorship of the "
        "universe — collapses a net Sharpe of 0.82 to -0.01 under otherwise "
        "identical code. Second, I show across two asset classes that the "
        "discipline holds when the number is good: crypto funding carry produces "
        "a real signal that I decline to claim because it fails the Deflated "
        "Sharpe bar and is structurally crash-prone. Third, the entire apparatus "
        "is falsification-first — a planted-signal / pure-noise harness gates "
        "every result in continuous integration — and I document the artifacts it "
        "caught during development, several of which would have produced false "
        "positives."))

    # ---- 2 ----
    S.append(p("2.&nbsp;&nbsp;Why the evidentiary bar is set where it is", H))
    S.append(p(
        "Four findings dictate the design, and I treat them as constraints rather "
        "than suggestions. (i) Published anomaly returns fall 26% out-of-sample "
        "and 58% post-publication (McLean and Pontiff, 2016). (ii) At honest "
        "multiple-testing thresholds, most published factors are likely false "
        "discoveries (Harvey, Liu, and Zhu, 2016); I therefore count every trial "
        "and never reset the count, which presently stands at N = 9 and feeds the "
        "Deflated Sharpe Ratio directly. (iii) Anomaly profits concentrate in "
        "high-turnover implementations whose costs exceed their gross returns "
        "(Novy-Marx and Velikov, 2016); no gross-only number is reported and "
        "turnover is a headline statistic. (iv) The expected maximum Sharpe of N "
        "noise strategies grows with N (Bailey and Lopez de Prado, 2014); the "
        "Deflated Sharpe benchmarks each result against that expected maximum at "
        "the true N."))

    # ---- 3 ----
    S.append(p("3.&nbsp;&nbsp;Data and the survivorship exhibit", H))
    S.append(p(
        "Daily adjusted prices are obtained via yfinance over 2009–2026. "
        "Point-in-time S&amp;P 500 membership is reconstructed by walking the "
        "index changes table backward from current constituents: 810 names held a "
        "seat at some point in 2010–2026, of which 661 (81.6%) have retrievable "
        "price history. Features are masked to members-as-of-date before "
        "cross-sectional normalization, so a name absent from the index cannot "
        "shift the statistics the model observes."))
    S.append(p(
        "The static present-day universe is retained as a measured exhibit. Under "
        "identical code, features, and costs, the only change being the universe "
        "definition, the result inverts:", BODY))
    S.append(p("Table 1. The same pipeline on a biased versus an honest universe.", CAP))
    S.append(make_table([
        ["", "Static (biased)", "Point-in-time (honest)"],
        ["Mean rank IC", "0.0333", "0.0052"],
        ["Net Sharpe", "0.82", "-0.01"],
        ["Equal-weight long-only Sharpe", "1.00", "0.81"],
    ], [doc.width * 0.46, doc.width * 0.27, doc.width * 0.27]))
    S.append(Spacer(1, 4))
    S.append(p(
        "Survivorship bias did not inflate the result; it was the result. A "
        "universe of known survivors renders “buy the dip” unfalsifiable, since "
        "every drawdown in the panel is one the firm lived through. I bound the "
        "residual bias rather than impute it: forcing a synthetic -30% final "
        "return onto the 29 names that die mid-window moves the net Sharpe by "
        "+0.006 against a zero-return control, so the missing delisting returns "
        "can neither rescue nor sink the result. Imputation is avoided on "
        "principle — delistings are missing because the firm failed, so any model "
        "fit on survivors re-injects the very bias just removed. The 149 "
        "never-priced names are the unbounded residual and require CRSP-grade data."))

    # ---- 4 ----
    S.append(p("4.&nbsp;&nbsp;Methodology", H))
    S.append(p(
        "<b>Validation.</b> Expanding walk-forward with an embargo of at least the "
        "label horizon: a model trained through date t observes no row whose "
        "21-day forward label overlaps the test window. k-fold cross-validation "
        "is inadmissible on financial panels, since random folds place future "
        "observations in the training set of past ones."))
    S.append(p(
        "<b>Falsification gate.</b> Before any real-data result is read, the "
        "pipeline must recover a synthetic planted signal (Deflated Sharpe 0.992) "
        "and reject a pure-noise panel of identical shape (0.078). Both run in "
        "continuous integration on every commit; a change that introduces leakage "
        "causes noise mode to “find alpha” and fails the build."))
    S.append(p(
        "<b>Inference.</b> Daily rank ICs of overlapping 21-day labels share 20 "
        "days of information; treating them as independent overstates significance "
        "by roughly the square root of the horizon. On the planted panel a naive t "
        "of 7.76 falls to a Newey-West t of 2.00, as theory predicts. All reported "
        "t-statistics are Newey-West with lags equal to the horizon."))
    S.append(p(
        "<b>Construction, costs, and baselines.</b> Decile dollar-neutral "
        "long-short, 10 bps linear cost on turnover, sector-demeaned and projected "
        "to zero ex-ante beta with rolling past-only estimates; realized beta "
        "drift is measured (mean 0.05, 95th percentile 0.23) rather than assumed "
        "zero. Every model must beat equal-weight and a one-line 12-1 momentum "
        "rank on identical dates and costs. On the planted panel the one-line "
        "baseline beats the five-feature ridge, 1.19 to 0.86 — a multi-feature "
        "model is not automatically superior to its strongest ingredient."))

    # ---- 5 ----
    S.append(p("5.&nbsp;&nbsp;Results", H))
    S.append(p("5.1&nbsp;&nbsp;Equity price features", H2))
    S.append(p(
        "No equity configuration produces a defensible edge. The best Deflated "
        "Sharpe is 0.04 and no correctly-signed Newey-West t reaches 2. Because "
        "linear, tree, and shallow-network learners on identical features and "
        "harness are uniformly null — and each recovered the planted signal "
        "beforehand — the null is localized in the features, not the learner.", BODY))
    S.append(p("Table 2. Seven equity trials, point-in-time universe, net of "
               "costs, at the true N for the Deflated Sharpe (DSR).", CAP))
    S.append(make_table([
        ["#", "Specification", "IC", "t<sub>NW</sub>", "Net SR", "DSR", "Turn."],
        ["1", "Static universe (biased exhibit)", "0.033", "—", "+0.82", "0.998", "3.8x"],
        ["2", "Point-in-time universe", "0.005", "+0.54", "-0.01", "0.29", "7.3x"],
        ["3", "+ sector / beta neutralization", "0.005", "—", "-0.38", "0.01", "7.4x"],
        ["4", "Quarterly horizon (turnover attack)", "-0.028", "-1.95", "-0.35", "0.01", "2.4x"],
        ["5", "Residualized label", "+0.023", "+1.91", "-0.77", "~0", "3.5x"],
        ["6", "Gradient boosting", "+0.008", "+0.80", "-0.12", "0.04", "5.4x"],
        ["7", "Shallow neural network", "+0.009", "+1.21", "-0.28", "0.01", "5.8x"],
    ], [0.05 * doc.width, 0.37 * doc.width, 0.115 * doc.width, 0.105 * doc.width,
        0.12 * doc.width, 0.09 * doc.width, 0.10 * doc.width]))
    S.append(Spacer(1, 4))

    S.append(p("5.2&nbsp;&nbsp;Cross-asset extension: funding-rate carry", H2))
    S.append(p(
        "To address the concern that a uniformly null pipeline may simply be "
        "unable to find anything, I extend the identical machinery to perpetual-"
        "futures funding carry — a premium with a structural origin (payment to "
        "the party warehousing crowded leveraged-long demand), on a universe with "
        "no survivorship gap, since delisted contracts retain full terminal "
        "histories. The label is funding-inclusive total return; a price-only "
        "label is shown on synthetic data to reject a true premium."))
    S.append(p("Table 3. Trial 8 — crypto funding carry against pre-registered "
               "criteria (top-30 by dollar volume, weekly, 7 bps per side, N=8).", CAP))
    S.append(make_table([
        ["Criterion", "Value", "Bar", "Pass"],
        ["IC Newey-West t", "-3.54", "<= -2", "Yes"],
        ["Net Sharpe", "+0.87", "> 0", "Yes"],
        ["Deflated Sharpe (N=8)", "0.865", ">= 0.95", "No"],
        ["Survives excluding top 3 names", "0.68", "> 0", "Yes"],
        ["Shuffled-funding control", "0.08", "~ 0", "Yes"],
        ["Skew / max drawdown", "-1.87 / -74%", "—", "(crash-prone)"],
    ], [0.40 * doc.width, 0.22 * doc.width, 0.18 * doc.width, 0.20 * doc.width]))
    S.append(Spacer(1, 4))
    S.append(p(
        "The single failed criterion is the Deflated Sharpe, and it was not "
        "relaxed; “0.865 is close” is precisely the rationalization the protocol "
        "forbids. Diagnostics confirm the signal is genuine rather than an "
        "artifact: delaying entry one to five days decays the Sharpe gracefully "
        "(0.87 to 0.39) rather than collapsing, which excludes a timing leak; only "
        "two of twenty-eight delisted contracts were held at expiry, excluding "
        "delisting optimism; and the shuffled-funding control is flat. The premium "
        "is real but decaying — a Sharpe of 2.28 in 2020–21 falling to roughly 0.4 "
        "thereafter as basis-trade capital scaled — and its -1.87 skew is the "
        "structural cost of selling funding. This is the strongest evidence in the "
        "study: the pipeline detects a real premium where one exists, implying the "
        "seven equity nulls reflect genuine absence rather than incapacity."))

    S.append(p("5.3&nbsp;&nbsp;The disappearing deletion effect", H2))
    S.append(p(
        "A matched-control event study on 75 discretionary S&amp;P 500 deletions "
        "(2010–present) asks whether deleted names out-rebound a basket matched on "
        "size and trailing return over the 60 days after the effective date. They "
        "do not: the daily event-time portfolio has a net Sharpe of -0.04, a "
        "Newey-West t of -0.10, and a Deflated Sharpe of 0.05. Deleted names do "
        "rebound (about +4.8% over 60 days), but a matched control rebounds +2.6% "
        "of that, leaving a +2.2% residual that is insignificant (t = 0.87) and "
        "negative before 2015. The rebound is small-loser mean reversion, not an "
        "index-deletion effect — Greenwood and Sammon's (2025) disappearing index "
        "effect reproduced in the post-effective window, with the control basket "
        "as the discipline that separates a real anomaly from a mechanical one."))

    # ---- 6 ----
    S.append(p("6.&nbsp;&nbsp;What the falsification harness caught", H))
    S.append(p(
        "Several development-stage artifacts would have produced false positives. "
        "A baseline expected to be near 0.9 reading an equal-weight Sharpe of 3.34 "
        "exposed a return-construction bug in which pad-filled delisted names "
        "contributed frozen zero returns and crushed measured volatility. On "
        "signal-free synthetic data, residualized-label momentum carried a "
        "volatility-regime-dependent IC of +0.06 to +0.13 from estimation error "
        "alone — an artifact that masquerades as “momentum works in calm regimes,” "
        "a hypothesis I had registered, and which now requires a paired control "
        "before any run. In three separate settings absolute statistics misled "
        "where paired controls did not, which is now standing practice. A "
        "carry-book weighting bug that accumulated stale positions was caught in "
        "pre-trial review rather than after the fact. Finally, diffing successive "
        "vendor downloads of the same history reveals that “point-in-time” has a "
        "data-values dimension: the vendor re-serves history with float-level "
        "noise and occasional large corrections, so a backtest and a live model "
        "train on materially different versions of the past."))

    # ---- 7 ----
    S.append(p("7.&nbsp;&nbsp;Capacity and execution", H))
    S.append(p(
        "A square-root market-impact sweep on the deployed equity configuration "
        "(spread 10 bps, ADV coverage 87.6%) gives annual cost drag of 1.36% at "
        "$1M, 2.27% at $10M, 5.16% at $100M, and 14.30% at $1B. Because the gross "
        "edge is negative the formal capacity is zero; the informative object is "
        "the drag curve, which states that any strategy at this turnover (3.46x "
        "per year) needs at least 1.4% of true gross annual alpha to exist at $1M "
        "and 5% at $100M. The carry strategy, at 35x annual turnover, is a second "
        "reason it does not graduate beyond a logged result."))

    # ---- 8 ----
    S.append(p("8.&nbsp;&nbsp;Live verification", H))
    S.append(p(
        "The best honest equity configuration is paper-traded against a broker API "
        "— not in expectation of profit, which the evidence excludes, but because "
        "live information coefficient versus backtest information coefficient is "
        "the ultimate out-of-sample test of the pipeline. A daily job rebuilds the "
        "universe, trains only on fully realized labels, writes the full "
        "prediction cross-section to a write-once log before any order exists, and "
        "submits capped, integer-share orders to a paper endpoint. A control arm "
        "shadow-logs the momentum baseline on the same names, so a future "
        "divergence can be attributed to the model rather than the period. The "
        "first interpretable comparison matures roughly 21 trading days after the "
        "first cycle; early P&amp;L is explicitly not interpreted. Two "
        "unbackfillable datasets — daily data-revision fingerprints and a "
        "short-borrow-fee cross-section — accrue in parallel."))

    # ---- 9 ----
    S.append(p("9.&nbsp;&nbsp;Limitations and the path to institutional grade", H))
    S.append(p(
        "Residual survivorship bias remains (149 unpriceable names; missing "
        "delisting returns, bounded at 0.006 Sharpe for priceable failures); "
        "sectors are as-of-today; betas are estimated; headline costs are linear "
        "with impact priced separately; and the evidence spans two asset classes "
        "over roughly fifteen and six years. An institutional-grade version would "
        "require CRSP-quality delisting returns, point-in-time fundamentals and "
        "classifications, borrow availability and fees on the short book, an "
        "impact model calibrated to fills, multi-market replication, and an order "
        "of magnitude more trials under the same logging discipline. Each "
        "limitation is named deliberately; naming them is part of the result."))

    # ---- 10 ----
    S.append(p("10.&nbsp;&nbsp;Conclusion", H))
    S.append(p(
        "I tested nine pre-registered hypotheses across two asset classes and "
        "graduated none to production. Read carelessly that is failure; read "
        "correctly it is the point. A research process is valuable only if it can "
        "distinguish a real premium from a lucky one and a robust premium from a "
        "decaying, skewed one, and then act on the distinction even when the "
        "number in front of it is good. The pipeline destroyed its own best result "
        "upon correcting the universe, recovered genuine crypto carry yet declined "
        "it on a deflated-Sharpe standard it was right to enforce, and showed a "
        "famous anomaly to be a matched-control artifact. The strategies failed; "
        "the judgment did not."))

    # ---- references ----
    S.append(p("References", H))
    for r in [
        "Bailey, D., and M. Lopez de Prado (2014). The Deflated Sharpe Ratio: "
        "Correcting for Selection Bias, Backtest Overfitting, and Non-Normality. "
        "<i>Journal of Portfolio Management</i> 40(5), 94–107.",
        "Greenwood, R., and M. Sammon (2025). The Disappearing Index Effect. "
        "<i>Journal of Finance</i>, forthcoming.",
        "Gu, S., B. Kelly, and D. Xiu (2020). Empirical Asset Pricing via Machine "
        "Learning. <i>Review of Financial Studies</i> 33(5), 2223–2273.",
        "Harvey, C., Y. Liu, and H. Zhu (2016). ... and the Cross-Section of "
        "Expected Returns. <i>Review of Financial Studies</i> 29(1), 5–68.",
        "Jegadeesh, N., and S. Titman (1993). Returns to Buying Winners and "
        "Selling Losers. <i>Journal of Finance</i> 48(1), 65–91.",
        "Lopez de Prado, M. (2018). <i>Advances in Financial Machine Learning.</i> "
        "Wiley.",
        "McLean, R. D., and J. Pontiff (2016). Does Academic Research Destroy "
        "Stock Return Predictability? <i>Journal of Finance</i> 71(1), 5–32.",
        "Novy-Marx, R., and M. Velikov (2016). A Taxonomy of Anomalies and Their "
        "Trading Costs. <i>Review of Financial Studies</i> 29(1), 104–147.",
        "Shumway, T. (1997). The Delisting Bias in CRSP Data. <i>Journal of "
        "Finance</i> 52(1), 327–340.",
    ]:
        S.append(p(r, REF))

    S.append(Spacer(1, 10))
    S.append(p(
        "<super><font size=7>*</font></super> Independent undergraduate research "
        "project. All code, data, the complete trial log, and the live "
        "experiment are public. ASU theming is visual only and implies no "
        "institutional endorsement; this is a working paper and has not been "
        "peer reviewed. Every figure traces to a logged trial or a reproducible "
        "artifact in the project repository.", FOOT))

    doc.build(S)
    print(f"[paper] wrote {OUT}")


if __name__ == "__main__":
    build()
