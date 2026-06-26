"""H10 insider cluster-buy harness — offline known-answer tests. No network.

Covers: the pure Form 4 XML parser (P vs S, role flags, shares/price/date),
routine-vs-opportunistic classification on hand-built histories, the cross-
sectional cluster-buy signal (distinct-opportunistic-buyer counting AND a
poison-the-future PIT assertion), and the synthetic machinery gate.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import insider as ins
from quantlab import insider_data as idat


# A hand-written ownershipDocument with one PURCHASE (P/A), one SALE (S/D), and a
# transaction missing the optional price field (an award-like row). Officer +
# director flags set, 10% owner unset.
_SAMPLE_FORM4 = """<?xml version="1.0"?>
<ownershipDocument>
  <issuer>
    <issuerCik>0000320193</issuerCik>
    <issuerTradingSymbol>AAPL</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>DOE JANE</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector>
      <isOfficer>1</isOfficer>
      <isTenPercentOwner>0</isTenPercentOwner>
      <officerTitle>CFO</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2023-03-10</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>150.5</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2023-04-01</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>200</value></transactionShares>
        <transactionPricePerShare><value>160.0</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2023-05-01</value></transactionDate>
      <transactionCoding><transactionCode>A</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>50</value></transactionShares>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""


# --- pure Form 4 parser ---------------------------------------------------- #

def test_parse_form4_xml_known_answer():
    rows = idat.parse_form4_xml(_SAMPLE_FORM4)
    assert len(rows) == 3
    buy, sale, award = rows
    # issuer + owner identity carried on every row
    assert buy["issuer_cik"] == "0000320193"
    assert buy["ticker"] == "AAPL"
    assert buy["owner_name"] == "DOE JANE"
    # role flags
    assert buy["is_officer"] is True
    assert buy["is_director"] is True
    assert buy["is_tenpct"] is False
    # the PURCHASE: code P, acquired A, shares/price/date parsed numerically
    assert buy["code"] == "P"
    assert buy["acq_disp"] == "A"
    assert buy["shares"] == 1000.0
    assert buy["price"] == 150.5
    assert buy["transaction_date"] == "2023-03-10"
    # the SALE: code S, disposed D
    assert sale["code"] == "S"
    assert sale["acq_disp"] == "D"
    assert sale["shares"] == 200.0
    # the award row is missing price -> None, never crashes
    assert award["code"] == "A"
    assert award["price"] is None
    assert award["shares"] == 50.0


def test_parse_form4_xml_robust_to_garbage():
    assert idat.parse_form4_xml("not xml at all") == []
    assert idat.parse_form4_xml("<ownershipDocument></ownershipDocument>") == []


def test_is_ownership_xml_distinguishes_raw_from_html():
    assert idat.is_ownership_xml(_SAMPLE_FORM4) is True
    # an XSLT/HTML rendering is not the raw ownership doc
    assert idat.is_ownership_xml("<html><body>rendered</body></html>") is False


def test_parse_form4_accessions_filters_form4_only():
    submissions = {"filings": {"recent": {
        "form": ["10-K", "4", "8-K", "4"],
        "accessionNumber": ["a-10k", "a-4-old", "a-8k", "a-4-new"],
        "filingDate": ["2022-02-01", "2021-06-01", "2022-03-01", "2023-01-01"],
    }}}
    df = idat.parse_form4_accessions(submissions)
    assert list(df["accession"]) == ["a-4-old", "a-4-new"]   # sorted by filed
    assert df["filed"].is_monotonic_increasing


# --- routine vs opportunistic --------------------------------------------- #

def test_routine_same_month_three_consecutive_years():
    # bought every March for 3 consecutive years -> ROUTINE
    dates = ["2018-03-12", "2019-03-15", "2020-03-09"]
    assert ins.classify_routine_opportunistic(dates) == "routine"


def test_opportunistic_sporadic_months():
    # scattered months, no same-month 3-year run -> OPPORTUNISTIC
    dates = ["2018-03-12", "2019-07-15", "2021-11-09", "2022-01-20"]
    assert ins.classify_routine_opportunistic(dates) == "opportunistic"


def test_routine_requires_consecutive_not_just_three_marches():
    # three Marches but a gap year (2018, 2019, 2021) -> NOT 3 consecutive
    dates = ["2018-03-12", "2019-03-15", "2021-03-09"]
    assert ins.classify_routine_opportunistic(dates) == "opportunistic"


def test_classification_is_past_only():
    # As of 2020-01-01 the owner has only TWO prior same-month years -> not yet
    # routine; the 2020 buy (which would complete the run) is in the future and
    # must NOT count (law #1).
    dates = ["2018-03-12", "2019-03-15", "2020-03-09"]
    assert ins.classify_routine_opportunistic(
        dates, asof="2020-01-01") == "opportunistic"
    # As of after the third buy it IS routine.
    assert ins.classify_routine_opportunistic(
        dates, asof="2020-04-01") == "routine"


# --- cluster-buy signal ---------------------------------------------------- #

def _purchases(rows):
    """Build a filed-date-indexed purchases panel from (filed, owner, ticker) and
    optional transaction_date tuples."""
    recs = []
    for r in rows:
        filed, owner, tkr = r[0], r[1], r[2]
        txn = r[3] if len(r) > 3 else filed
        recs.append({"filed_date": pd.Timestamp(filed), "owner_name": owner,
                     "ticker": tkr, "transaction_date": pd.Timestamp(txn),
                     "shares": 1000.0, "value": np.nan, "role": "officer"})
    return pd.DataFrame(recs).set_index("filed_date").sort_index()


def test_cluster_signal_counts_distinct_opportunistic_buyers_in_window():
    # AAA gets 3 distinct opportunistic buyers in the window; BBB gets 1.
    rows = [
        ("2023-01-05", "A1", "AAA"), ("2023-01-06", "A2", "AAA"),
        ("2023-01-07", "A3", "AAA"),
        ("2023-01-05", "B1", "BBB"),
    ]
    panel = _purchases(rows)
    asof = pd.DatetimeIndex(["2023-01-31"])
    sig = ins.cluster_buy_signal(panel, asof, tickers=["AAA", "BBB"],
                                 window_days=90)
    # AAA (count 3) scores strictly higher than BBB (count 1) cross-sectionally.
    assert sig.loc["2023-01-31", "AAA"] > sig.loc["2023-01-31", "BBB"]


def test_cluster_signal_pit_excludes_future_filed_buys():
    # Two clean opportunistic buyers on AAA filed BEFORE t; a poison buy on AAA is
    # filed AFTER t. The score at t must be identical with and without the poison.
    base = [
        ("2023-01-05", "A1", "AAA"), ("2023-01-06", "A2", "AAA"),
        ("2023-01-05", "B1", "BBB"),
    ]
    t = pd.DatetimeIndex(["2023-01-15"])
    sig_clean = ins.cluster_buy_signal(_purchases(base), t,
                                       tickers=["AAA", "BBB"], window_days=90)
    poisoned = base + [("2023-02-20", "A9", "AAA")]   # filed AFTER t
    sig_poison = ins.cluster_buy_signal(_purchases(poisoned), t,
                                        tickers=["AAA", "BBB"], window_days=90)
    pd.testing.assert_frame_equal(sig_clean, sig_poison)


def test_cluster_signal_window_drops_stale_buys():
    # A buy outside the trailing window must not contribute at t.
    rows = [("2023-01-05", "A1", "AAA"), ("2022-01-05", "A2", "AAA"),
            ("2023-01-05", "B1", "BBB"), ("2023-01-06", "B2", "BBB")]
    panel = _purchases(rows)
    asof = pd.DatetimeIndex(["2023-02-01"])
    sig = ins.cluster_buy_signal(panel, asof, tickers=["AAA", "BBB"],
                                 window_days=90)
    # In-window distinct counts: AAA=1 (the 2022 buy is >90d stale), BBB=2.
    assert sig.loc["2023-02-01", "BBB"] > sig.loc["2023-02-01", "AAA"]


def test_cluster_signal_routine_buyers_excluded():
    # A routine buyer (same March, 3 consecutive years) should NOT count toward
    # the opportunistic cluster score. R1 buys AAA every March; as of 2021-03-31
    # R1 is routine (3 prior-or-current same-month years), while AAA's single
    # opportunistic buyer O1 is the only one that should count.
    rows = [
        ("2018-03-10", "R1", "AAA"), ("2019-03-10", "R1", "AAA"),
        ("2020-03-10", "R1", "AAA"), ("2021-03-10", "R1", "AAA"),
        ("2021-03-12", "O1", "AAA"),                       # opportunistic
        ("2021-03-11", "B1", "BBB"), ("2021-03-12", "B2", "BBB"),  # two opportunistic
    ]
    panel = _purchases(rows)
    asof = pd.DatetimeIndex(["2021-03-31"])
    sig = ins.cluster_buy_signal(panel, asof, tickers=["AAA", "BBB"],
                                 window_days=90)
    # BBB has 2 distinct opportunistic buyers; AAA has only 1 (R1 filtered as
    # routine), so BBB outscores AAA despite AAA having more total buyers.
    assert sig.loc["2021-03-31", "BBB"] > sig.loc["2021-03-31", "AAA"]


# --- machinery gate -------------------------------------------------------- #

def test_machinery_gate_planted_beats_null():
    gate = ins.machinery_gate(seeds=(7, 11, 23))
    # The core known answer: planted beats null, paired per seed, by a wide margin.
    assert gate["passed"], gate["diffs"]
    assert min(gate["diffs"]) > 1.0                       # decisive separation
    assert min(gate["planted_sr"]) > 2.0                  # planted clearly works
    # The null is signal-free; its long-short Sharpe over ~120 monthly periods is
    # luck-level (|SR| < 1), never the planted scale. (A larger sample would shrink
    # this further; 120 periods keeps the gate fast.)
    assert max(abs(s) for s in gate["null_sr"]) < 1.0


# --- H10 STAGE-2: sells() filtering, net signal, long-vs-EW book, label_routine #

# A hand-written ownershipDocument with one BUY (P/A) and two SELLS (S/D) so the
# sells() filter has a known answer (S/D only; the P/A buy must be excluded).
_SAMPLE_FORM4_SELLS = """<?xml version="1.0"?>
<ownershipDocument>
  <issuer>
    <issuerCik>0000320193</issuerCik>
    <issuerTradingSymbol>AAPL</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>SELL SAM</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship>
      <isOfficer>1</isOfficer>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2023-03-10</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>150.0</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2023-04-01</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>200</value></transactionShares>
        <transactionPricePerShare><value>160.0</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2023-05-02</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>300</value></transactionShares>
        <transactionPricePerShare><value>170.0</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""


class _StubInsiderSource(idat.InsiderSource):
    """InsiderSource whose accession enumeration + XML fetch are stubbed to a
    single hand-built Form 4, so sells()/purchases() run their real filters with no
    network. ticker_cik is identity (the 'ticker' IS the cik)."""

    def __init__(self, xml):
        self._xml = xml

    def ticker_cik(self, ticker):
        return "320193"

    def list_form4_accessions(self, cik):
        return pd.DataFrame({"accession": ["acc-1"],
                             "filed": [pd.Timestamp("2023-05-05")]})

    def fetch_form4_raw(self, cik, accession):
        return self._xml


def test_sells_filters_S_D_only():
    src = _StubInsiderSource(_SAMPLE_FORM4_SELLS)
    sells = src.sells("AAPL")
    # exactly the two S/D disposals (the P/A buy is excluded).
    assert len(sells) == 2
    assert set(sells["shares"]) == {200.0, 300.0}
    # symmetric columns + filing-date index, identical to purchases().
    buys = src.purchases("AAPL")
    assert list(sells.columns) == list(buys.columns)
    assert sells.index.name == "filed_date"
    assert len(buys) == 1 and buys.iloc[0]["shares"] == 1000.0


def test_fast_labeler_matches_pinned_classifier():
    """``_label_opportunistic_fast`` (the linear-time path the net signal uses) must
    be byte-identical to the pinned ``label_opportunistic`` — both known-answer and
    randomized panels. This is the parity contract that lets the harness use the
    fast path without changing the registered classification."""
    cases = [
        [("2018-03-10", "R1", "AAA"), ("2019-03-10", "R1", "AAA"),
         ("2020-03-10", "R1", "AAA"), ("2021-03-10", "R1", "AAA"),
         ("2021-03-12", "O1", "AAA"), ("2021-03-11", "B1", "BBB"),
         ("2021-03-12", "B2", "BBB")],
        [("2018-03-12", "R1", "AAA"), ("2019-03-15", "R1", "AAA"),
         ("2021-03-09", "R1", "AAA")],                       # gap year -> opp
    ]
    for c in cases:
        p = _purchases(c)
        np.testing.assert_array_equal(
            ins.label_opportunistic(p).to_numpy(), ins._label_opportunistic_fast(p))
    # randomized: many owners/tickers/months.
    rng = np.random.default_rng(1)
    owners = [f"OW{i}" for i in range(15)]
    tks = ["AAA", "BBB", "CCC"]
    rows = []
    for _ in range(300):
        y = int(rng.integers(2010, 2020))
        m = int(rng.integers(1, 13))
        rows.append((f"{y}-{m:02d}-15", str(rng.choice(owners)), str(rng.choice(tks))))
    p = _purchases(rows)
    np.testing.assert_array_equal(
        ins.label_opportunistic(p).to_numpy(), ins._label_opportunistic_fast(p))


def test_fast_labeler_same_filed_date_block_parity():
    """Adversarial regression (2026-06-25 review): rows of the SAME (owner, ticker)
    sharing ONE filed_date — exactly what a single multi-transaction Form 4 emits —
    must NOT count toward each other. The pinned classifier uses strict
    ``filed_date < t``, so same-day siblings are mutually invisible; an earlier fast
    path committed each sibling before classifying the next and over-branded the last
    one ROUTINE. Pin both the hand-built completing-run case and a collision-heavy
    fuzz."""
    # Four buys, ONE filing date, transaction months in 4 consecutive Marches: the
    # pinned classifier sees NO strictly-prior history for any of them -> all four
    # OPPORTUNISTIC. A blind running-history walk would brand the 4th ROUTINE.
    same_day = [
        ("2021-05-05", "R1", "AAA", "2018-03-10"),
        ("2021-05-05", "R1", "AAA", "2019-03-10"),
        ("2021-05-05", "R1", "AAA", "2020-03-10"),
        ("2021-05-05", "R1", "AAA", "2021-03-10"),
    ]
    p = _purchases(same_day)
    pinned = ins.label_opportunistic(p).to_numpy()
    fast = ins._label_opportunistic_fast(p)
    np.testing.assert_array_equal(pinned, fast)
    assert pinned.all()                       # all opportunistic (no prior history)
    # Collision-heavy fuzz: filed dates drawn from a SMALL pool (frequent same-day
    # collisions per owner+ticker) while transaction dates span years/months.
    rng = np.random.default_rng(7)
    owners = [f"OW{i}" for i in range(6)]
    tks = ["AAA", "BBB"]
    filed_pool = [f"20{y:02d}-05-05" for y in range(18, 23)]   # only 5 filed dates
    rows = []
    for _ in range(400):
        ty = int(rng.integers(2010, 2022))
        tm = int(rng.integers(1, 13))
        rows.append((str(rng.choice(filed_pool)), str(rng.choice(owners)),
                     str(rng.choice(tks)), f"{ty}-{tm:02d}-10"))
    p = _purchases(rows)
    np.testing.assert_array_equal(
        ins.label_opportunistic(p).to_numpy(), ins._label_opportunistic_fast(p))


def test_label_routine_is_past_only_complement():
    # R1 buys AAA every March 2018-2021 (routine after 3 consecutive years);
    # O1 buys once. label_routine == ~label_opportunistic, row-aligned.
    rows = [
        ("2018-03-10", "R1", "AAA"), ("2019-03-10", "R1", "AAA"),
        ("2020-03-10", "R1", "AAA"), ("2021-03-10", "R1", "AAA"),
        ("2021-06-01", "O1", "AAA"),
    ]
    panel = _purchases(rows)
    opp = ins.label_opportunistic(panel)
    routine = ins.label_routine(panel)
    pd.testing.assert_series_equal(routine, ~opp, check_names=False)
    # the 4th R1 buy (after 3 prior consecutive same-month years) is ROUTINE;
    # O1's lone buy is never routine.
    df = panel.reset_index()
    r1_last = df.index[(df["owner_name"] == "R1")][-1]
    o1_idx = df.index[df["owner_name"] == "O1"][0]
    assert bool(routine.iloc[r1_last]) is True
    assert bool(routine.iloc[o1_idx]) is False


def test_net_cluster_buy_signal_net_math_and_mask():
    # AAA: 3 distinct opportunistic buyers, 1 opportunistic seller -> net 2, k-mask 3.
    # BBB: 1 buyer, 0 sellers -> net 1, k-mask 1.
    buys = _purchases([
        ("2023-01-05", "A1", "AAA"), ("2023-01-06", "A2", "AAA"),
        ("2023-01-07", "A3", "AAA"), ("2023-01-05", "B1", "BBB"),
    ])
    sells = _purchases([("2023-01-08", "S1", "AAA")])
    asof = pd.DatetimeIndex(["2023-01-31"])
    sig, mask = ins.net_cluster_buy_signal(
        buys, sells, asof, tickers=["AAA", "BBB"], window_days=90)
    # buyer-count mask (sellers never enter the mask).
    assert mask.loc["2023-01-31", "AAA"] == 3.0
    assert mask.loc["2023-01-31", "BBB"] == 1.0
    # net AAA=3-1=2 > net BBB=1-0=1, so AAA's z-score is the larger.
    assert sig.loc["2023-01-31", "AAA"] > sig.loc["2023-01-31", "BBB"]


def test_net_cluster_buy_signal_sector_demean():
    # Two sectors; demeaning within sector changes the ranking vs no-demean.
    # AAA,BBB in S0; CCC,DDD in S1. Counts: AAA=4, BBB=0, CCC=2, DDD=0.
    rows = []
    for i in range(4):
        rows.append((f"2023-01-0{i+1}", f"A{i}", "AAA"))
    for i in range(2):
        rows.append((f"2023-01-0{i+1}", f"C{i}", "CCC"))
    buys = _purchases(rows)
    asof = pd.DatetimeIndex(["2023-01-31"])
    smap = {"AAA": "S0", "BBB": "S0", "CCC": "S1", "DDD": "S1"}
    sig, _ = ins.net_cluster_buy_signal(
        buys, None, asof, tickers=["AAA", "BBB", "CCC", "DDD"],
        window_days=90, sector_map=smap)
    # within S0: AAA(4) demeans to +2, BBB(0) to -2; within S1: CCC(2)->+1, DDD(0)->-1.
    # After cross-sectional z, AAA is the top and BBB the bottom.
    assert sig.loc["2023-01-31", "AAA"] == sig.loc["2023-01-31"].max()
    assert sig.loc["2023-01-31", "BBB"] == sig.loc["2023-01-31"].min()


def test_long_vs_ew_weights_gated_long_ew_short_dollar_neutral():
    # 6 names, all priceable. AAA,BBB pass k>=2; CCC has 1 buyer (gated out);
    # DDD/EEE/FFF have 0. Top-decile long among eligible {AAA,BBB}.
    cols = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    d = pd.Timestamp("2023-01-31")
    signal = pd.DataFrame(
        [[3.0, 2.0, 1.0, 0.0, 0.0, 0.0]], index=[d], columns=cols)
    mask = pd.DataFrame(
        [[3.0, 2.0, 1.0, 0.0, 0.0, 0.0]], index=[d], columns=cols)
    prices = pd.DataFrame([[10.0] * 6], index=[d], columns=cols)
    w = ins.long_vs_ew_weights(signal, mask, prices, quantile=0.10)
    # dollar-neutral: sums to ~0.
    assert abs(float(w.loc[d].sum())) < 1e-12
    # the long tilt only lands on cluster-eligible names (AAA strongest signal).
    longs = w.loc[d][w.loc[d] > w.loc[d].min() + 1e-12]
    assert "AAA" in longs.index
    assert "CCC" not in longs.index            # single buyer, gated out of long
    # short leg = full priceable universe (every name carries the -0.5/6 EW short).
    assert (w.loc[d] < 0).sum() >= 1


def test_long_vs_ew_weights_zero_row_when_under_two_eligible():
    # Only one name passes k>=2 -> no cluster -> all-zero row (no position).
    cols = ["AAA", "BBB", "CCC"]
    d = pd.Timestamp("2023-01-31")
    signal = pd.DataFrame([[3.0, 1.0, 0.0]], index=[d], columns=cols)
    mask = pd.DataFrame([[2.0, 1.0, 0.0]], index=[d], columns=cols)
    prices = pd.DataFrame([[10.0, 10.0, 10.0]], index=[d], columns=cols)
    w = ins.long_vs_ew_weights(signal, mask, prices, quantile=0.10)
    assert (w.loc[d] == 0.0).all()


def test_long_vs_ew_book_planted_beats_null_synthetic():
    """The long-vs-EW book on a PLANTED synthetic world beats the NULL world
    (validates the registered book shape; synthetic has no sells so net=buys)."""
    from quantlab.synthetic import make_insider_panel

    def _book_sr(mode):
        panel = make_insider_panel(mode=mode, seed=7, n_firms=120, n_periods=120)
        purchases = panel.attrs["purchases"]
        asof = panel.index
        sig, mask = ins.net_cluster_buy_signal(
            purchases, None, asof, tickers=list(panel.columns),
            window_days=panel.attrs.get("window_days", 90), classify="opportunistic")
        w = ins.long_vs_ew_weights(sig, mask, panel.reindex(asof), quantile=0.10)
        fwd = panel.pct_change(fill_method=None).shift(-1).reindex_like(w)
        net = (w * fwd).sum(axis=1, min_count=1).dropna()
        return ins.metrics.sharpe(net, periods=ins.PERIODS_PER_YEAR)

    sr_planted = _book_sr("planted_opportunistic")
    sr_null = _book_sr("null_opportunistic")
    assert sr_planted > sr_null
    assert sr_planted > 0.5
