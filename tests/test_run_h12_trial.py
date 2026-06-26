"""H12 graded run (run_h12_trial._run_trial) — offline known-answer / regression.

No-network stub sources drive the WHOLE registered long-ALL-cluster-vs-EW graded
run, pinning the construction the H12 freeze block sets — the POWERED redesign of
H10 (long ALL eligible cluster names, QUANTILE=1.0, NOT the top decile):

  * the long basket holds ALL k>=2 cluster names each date (basket size ==
    #eligible, NOT int(#eligible*0.1) — the load-bearing difference from H10);
  * the book is dollar-neutral (each date's weights sum to ~0);
  * the routine control arm + entry-lag arms are built;
  * the printed verdict is EXACTLY the 7-gate conjunction (no hidden gate);
  * the structured dict has the expected keys;
  * the frozen constants are pinned (especially QUANTILE == 1.0);
  * a poison-the-future PIT pin: a buy/sell filed AFTER an as-of date must NOT
    change any signal value at or before that date.

The price/universe stub provides universe()/prices()/prices_monthly()/start/end
(strings)/ _cik_for (survivorship-safe=True). The bulk insider stub exposes
``transactions(ciks, kind, with_cik=True)`` returning hand-built panels (with an
issuer_cik column == ticker in this offline world) carrying a PLANTED
value-orthogonal opportunistic-cluster alpha. NO socket opens:
fetch_sp500_tables / sector_map are monkeypatched.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from quantlab import insider as ins
from quantlab import universe as uni
import run_h12_trial as rht


# --------------------------------------------------------------------------- #
# A small PIT world: enough names + monthly periods to clear the POWER GATE.
# --------------------------------------------------------------------------- #
_START = "2010-01-01"
_END = "2018-12-31"
_MONTHS = pd.date_range(_START, _END, freq="ME")           # ~108 month-ends
# H12 longs ALL eligible (k>=2) names, so even a modest universe clears the frozen
# POWER floor (median basket >= 5): ~40 firms, most getting a >=2-buyer cluster.
_N_FIRMS = 40
_TICKERS = [f"INS{i:03d}" for i in range(_N_FIRMS)]
_SECTORS = {t: f"S{i % 5}" for i, t in enumerate(_TICKERS)}


def _fake_tables():
    current = pd.DataFrame(
        {"ticker": _TICKERS, "sector": [_SECTORS[t] for t in _TICKERS]})
    changes = pd.DataFrame(columns=["date", "added", "removed", "reason"])
    return current, changes


def _build_world(seed=7, planted=True):
    """Build (prices, purchases, sells) with a PLANTED opportunistic-cluster alpha.

    On each month, a handful of firms receive a CLUSTER of >=2 distinct
    OPPORTUNISTIC buyers (filed that month). In the planted world EVERY such
    cluster firm earns a positive forward idiosyncratic return the NEXT month
    (proportional to buyer-count), so the LONG-ALL-CLUSTER book (not just a decile)
    earns it. Routine buyers (same calendar month every year on a fixed home firm)
    also buy but never move returns. Value-orthogonal: returns carry no shared
    value tilt."""
    rng = np.random.default_rng(seed)
    n = _N_FIRMS
    rets = rng.standard_normal((len(_MONTHS), n)) * 0.04        # idio noise

    buy_rows, sell_rows = [], []
    # ROUTINE insiders: fixed home firm + fixed month every year (>=3 yr -> routine)
    for k in range(6):
        home = _TICKERS[int(rng.integers(0, n))]
        month = int(rng.integers(1, 13))
        owner = f"ROUTINE_{k:02d}"
        for y in range(2010, 2019):
            d = pd.Timestamp(year=y, month=month, day=15)
            if _MONTHS[0] <= d <= _MONTHS[-1]:
                buy_rows.append({"filed_date": d, "owner_name": owner,
                                 "role": "officer", "shares": 1000.0,
                                 "value": np.nan, "transaction_date": d,
                                 "ticker": home, "accession": f"r{k}{y}"})

    # OPPORTUNISTIC clusters: about HALF the firms get a >=2-distinct-buyer cluster
    # each month. This keeps the long-ALL-cluster basket large enough to clear the
    # POWER floor (median basket >= 5) while leaving it a STRICT SUBSET of the
    # priceable universe — the long-vs-EW book is degenerate (all weights cancel)
    # only when eligible == priceable, which the real world (median ~24 of ~500)
    # never hits but a too-dense synthetic world would.
    opp_pool = [f"OPP_{j:05d}" for j in range(n * 12)]
    for t in range(len(_MONTHS) - 1):
        chosen = rng.choice(n, size=int(n * 0.5), replace=False)   # ~20 of 40
        for fi in chosen:
            fi = int(fi)
            firm = _TICKERS[fi]
            n_buyers = int(rng.integers(2, 5))
            buyers = rng.choice(opp_pool, size=n_buyers, replace=False)
            d = _MONTHS[t]
            for b in buyers:
                fd = d - pd.Timedelta(days=int(rng.integers(0, 20)))
                buy_rows.append({"filed_date": fd, "owner_name": str(b),
                                 "role": "director", "shares": 5000.0,
                                 "value": np.nan, "transaction_date": fd,
                                 "ticker": firm, "accession": f"o{t}{fi}{b}"})
            if planted:
                rets[t + 1, fi] += 0.01 * n_buyers

    # A few opportunistic SELLS (so net = buyers - sellers is exercised), on firms
    # that did NOT get a buy cluster, return-neutral.
    for t in range(0, len(_MONTHS) - 1, 7):
        fi = int(rng.integers(0, n))
        firm = _TICKERS[fi]
        d = _MONTHS[t]
        for j in range(2):
            sell_rows.append({"filed_date": d, "owner_name": f"SELLER_{t}_{j}",
                              "role": "officer", "shares": 3000.0,
                              "value": np.nan, "transaction_date": d,
                              "ticker": firm, "accession": f"s{t}{j}"})

    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)),
                          index=_MONTHS, columns=_TICKERS)
    cols = ["owner_name", "role", "shares", "value", "transaction_date",
            "ticker", "accession"]
    purchases = pd.DataFrame(buy_rows).set_index("filed_date").sort_index()
    purchases.index.name = "filed_date"
    sells = (pd.DataFrame(sell_rows).set_index("filed_date").sort_index()
             if sell_rows else pd.DataFrame(
                 columns=cols, index=pd.DatetimeIndex([], name="filed_date")))
    sells.index.name = "filed_date"
    return prices, purchases, sells


class _StubSource:
    """Survivorship-safe price/universe stub. _cik_for is identity (each ticker IS
    its own CIK in this offline world). start/end are STRINGS (as the real source)."""
    survivorship_safe = True
    start = _START
    end = _END

    def __init__(self, prices):
        self._prices = prices

    def _cik_for(self, ticker):
        return ticker            # identity resolution offline

    def universe(self):
        return list(_TICKERS)

    def prices(self, universe, asof):
        cols = [c for c in self._prices.columns if c in set(universe)]
        return self._prices[cols].reindex(asof, method="ffill")

    def prices_monthly(self, universe):
        cols = [c for c in self._prices.columns if c in set(universe)]
        return self._prices[cols]


class _StubBulkSource:
    """Hand-built BULK Form 4 source mirroring BulkInsiderSource.transactions:
    ``transactions(ciks, kind, with_cik=True)`` returns the CIK-sliced panel with
    an ``issuer_cik`` column (CIK == ticker in this offline world)."""

    def __init__(self, purchases, sells):
        self._buys = purchases
        self._sells = sells

    def transactions(self, ciks, kind="P", all_owners=False,
                     include_amendments=False, with_cik=False):
        src = self._buys if kind == "P" else self._sells
        cik_set = set(ciks)
        out = src[src["ticker"].isin(cik_set)].copy()
        if with_cik:
            # in this offline world the issuer_cik IS the ticker (identity _cik_for).
            out["issuer_cik"] = out["ticker"].values
        out.index.name = "filed_date"
        return out


@pytest.fixture
def patched(monkeypatch):
    """Patch the (network) universe-table fetch and return fresh price + bulk
    insider stubs (used by the lighter, self-contained tests)."""
    monkeypatch.setattr(uni, "fetch_sp500_tables", lambda *a, **k: _fake_tables())
    prices, purchases, sells = _build_world(seed=7, planted=True)
    return _StubSource(prices), _StubBulkSource(purchases, sells)


# The graded run is the heaviest path, so run the WHOLE trial ONCE (module-scoped)
# and share the result across assertions.
_GRADED = {}


@pytest.fixture(scope="module")
def graded():
    if not _GRADED:
        prices, purchases, sells = _build_world(seed=7, planted=True)
        src = _StubSource(prices)
        bulk = _StubBulkSource(purchases, sells)
        mp = pytest.MonkeyPatch()
        mp.setattr(uni, "fetch_sp500_tables", lambda *a, **k: _fake_tables())
        out = rht._run_trial(src, bulk, n_trials=13)
        mp.undo()
        _GRADED.update({"src": src, "bulk": bulk, "out": out})
    return _GRADED


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_registered_constants_pinned():
    """Pin the frozen knobs so a silent edit to the construction trips a test.
    The load-bearing one: QUANTILE == 1.0 (long ALL eligible, NOT a decile)."""
    assert rht.QUANTILE == 1.0
    assert rht.WINDOW_DAYS == 90
    assert rht.CLUSTER_K == 2
    assert rht.COST_BPS_PER_SIDE == 10.0
    assert rht.PERIODS_PER_YEAR == 12
    assert rht.SURVIVORSHIP_DOWN == -0.30
    assert rht.REBALANCE_FREQ == "ME"
    assert rht.N_TRIALS_DEFAULT == 13
    assert rht.MIN_N_OBS == 60
    assert rht.MIN_BASKET == 5


def test_graded_run_end_to_end_offline(graded):
    out = graded["out"]
    for key in ("opp", "routine", "opp_lag", "routine_lag", "sr_ew", "sr_mom",
                "shuffle_sr", "bounded_sr", "pbo", "pbo_blocked", "mde_ann",
                "n_obs", "median_basket", "gates", "graduate", "universe",
                "cik_by_member", "member_by_cik"):
        assert key in out, key
    assert set(out["universe"]) == set(_TICKERS)
    # the POWER GATE was cleared (n_obs >= 60 non-empty baskets, median >= 5).
    assert out["n_obs_basket"] >= rht.MIN_N_OBS
    assert out["median_basket"] >= rht.MIN_BASKET


def test_long_basket_holds_ALL_eligible_not_a_decile(graded):
    """THE H12 DISTINGUISHER: the long leg holds EVERY k>=2 cluster name each date
    (basket size == #eligible), NOT the top decile int(#eligible*0.1) that H10
    would take. This is what QUANTILE=1.0 means and is the whole redesign."""
    out, src, bulk = graded["out"], graded["src"], graded["bulk"]
    weights = out["opp"]["weights"]
    members = src.universe()
    sectors = uni.sector_map(_fake_tables()[0], members)
    asof = pd.date_range(src.start, src.end, freq=rht.REBALANCE_FREQ)
    purchases, sells, _, _ = rht.assemble_bulk_panels(src, bulk, members)
    prices = src.prices(members, asof)
    _, mask = ins.net_cluster_buy_signal(
        purchases, sells, asof, tickers=list(prices.columns),
        window_days=rht.WINDOW_DAYS, sector_map=sectors, classify="opportunistic")
    mask = mask.reindex(index=weights.index, columns=weights.columns, fill_value=0.0)

    longs = weights > 0
    # (a) every LONG cell must have buyer-count >= 2 (the cluster gate holds).
    assert ((~longs) | (mask >= 2)).to_numpy().all()

    # (b) THE distinguishing assertion: on dates that take a position, the number
    # of long names equals the number of PRICEABLE k>=2-eligible names (long ALL),
    # which is STRICTLY MORE than a decile would take whenever >= ~10 are eligible.
    px = prices.reindex(weights.index)
    n_checked, n_strictly_more_than_decile = 0, 0
    for d in weights.index:
        priceable = px.loc[d].dropna().index
        eligible = [c for c in priceable if mask.loc[d, c] >= 2]
        n_long = int(longs.loc[d].sum())
        if n_long == 0:
            continue                       # no-position date (fewer than 2 eligible)
        n_checked += 1
        # long ALL eligible: basket size == #eligible, NOT int(#eligible*0.1).
        assert n_long == len(eligible), (d, n_long, len(eligible))
        decile = max(1, int(len(eligible) * 0.10))
        if n_long > decile:
            n_strictly_more_than_decile += 1
    assert n_checked > 0
    # On most dates the full-basket book is strictly broader than a decile book —
    # proves we are NOT silently behaving like H10's top-decile selection.
    assert n_strictly_more_than_decile > 0


def test_book_is_dollar_neutral(graded):
    """Each rebalance's weights sum to ~0 (long $0.5 vs short $0.5 EW)."""
    weights = graded["out"]["opp"]["weights"]
    row_sums = weights.sum(axis=1)
    assert np.allclose(row_sums.to_numpy(), 0.0, atol=1e-9)
    assert (weights.abs().sum(axis=1) > 0).any()


def test_routine_and_lag_arms_built(graded):
    """The routine control arm and both entry-lag arms exist with finite SRs."""
    out = graded["out"]
    assert np.isfinite(out["routine"]["sharpe"])
    assert np.isfinite(out["opp_lag"]["sharpe"])
    assert np.isfinite(out["routine_lag"]["sharpe"])


def test_verdict_is_the_seven_gate_conjunction(graded):
    """The reported graduate flag is EXACTLY the 7-gate conjunction — no hidden
    gate, no missing gate."""
    out = graded["out"]
    g = out["gates"]
    expected = (g["t_nw"] and g["sr_pos"] and g["beats_baselines"] and g["dsr"]
                and g["pbo"] and g["entry_lag"] and g["routine_diff"]
                and g["survivorship"])
    assert out["graduate"] == bool(expected)
    for k in ("t_nw", "sr_pos", "beats_baselines", "dsr", "pbo", "entry_lag",
              "routine_diff", "survivorship"):
        assert isinstance(g[k], (bool, np.bool_)), k


def test_cik_to_member_rekey_aligns_with_price_panel(graded):
    """The bulk panel's ticker is re-keyed to the PIT member symbol via issuer_cik
    so the signal cross-section aligns with the price panel (every signal column is
    a universe member)."""
    out, src, bulk = graded["out"], graded["src"], graded["bulk"]
    members = set(src.universe())
    purchases, sells, cik_by_member, member_by_cik = rht.assemble_bulk_panels(
        src, bulk, members=src.universe())
    # every re-keyed ticker is a universe member (no stray issuer symbols).
    assert set(purchases["ticker"]).issubset(members)
    assert set(sells["ticker"]).issubset(members)
    # the issuer_cik column was consumed (dropped) after the re-key.
    assert "issuer_cik" not in purchases.columns
    # offline identity map: each member's CIK inverts back to A member (the same
    # one here, since _cik_for is identity and there are no collisions).
    assert set(member_by_cik.values()) <= members
    for m, c in cik_by_member.items():
        assert member_by_cik[c] == m            # identity round-trip, no collisions


def test_planted_world_beats_null_world():
    """The long-ALL-cluster book on a PLANTED world earns a higher opportunistic SR
    than on the NULL world (validates the registered long-all book shape)."""
    members = list(_TICKERS)
    current = pd.DataFrame(
        {"ticker": _TICKERS, "sector": [_SECTORS[t] for t in _TICKERS]})
    sectors = uni.sector_map(current, members)
    asof = pd.date_range(_START, _END, freq=rht.REBALANCE_FREQ)

    def _opp_sr(planted):
        prices, purchases, sells = _build_world(seed=7, planted=planted)
        sig, mask = ins.net_cluster_buy_signal(
            purchases, sells, asof, tickers=list(prices.columns),
            window_days=rht.WINDOW_DAYS, sector_map=sectors,
            classify="opportunistic")
        arm = rht.run_arm(sig, mask, prices, 13, "OPP")
        return arm["sharpe"]

    sr_planted = _opp_sr(True)
    sr_null = _opp_sr(False)
    assert sr_planted > sr_null


def test_poison_the_future_pit(patched):
    """A buy/sell filed AFTER an as-of date must NOT change any signal value at or
    before that date (PIT pin, law #1)."""
    src, bulk = patched
    members = src.universe()
    current, _ = uni.fetch_sp500_tables()
    sectors = uni.sector_map(current, members)
    asof = pd.date_range(src.start, src.end, freq=rht.REBALANCE_FREQ)
    purchases, sells, _, _ = rht.assemble_bulk_panels(src, bulk, members)
    prices = src.prices(members, asof)

    sig_clean, mask_clean = ins.net_cluster_buy_signal(
        purchases, sells, asof, tickers=list(prices.columns),
        window_days=rht.WINDOW_DAYS, sector_map=sectors, classify="opportunistic")

    cutoff = asof[len(asof) // 2]
    poison_date = cutoff + pd.Timedelta(days=400)
    pbuy = pd.DataFrame([{
        "owner_name": "FUTURE_BUYER", "role": "director", "shares": 9999.0,
        "value": np.nan, "transaction_date": poison_date, "ticker": _TICKERS[0],
        "accession": "poisonb"}],
        index=pd.DatetimeIndex([poison_date], name="filed_date"))
    psell = pd.DataFrame([{
        "owner_name": "FUTURE_SELLER", "role": "officer", "shares": 9999.0,
        "value": np.nan, "transaction_date": poison_date, "ticker": _TICKERS[1],
        "accession": "poisons"}],
        index=pd.DatetimeIndex([poison_date], name="filed_date"))
    purchases_p = pd.concat([purchases, pbuy]).sort_index()
    sells_p = pd.concat([sells, psell]).sort_index()
    sig_poison, mask_poison = ins.net_cluster_buy_signal(
        purchases_p, sells_p, asof, tickers=list(prices.columns),
        window_days=rht.WINDOW_DAYS, sector_map=sectors, classify="opportunistic")

    upto = asof[asof <= cutoff]
    pd.testing.assert_frame_equal(sig_clean.loc[upto], sig_poison.loc[upto])
    pd.testing.assert_frame_equal(mask_clean.loc[upto], mask_poison.loc[upto])


def test_source_not_wired_refusal():
    """A survivorship-safe source missing the richer interface triggers the
    SOURCE-NOT-WIRED refusal (sys.exit), spending no trial."""
    class _Bare:
        survivorship_safe = True
    with pytest.raises(SystemExit) as exc:
        rht._run_trial(_Bare(), None, n_trials=13)
    assert "SOURCE NOT WIRED" in str(exc.value)


def test_power_gate_aborts_when_thin(patched, monkeypatch):
    """An underpowered world (too few non-empty baskets) must ABORT via the POWER
    GATE, spending no trial."""
    src, bulk = patched
    prices, purchases, sells = _build_world(seed=7, planted=True)
    keep = _TICKERS[:3]                 # 3 firms -> basket can never reach median>=5
    prices = prices[keep]
    purchases = purchases[purchases["ticker"].isin(keep)]
    sells = sells[sells["ticker"].isin(keep)]
    src2 = _StubSource(prices)
    monkeypatch.setattr(src2, "universe", lambda: keep)
    bulk2 = _StubBulkSource(purchases, sells)
    with pytest.raises(SystemExit) as exc:
        rht._run_trial(src2, bulk2, n_trials=13)
    assert "POWER GATE" in str(exc.value)
