"""H10 opportunistic insider cluster-buying harness — classification, signal, gate.

Hypothesis (Lakonishok-Lee 2001; Cohen-Malloy-Pomorski 2012 "Decoding Inside
Information"): a firm whose own insiders CLUSTER-BUY its stock on the open market
earns positive forward returns — but ONLY for OPPORTUNISTIC insiders. Insiders
who trade the SAME calendar month every year (ROUTINE: liquidity-driven, e.g. an
annual diversification sale or a scheduled purchase) carry NO information; the
signal lives in trades that break that routine.

This module turns a stream of open-market PURCHASES (from
``quantlab.insider_data.InsiderSource.purchases``) into a cross-sectional score:

1. ``classify_routine_opportunistic`` — per owner, ROUTINE if they bought in the
   same calendar month for >= 3 prior CONSECUTIVE years; else OPPORTUNISTIC.
   PAST-ONLY: classification as of date t uses only that owner's purchases BEFORE
   t (law #1 — no lookahead).
2. ``cluster_buy_signal`` — per (date, ticker), count DISTINCT opportunistic
   insiders who bought (filed_date <= t) within a trailing window, cross-
   sectionally z-scored. Clustering (many distinct insiders, not one big trade) is
   the CMP information proxy; counting DISTINCT owners makes it robust to one
   insider splitting a buy across days.
3. ``machinery_gate`` — synthetic falsification (law #2/#4): planted-opportunistic
   world must beat the null world, paired per seed, before any real H10 run.

PIT safety is the whole game here: a Form 4 enters the signal only by its FILING
date (the submissions filingDate, due within 2 business days of the trade), and
the routine/opportunistic label at t uses only prior-year history. Every function
below documents which past-only inputs it touches.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab import metrics

PERIODS_PER_YEAR = 12


def classify_routine_opportunistic(
    owner_purchase_dates: list,
    asof: object | None = None,
    min_consecutive_years: int = 3,
) -> str:
    """Classify one owner as ``"routine"`` or ``"opportunistic"`` per Cohen-Malloy-
    Pomorski (2012).

    ROUTINE: the owner purchased in the SAME calendar month for at least
    ``min_consecutive_years`` (default 3) prior CONSECUTIVE years — a mechanical,
    information-free trading pattern. Otherwise OPPORTUNISTIC.

    PAST-ONLY (law #1): if ``asof`` is given, only purchases STRICTLY BEFORE
    ``asof`` are considered, so the label at date t never peeks at t-or-later
    trades. ``owner_purchase_dates`` is any iterable of date-likes (the owner's
    own open-market purchase dates).

    Logic: for each calendar month m, collect the set of years the owner bought in
    month m. A run of >= ``min_consecutive_years`` consecutive years within ANY
    single month m brands the owner ROUTINE. This matches CMP's "trades in the same
    calendar month in N consecutive years" definition.
    """
    dates = pd.to_datetime(pd.Series(list(owner_purchase_dates))).dropna()
    if asof is not None:
        dates = dates[dates < pd.Timestamp(asof)]
    if dates.empty:
        return "opportunistic"

    # month -> sorted unique years bought in that month
    by_month: dict[int, set[int]] = {}
    for d in dates:
        by_month.setdefault(d.month, set()).add(d.year)

    for years in by_month.values():
        yrs = sorted(years)
        run = 1
        for i in range(1, len(yrs)):
            run = run + 1 if yrs[i] == yrs[i - 1] + 1 else 1
            if run >= min_consecutive_years:
                return "routine"
        if run >= min_consecutive_years:        # single-month list of length >= N
            return "routine"
    return "opportunistic"


def label_opportunistic(
    purchases_panel: pd.DataFrame,
    min_consecutive_years: int = 3,
) -> pd.Series:
    """Per-PURCHASE opportunistic/routine label, PIT.

    ``purchases_panel`` is the long-form purchases table (one row per insider buy,
    indexed by ``filed_date``) with at least ``owner_name``, ``ticker``, and a
    ``transaction_date``. For each row, the owner is classified using ONLY that
    owner's OTHER purchases of the SAME ticker that were FILED STRICTLY BEFORE this
    row's filed_date (past-only — no lookahead, and an owner's routine is firm-
    specific). Returns a boolean Series (True = opportunistic) aligned to the
    panel's row order.

    A buy with no prior history for that owner/ticker is OPPORTUNISTIC by default
    (you cannot yet have established a 3-year routine) — the conservative choice
    that never manufactures signal from absence.
    """
    df = purchases_panel.reset_index()
    filed_col = "filed_date" if "filed_date" in df.columns else df.columns[0]
    out = np.empty(len(df), dtype=bool)
    for pos in range(len(df)):
        row = df.iloc[pos]
        owner, tkr = row.get("owner_name"), row.get("ticker")
        t = pd.Timestamp(row[filed_col])
        prior = df[(df["owner_name"] == owner) & (df.get("ticker") == tkr)
                   & (pd.to_datetime(df[filed_col]) < t)]
        # classify on the owner's prior TRANSACTION dates (the calendar-month
        # pattern is about when they trade, not when the form posted).
        hist = prior["transaction_date"] if "transaction_date" in prior else prior[filed_col]
        klass = classify_routine_opportunistic(
            list(hist), asof=None, min_consecutive_years=min_consecutive_years)
        out[pos] = (klass == "opportunistic")
    return pd.Series(out, index=purchases_panel.index, name="opportunistic")


def cluster_buy_signal(
    purchases_panel: pd.DataFrame,
    asof_dates: pd.DatetimeIndex,
    tickers: list[str] | None = None,
    window_days: int = 90,
    min_consecutive_years: int = 3,
) -> pd.DataFrame:
    """Cross-sectional (date x ticker) opportunistic-cluster-buy score.

    For each as-of date t and ticker, COUNT the DISTINCT OPPORTUNISTIC insiders
    who bought that ticker on the open market with ``filed_date <= t`` and
    ``filed_date > t - window_days`` (a trailing window), then cross-sectionally
    z-score the counts per date. High score = many distinct opportunistic insiders
    recently bought = the CMP cluster-buy signal.

    ECONOMIC LOGIC: a single large buy can be liquidity/diversification noise; a
    CLUSTER of distinct insiders independently buying is the strongest in-sample
    proxy for shared private optimism (CMP 2012). Counting DISTINCT owners (not
    trades or shares) makes the score robust to one insider slicing a buy across
    days and to a few mega-trades dominating.

    PIT SAFETY (law #1), two layers:
      * Only Form 4s FILED on/before t enter the count (``filed_date <= t``) — a
        purchase filed AFTER t is invisible at t. (Pinned by a poison-the-future
        test: injecting a future-filed buy must NOT change any score at t.)
      * The opportunistic/routine label for each buy uses ONLY that owner's prior
        FILED purchases of the same ticker (``label_opportunistic`` is past-only).

    ``purchases_panel`` is the long-form purchases table indexed by ``filed_date``;
    ``tickers`` defaults to the panel's distinct tickers. Returns NaN-free zeros
    for dates/tickers with no qualifying buys (a flat row is z-scored to 0).
    """
    asof_dates = pd.DatetimeIndex(asof_dates)
    if purchases_panel.empty:
        return pd.DataFrame(0.0, index=asof_dates, columns=(tickers or []))

    df = purchases_panel.reset_index()
    filed_col = "filed_date" if "filed_date" in df.columns else df.columns[0]
    df[filed_col] = pd.to_datetime(df[filed_col])
    if tickers is None:
        tickers = sorted(df["ticker"].dropna().unique().tolist())

    # PIT past-only opportunistic flag per buy.
    df["_opp"] = label_opportunistic(
        purchases_panel, min_consecutive_years=min_consecutive_years).to_numpy()
    opp = df[df["_opp"]]

    window = pd.Timedelta(days=window_days)
    counts = pd.DataFrame(0.0, index=asof_dates, columns=tickers)
    for t in asof_dates:
        lo = t - window
        # filed_date <= t (no future) AND within the trailing window.
        win = opp[(opp[filed_col] <= t) & (opp[filed_col] > lo)]
        if win.empty:
            continue
        # DISTINCT opportunistic owners per ticker.
        distinct = win.groupby("ticker")["owner_name"].nunique()
        for tkr, c in distinct.items():
            if tkr in counts.columns:
                counts.loc[t, tkr] = float(c)

    return _zscore_rows(counts)


def _zscore_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-date cross-sectional z-score (demean / std). A flat row -> all zeros."""
    centered = panel.sub(panel.mean(axis=1), axis=0)
    return centered.div(panel.std(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)


def signal_weights(signal: pd.DataFrame, quantile: float = 0.2) -> pd.DataFrame:
    """Dollar-neutral equal-weight quintiles: LONG the highest cluster-buy score,
    SHORT the lowest, per rebalance date (full reset). Mirrors
    ``fundamentals.quality_weights`` so H10 trades the same book shape."""
    target = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    for d in signal.index:
        row = signal.loc[d].replace(0.0, np.nan).dropna()
        n = int(len(row) * quantile)
        if n < 2:
            continue
        longs, shorts = row.nlargest(n).index, row.nsmallest(n).index
        target.loc[d, longs] = 0.5 / n
        target.loc[d, shorts] = -0.5 / n
    return target


def cluster_backtest(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    quantile: float = 0.2,
    cost_bps_per_side: float = 0.0,
) -> dict:
    """Period book: weights at t earn the t->t+1 return. Returns net/gross series
    and annual turnover. Mirrors ``fundamentals.quality_backtest`` (EW quintiles)
    so the synthetic machinery gate is the same machine pointed at insider data."""
    fwd = prices.pct_change(fill_method=None).shift(-1).reindex_like(signal)
    w = signal_weights(signal, quantile=quantile)
    gross = (w * fwd).sum(axis=1, min_count=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    net = (gross - turnover * cost_bps_per_side / 1e4).dropna()
    return {"net": net, "gross": gross.dropna(),
            "annual_turnover": float(turnover.sum() / max(len(w), 1) * PERIODS_PER_YEAR)}


def machinery_gate(
    seeds=(7, 11, 23),
    n_firms: int = 120,
    n_periods: int = 120,
    min_differential: float = 0.5,
) -> dict:
    """Falsification gate (law #2/#4): the planted-opportunistic world must beat
    the null world, paired per seed, before any real H10 run.

    For each seed, build paired synthetic worlds (identical draws except the
    return link), compute the opportunistic-cluster-buy signal from the planted
    Form-4-event panel, run the EW long-short book, and require
    ``Sharpe(planted) - Sharpe(null) > min_differential``. Imports synthetic
    lazily so the data layer carries no synthetic dependency.
    """
    from quantlab.synthetic import make_insider_panel
    diffs, planted, null = [], [], []
    for s in seeds:
        p = make_insider_panel(mode="planted_opportunistic", seed=s,
                               n_firms=n_firms, n_periods=n_periods)
        n = make_insider_panel(mode="null_opportunistic", seed=s,
                               n_firms=n_firms, n_periods=n_periods)
        sr_p = _world_sharpe(p)
        sr_n = _world_sharpe(n)
        diffs.append(sr_p - sr_n)
        planted.append(sr_p)
        null.append(sr_n)
    return {"passed": min(diffs) > min_differential, "diffs": diffs,
            "planted_sr": planted, "null_sr": null}


def _world_sharpe(panel: pd.DataFrame) -> float:
    """Build the cluster-buy signal from a synthetic world's event panel and score
    the EW long-short book's Sharpe. The event panel rides in
    ``attrs["purchases"]`` (long-form, filed-date-indexed)."""
    purchases = panel.attrs["purchases"]
    asof = panel.index
    sig = cluster_buy_signal(purchases, asof, tickers=list(panel.columns),
                             window_days=panel.attrs.get("window_days", 90))
    net = cluster_backtest(sig, panel, cost_bps_per_side=0.0)["net"]
    return metrics.sharpe(net, periods=PERIODS_PER_YEAR)


# =========================================================================== #
# H10 STAGE-2 (frozen 2026-06-25) — the registered net signal + long-vs-EW book.
# These ADD to the module; the pinned functions above are untouched. Every new
# function carries a one-line PIT-safety argument (law #1).
# =========================================================================== #

def _label_opportunistic_fast(
    panel: pd.DataFrame,
    min_consecutive_years: int = 3,
) -> np.ndarray:
    """Vectorized-by-group equivalent of ``label_opportunistic``'s per-row classify,
    returning a boolean array (True = opportunistic) aligned to ``panel`` row order.

    IDENTICAL semantics to the pinned ``classify_routine_opportunistic`` +
    ``label_opportunistic``, computed in ONE pass per (owner, ticker) group instead
    of an O(n^2) full-frame filter per row (the pinned path is correct but quadratic;
    a graded run over years of S&P-500 Form 4s needs the linear path). Verified to
    match the pinned classifier by a known-answer test.

    For each (owner, ticker) group sorted by filed_date, walk rows oldest->newest;
    a row is ROUTINE iff the owner's PRIOR transactions (strictly earlier filed_date,
    same owner+ticker) already contain, in SOME calendar month, a run of
    ``min_consecutive_years`` CONSECUTIVE years. Strictly-prior == past-only (law #1):
    the current row's own date never counts toward its own label.
    """
    df = panel.reset_index()
    filed_col = "filed_date" if "filed_date" in df.columns else df.columns[0]
    filed = pd.to_datetime(df[filed_col])
    txn = (pd.to_datetime(df["transaction_date"]) if "transaction_date" in df.columns
           else filed)
    out = np.ones(len(df), dtype=bool)                       # default: opportunistic
    owners = df["owner_name"].to_numpy()
    tickers = (df["ticker"].to_numpy() if "ticker" in df.columns
               else np.array([None] * len(df)))

    order = np.lexsort((filed.to_numpy(), tickers, owners))  # group, then by filed
    filed_np = filed.to_numpy()
    for start in _group_runs(owners, tickers, order):
        # rows of this (owner,ticker) group in filed-date order.
        idxs = order[start[0]:start[1]]
        # month -> set of prior years seen so far (strictly EARLIER filed_date).
        by_month: dict[int, set[int]] = {}
        # Process the group in FILED-DATE BLOCKS. Rows sharing a filed_date must NOT
        # count toward each other's label: the pinned ``label_opportunistic`` selects
        # prior history with STRICT ``filed_date < t``, so same-day siblings (a single
        # multi-transaction Form 4 emits several rows under one filing date) are
        # mutually invisible. Classify every row in a same-filed-date block against
        # the prior history FIRST, THEN commit the whole block's transaction dates —
        # byte-identical to the pinned per-row classifier (law #1, past-only).
        bi, m = 0, len(idxs)
        while bi < m:
            bj = bi + 1
            fd = filed_np[idxs[bi]]
            while bj < m and filed_np[idxs[bj]] == fd:
                bj += 1
            block = idxs[bi:bj]
            for pos in block:           # classify against PRIOR-only history
                out[pos] = not _routine_from_months(by_month, min_consecutive_years)
            for pos in block:           # then commit this block's transaction dates
                d = txn.iloc[pos]       # (pinned classifier keys on transaction date)
                if pd.notna(d):
                    by_month.setdefault(d.month, set()).add(int(d.year))
            bi = bj
    return out


def _group_runs(owners, tickers, order):
    """Yield (start, stop) row-slices of ``order`` that share an (owner, ticker)."""
    n = len(order)
    i = 0
    while i < n:
        j = i + 1
        oi, ti = owners[order[i]], tickers[order[i]]
        while j < n and owners[order[j]] == oi and tickers[order[j]] == ti:
            j += 1
        yield (i, j)
        i = j


def _routine_from_months(by_month: dict, min_consecutive_years: int) -> bool:
    """True iff some calendar month has a run of >= N CONSECUTIVE years — the exact
    rule of the pinned ``classify_routine_opportunistic`` (empty history -> False)."""
    for years in by_month.values():
        yrs = sorted(years)
        run = 1
        for k in range(1, len(yrs)):
            run = run + 1 if yrs[k] == yrs[k - 1] + 1 else 1
            if run >= min_consecutive_years:
                return True
        if run >= min_consecutive_years:
            return True
    return False


def label_routine(
    purchases_panel: pd.DataFrame,
    min_consecutive_years: int = 3,
) -> pd.Series:
    """Per-PURCHASE ROUTINE label — the past-only complement of
    ``label_opportunistic`` (True = ROUTINE).

    ROUTINE is exactly ``not opportunistic``: an owner who bought the SAME ticker
    in the SAME calendar month for >= ``min_consecutive_years`` prior CONSECUTIVE
    years (Cohen-Malloy-Pomorski). Reuses ``label_opportunistic`` (the pinned
    classifier) and negates it, so the routine/opportunistic split is one source
    of truth — the control arm and the primary arm can never drift apart.

    PIT (law #1): inherits ``label_opportunistic``'s discipline verbatim — each
    row is classified using ONLY that owner's prior FILED purchases of the same
    ticker, so the routine flag at a buy never peeks at that buy or any later one.
    """
    return ~label_opportunistic(
        purchases_panel, min_consecutive_years=min_consecutive_years
    ).rename("routine")


def _label_sells_routine_or_opp(
    sells_panel: pd.DataFrame,
    min_consecutive_years: int,
    want_opportunistic: bool,
) -> np.ndarray:
    """Boolean mask over sell rows: opportunistic (or routine) SELLERS, classified
    on each owner's PRIOR SELL history of the SAME ticker, past-only.

    CMP classify buys on prior BUY history and sells on prior SELL history; the
    net signal subtracts opportunistic sellers, so a seller's routine/opportunistic
    label must come from their own selling calendar, not their buying one. Reuses
    ``classify_routine_opportunistic`` (the pinned per-owner classifier) on prior
    sell transaction dates.

    PIT (law #1): for each sell row, only that owner's OTHER sells of the same
    ticker FILED STRICTLY BEFORE this row's filed_date establish the routine
    pattern — a future-filed sale cannot retro-brand an earlier one routine.
    """
    if sells_panel is None or sells_panel.empty:
        return np.zeros(0, dtype=bool)
    opp = _label_opportunistic_fast(
        sells_panel, min_consecutive_years=min_consecutive_years)
    return opp if want_opportunistic else ~opp


def _distinct_buyer_counts(
    panel: pd.DataFrame,
    asof_dates: pd.DatetimeIndex,
    tickers: list[str],
    window_days: int,
    keep_mask: np.ndarray,
) -> pd.DataFrame:
    """Per date x ticker count of DISTINCT owners (from rows ``keep_mask`` selects)
    filed within the trailing window and on/before t. The shared counting core of
    ``net_cluster_buy_signal`` for both the buy and sell legs."""
    counts = pd.DataFrame(0.0, index=asof_dates, columns=tickers)
    if panel is None or panel.empty or not keep_mask.any():
        return counts
    df = panel.reset_index()
    filed_col = "filed_date" if "filed_date" in df.columns else df.columns[0]
    df[filed_col] = pd.to_datetime(df[filed_col])
    kept = df[keep_mask]
    window = pd.Timedelta(days=window_days)
    for t in asof_dates:
        lo = t - window
        win = kept[(kept[filed_col] <= t) & (kept[filed_col] > lo)]
        if win.empty:
            continue
        distinct = win.groupby("ticker")["owner_name"].nunique()
        for tkr, c in distinct.items():
            if tkr in counts.columns:
                counts.loc[t, tkr] = float(c)
    return counts


def net_cluster_buy_signal(
    purchases_panel: pd.DataFrame,
    sells_panel: pd.DataFrame | None = None,
    asof_dates: pd.DatetimeIndex | None = None,
    tickers: list[str] | None = None,
    window_days: int = 90,
    min_consecutive_years: int = 3,
    sector_map: dict[str, str] | None = None,
    classify: str = "opportunistic",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The H10 frozen NET cluster-buy signal: distinct (opportunistic) BUYERS minus
    distinct (opportunistic) SELLERS in a trailing window, sector-demeaned then
    cross-sectionally z-scored. Returns ``(signal_z, n_buyers_mask)``.

    Per date t x name: ``net = (distinct OPPORTUNISTIC buyers in window) −
    (distinct OPPORTUNISTIC sellers in window)``, counting owners whose buy filed
    with ``filed_date <= t`` and ``filed_date > t − window_days`` (a trailing
    window). When ``classify == "routine"`` the SAME machine counts ROUTINE
    buyers/sellers instead — the registered control arm. ``n_buyers_mask`` is the
    integer count of qualifying BUYERS per date x name (the k>=2 cluster gate input;
    sellers never enter the gate — a name is long-eligible on its buyer cluster).

    If ``sector_map`` is given, ``net`` is SECTOR-DEMEANED within GICS sector
    (group-demean per date) BEFORE the cross-sectional ``_zscore_rows`` — the
    registered sector-neutrality, handled in the signal. Sellers default to empty,
    in which case ``net`` reduces to buys-only (the synthetic world has no sells).

    PIT SAFETY (law #1), inherited verbatim from ``cluster_buy_signal``: only Form
    4s FILED on/before t enter either count (a buy/sell filed AFTER t is invisible
    at t — pinned by a poison-the-future test), and each row's opportunistic/
    routine label uses ONLY that owner's prior filed trades of the same ticker
    (past-only). Sector-demeaning uses the near-static current GICS map (same
    convention H1 uses), which carries no forward information.
    """
    if classify not in ("opportunistic", "routine"):
        raise ValueError(
            f"classify must be 'opportunistic' or 'routine', got {classify!r}")
    want_opp = classify == "opportunistic"
    asof_dates = pd.DatetimeIndex(asof_dates)

    # Resolve the ticker universe from whatever legs carry rows.
    if tickers is None:
        tkrs: set[str] = set()
        for pnl in (purchases_panel, sells_panel):
            if pnl is not None and not pnl.empty and "ticker" in pnl.columns:
                tkrs |= set(pnl["ticker"].dropna().unique().tolist())
        tickers = sorted(tkrs)

    if (purchases_panel is None or purchases_panel.empty) and (
            sells_panel is None or sells_panel.empty):
        zero = pd.DataFrame(0.0, index=asof_dates, columns=tickers)
        return zero, zero.copy()

    # Past-only routine/opportunistic masks for the buy leg (label on buy history)
    # and the sell leg (label on sell history) — the requested class. Uses the
    # linear-time _label_opportunistic_fast (identical semantics to the pinned
    # label_opportunistic, pinned by a parity test) so a graded run over years of
    # S&P-500 Form 4s is tractable.
    if purchases_panel is not None and not purchases_panel.empty:
        opp_buy = _label_opportunistic_fast(
            purchases_panel, min_consecutive_years=min_consecutive_years)
        buy_mask = opp_buy if want_opp else ~opp_buy
        buyers = _distinct_buyer_counts(
            purchases_panel, asof_dates, tickers, window_days, buy_mask)
    else:
        buyers = pd.DataFrame(0.0, index=asof_dates, columns=tickers)

    if sells_panel is not None and not sells_panel.empty:
        sell_mask = _label_sells_routine_or_opp(
            sells_panel, min_consecutive_years, want_opportunistic=want_opp)
        sellers = _distinct_buyer_counts(
            sells_panel, asof_dates, tickers, window_days, sell_mask)
    else:
        sellers = pd.DataFrame(0.0, index=asof_dates, columns=tickers)

    net = buyers.sub(sellers, fill_value=0.0).reindex(columns=tickers, fill_value=0.0)

    # Sector-demean within GICS sector per date, BEFORE the cross-sectional z-score.
    if sector_map is not None:
        net = _sector_demean_rows(net, sector_map)

    return _zscore_rows(net), buyers.reindex(columns=tickers, fill_value=0.0)


def _sector_demean_rows(panel: pd.DataFrame, sector_map: dict[str, str]) -> pd.DataFrame:
    """Per-date group-demean within GICS sector: each cell minus its date x sector
    mean. Names absent from the map fall into an 'UNKNOWN' bucket (same convention
    as ``universe.sector_map``). PIT-safe: the current GICS map is near-static and
    carries no forward information (the convention H1 uses)."""
    sectors = pd.Series({c: sector_map.get(c, "UNKNOWN") for c in panel.columns})
    out = panel.copy()
    for sec in sectors.unique():
        cols = sectors.index[sectors == sec].tolist()
        cols = [c for c in cols if c in out.columns]
        if not cols:
            continue
        sub = out[cols]
        out[cols] = sub.sub(sub.mean(axis=1), axis=0)
    return out


def long_vs_ew_weights(
    signal: pd.DataFrame,
    n_buyers_mask: pd.DataFrame,
    prices_cols: pd.DataFrame,
    quantile: float = 0.10,
) -> pd.DataFrame:
    """The H10 frozen "long-vs-EW" book: dollar-neutral weights.

    LONG = equal-weight (+) of names in the TOP ``quantile`` of ``signal`` that
    ALSO pass the k>=2 cluster gate (``n_buyers_mask >= 2``); SHORT = equal-weight
    (−) of the FULL priceable universe that date (names with a non-NaN price in
    ``prices_cols``). The long leg sums to +0.5, the short leg to −0.5 (dollar-
    neutral by construction; beta ≈ 0 because both legs ≈ market beta). A date with
    fewer than 2 cluster-eligible names yields an all-zero row (no position).

    ``prices_cols`` is the (date x ticker) price frame on the signal's grid; its
    per-date non-NaN columns define the priceable benchmark universe (the EW short
    leg is the registered benchmark). ``signal`` and ``n_buyers_mask`` come from
    ``net_cluster_buy_signal``.

    PIT (law #1): every input is as-of-t (the signal/mask use only filings <= t;
    ``prices_cols`` carries the close known at t). The forward return is applied by
    ``cluster_backtest`` (weights at t earn t->t+1), so the book never holds on
    information it could not have had at t.
    """
    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    prices_cols = prices_cols.reindex(columns=signal.columns)
    mask = n_buyers_mask.reindex(index=signal.index, columns=signal.columns,
                                 fill_value=0.0)
    for d in signal.index:
        # Priceable benchmark universe this date (the EW short leg).
        px_row = prices_cols.loc[d] if d in prices_cols.index else None
        priceable = (px_row.dropna().index.tolist() if px_row is not None
                     else signal.columns.tolist())
        if not priceable:
            continue
        # Cluster-eligible LONG candidates: pass k>=2 AND be priceable.
        eligible = [c for c in priceable if mask.loc[d, c] >= 2]
        if len(eligible) < 2:
            continue  # <2 cluster names -> no position (all-zero row)
        sig_elig = signal.loc[d, eligible].dropna()
        if sig_elig.empty:
            continue
        n_long = int(len(sig_elig) * quantile)
        if n_long < 1:
            n_long = 1  # keep a non-empty top-decile long basket when any eligible
        longs = sig_elig.nlargest(n_long).index
        if len(longs) == 0:
            continue
        weights.loc[d, longs] = 0.5 / len(longs)
        weights.loc[d, priceable] = weights.loc[d, priceable] - 0.5 / len(priceable)
    return weights
