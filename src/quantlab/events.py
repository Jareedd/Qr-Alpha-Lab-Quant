"""H8 event study: do discretionary S&P 500 deletions rebound post-effective,
vs a SIZE- and MOMENTUM-matched control?

The whole methodology is the control. Deleted names are mechanically small
recent losers, and small recent losers bounce regardless of any index
effect -- so a raw post-deletion rebound proves nothing. The test is
whether deleted names out-rebound a contemporaneous basket matched on the
two things that drive that mechanical bounce: size and trailing return.
Greenwood-Sammon (2025) showed the announcement->effective deletion effect
has decayed to ~0.1% post-2010; this asks the strictly different
post-effective question, with a weak declared prior.

Point-in-time by construction: the matched pool at event date t is the
index membership at t; matching features use data through t only; the
position is entered the day AFTER the effective date and held forward.

Registered amendment (2026-06-13, pre-run): the registration named
"log-mcap" as a matching variable. Free point-in-time market cap is not
available, so size is proxied by log trailing-63d dollar volume -- a
standard liquidity/size proxy computable point-in-time from prices x
volume. Documented before the run, per law #5 (not a post-hoc change).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.universe import classify_removal_reason


def discretionary_deletions(
    changes: pd.DataFrame, start: str = "2010-01-01"
) -> pd.DataFrame:
    """(effective_date, ticker) for removals whose reason classifies as
    'discretionary' -- committee market-cap/representativeness deletions,
    NOT M&A/bankruptcy (those leave no tradeable post-event name) and NOT
    index migrations (Greenwood-Sammon's confound)."""
    rem = changes.dropna(subset=["removed"]).copy()
    rem["date"] = pd.to_datetime(rem["date"])
    rem = rem[rem["date"] >= pd.Timestamp(start)]
    rem["bucket"] = rem["reason"].map(classify_removal_reason)
    out = (rem[rem["bucket"] == "discretionary"][["date", "removed"]]
           .rename(columns={"date": "effective_date", "removed": "ticker"}))
    return out.dropna().drop_duplicates().reset_index(drop=True)


def _match_features(
    prices: pd.DataFrame, dollar_vol: pd.DataFrame, asof: pd.Timestamp,
    pool: list[str], vol_lb: int = 63, ret_lb: int = 126,
) -> pd.DataFrame | None:
    """Per-name (size proxy, trailing return) as of ``asof``, past-only."""
    px = prices.loc[:asof]
    if len(px) < ret_lb + 1:
        return None
    size = np.log(dollar_vol.loc[:asof].iloc[-vol_lb:].mean().replace(0, np.nan))
    mom = px.iloc[-1] / px.iloc[-ret_lb] - 1.0
    feat = pd.DataFrame({"size": size, "mom": mom}).reindex(pool)
    return feat.dropna()


def matched_controls(target: str, feat: pd.DataFrame, n: int = 10) -> list[str]:
    """``n`` nearest names to ``target`` by standardized (size, mom)
    distance, excluding the target itself."""
    z = (feat - feat.mean()) / (feat.std() + 1e-12)
    if target not in z.index:
        return []
    dist = ((z - z.loc[target]) ** 2).sum(axis=1).drop(target, errors="ignore")
    return dist.nsmallest(n).index.tolist()


def event_study(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    dollar_vol: pd.DataFrame,
    member_mask: pd.DataFrame | None = None,
    horizon: int = 60,
    entry_lag: int = 1,
    n_match: int = 10,
    cost_bps: float = 10.0,
) -> dict:
    """Run the matched-control event study. Returns the daily event-time
    excess-return portfolio (overlapping events averaged per calendar day),
    per-event total excess returns, and the matched control's own raw
    rebound (to show the deleted names must BEAT it, not merely bounce).

    Position: enter ``entry_lag`` days after the effective date (avoids the
    depressed forced-flow close), hold ``horizon`` trading days, long the
    deleted name and short the matched basket. Round-trip cost ``cost_bps``
    each way is charged once per event.
    """
    rets = prices.pct_change(fill_method=None)
    idx = prices.index
    daily, event_total, control_raw, used = [], [], [], []
    for _, row in events.iterrows():
        t = row["ticker"]
        d = pd.Timestamp(row["effective_date"])
        if t not in prices.columns:
            continue
        pos = int(idx.searchsorted(d, side="right")) - 1  # last day <= d
        entry = pos + entry_lag
        if pos < 126 or entry + horizon >= len(idx):
            continue
        asof = idx[pos]
        if member_mask is not None and asof in member_mask.index:
            pool = [c for c in member_mask.columns
                    if member_mask.loc[asof, c] and c != t and c in prices.columns]
        else:
            pool = [c for c in prices.columns if c != t]
        feat = _match_features(prices, dollar_vol, asof, pool + [t])
        if feat is None or t not in feat.index or len(feat) < n_match + 1:
            continue
        ctrls = matched_controls(t, feat, n=n_match)
        if not ctrls:
            continue
        win = slice(entry + 1, entry + horizon + 1)  # earns from entry+1
        tgt = rets[t].iloc[win]
        ctl = rets[ctrls].iloc[win].mean(axis=1)
        excess = tgt.to_numpy() - ctl.to_numpy()
        # one-time round-trip cost on the long leg, spread across the window
        excess[0] -= cost_bps / 1e4
        excess[-1] -= cost_bps / 1e4
        daily.append(pd.Series(excess, index=idx[win]))
        event_total.append(float(np.nansum(excess)))
        control_raw.append(float(np.nansum(ctl.to_numpy())))
        used.append((str(asof.date()), t))

    port = (pd.concat(daily, axis=1).mean(axis=1).dropna()
            if daily else pd.Series(dtype=float))
    return {
        "daily_portfolio": port,
        "event_total_excess": pd.Series(event_total),
        "control_raw_return": pd.Series(control_raw),
        "n_events": len(event_total),
        "events_used": used,
    }
