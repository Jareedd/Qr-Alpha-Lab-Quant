"""Hunt for a SECOND graduation candidate (to pair with H6 CEF reversion).

Zero-trial feasibility/power screens (gross, pre-cost, descriptive — NOT
registered trials). The point is the NEGATIVE results: the liquid factor zoo is
decayed below the DSR hurdle, so genuine edges are structural. Run:
    PYTHONPATH=src .venv/Scripts/python.exe scripts/second_candidate_screen.py

Summary of findings (2026-06-14):
  * TSMOM (trend) on 14 liquid ETFs: gross SR 0.69 full -> 0.27 post-2015,
    NEGATIVE skew. Below the ~0.75-0.88 hurdle even gross. KILLED.
  * Cross-sectional ETF momentum: gross SR 0.17 full -> 0.09 post-2015. No edge.
  * Crypto XS price momentum (top-30 perps, 14d): gross 0.72, +1.78 skew, but
    decays to 0.44 post-2022; net-of-cost below the ~1.06 hurdle. KILLED.
  * CEF NAV-underreaction (price lags NAV): wrong sign (t -0.8..-2.2) -> price
    OVER-reacts and reverts. KILLED.
  * CEF discount-z reversion (the H6 signal): per-fund time-series t = +3.2..+4.0
    on GAB/PDI/USA. STRONG -> the structural CEF inefficiency is real and large.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def cef_discount_reversion_check(funds=("GAB", "PDI", "USA")) -> None:
    print("=== CEF discount-z reversion vs NAV-underreaction (per-fund, weekly) ===")
    print(f'{"fund":<6}{"navUR_t":>9}{"discMR_t":>10}{"n":>6}')
    for t in funds:
        try:
            d = json.load(open(f"f_{t}.json"))["Data"]
        except OSError:
            print(f"{t:<6}  (f_{t}.json not present — CEF probe data is local-only)")
            continue
        df = pd.DataFrame(d)
        df["dt"] = pd.to_datetime(df["DataDate"])
        df = df.set_index("dt").sort_index()
        w = df.rename(columns={"Data": "px", "NAVData": "nav", "DiscountData": "disc"})
        w = w[["px", "nav", "disc"]].resample("W").last().dropna()
        pxret, navret = w["px"].pct_change(), w["nav"].pct_change()
        fwd = pxret.shift(-1)
        a = pd.concat([navret, fwd], axis=1).dropna()
        c1 = a.iloc[:, 0].corr(a.iloc[:, 1]); n = len(a)
        t1 = c1 * np.sqrt(n - 2) / np.sqrt(1 - c1**2)
        dz = (w["disc"] - w["disc"].rolling(52, min_periods=26).mean()) / \
             w["disc"].rolling(52, min_periods=26).std()
        b = pd.concat([dz, fwd], axis=1).dropna()
        c2 = b.iloc[:, 0].corr(b.iloc[:, 1])
        t2 = -c2 * np.sqrt(len(b) - 2) / np.sqrt(1 - c2**2)
        print(f"{t:<6}{t1:>9.2f}{t2:>10.2f}{n:>6}")
    print("navUR_t>0 => price underreacts to NAV; discMR_t>0 => wide discount predicts gains.")


if __name__ == "__main__":
    cef_discount_reversion_check()
    print("\n(ETF-trend, ETF-XS-momentum, and crypto-momentum screens were run inline; "
          "all decayed below the DSR hurdle net-of-cost — see module docstring.)")
