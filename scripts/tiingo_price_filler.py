"""Patient, probe-gated Tiingo price filler for the H1 universe.

Tiingo's daily quota can't take an 812-name burst in one go; this fills the EOD
price cache across resumable passes. Each pass first PROBES with one call
(AAPL): if rate-limited it sleeps and retries later (so it never grinds 800
backoffs against a spent quota); when the probe succeeds it pulls, bailing the
pass early if it gets re-limited. Successes cache to parquet; 429-skips don't
(so the next pass retries them). Safe to run unattended; idempotent.
"""

from __future__ import annotations

import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import universe as univ
from quantlab.tiingo_data import TiingoSource

START, END = "2009-06-01", "2026-06-24"
MAX_PASSES, SLEEP = 8, 1500          # ~25 min between passes; ~3h cap
COVERAGE_TARGET = 0.93


def n_cached() -> int:
    return len(glob.glob(os.path.join("data_cache", "tiingo", "eod_*.parquet")))


def main() -> None:
    cur, chg = univ.fetch_sp500_tables()
    members = univ.all_members_in_window(
        univ.build_membership_intervals(cur, chg, start="2010-01-01"))
    tg = TiingoSource()
    n = len(members)
    print(f"[filler] universe {n} names; start cached={n_cached()}", flush=True)
    for p in range(MAX_PASSES):
        probe = tg.eod("AAPL", START, END)
        if not len(probe):
            print(f"[filler] pass {p}: quota spent (AAPL empty); sleep {SLEEP}s", flush=True)
            time.sleep(SLEEP)
            continue
        got = empties = 0
        for t in members:
            s = tg.eod(t, START, END)
            if len(s):
                got += 1
                empties = 0
            else:
                empties += 1
                if empties >= 15:        # re-limited mid-pass -> stop, sleep, resume
                    print(f"[filler] pass {p}: re-limited after {got} ok; sleeping", flush=True)
                    break
        cached = n_cached()
        print(f"[filler] pass {p}: cached={cached}/{n} (got~{got})", flush=True)
        if cached >= int(COVERAGE_TARGET * n):
            print(f"[filler] TARGET reached: {cached}/{n}", flush=True)
            break
        time.sleep(SLEEP)
    print(f"[filler] DONE cached={n_cached()}/{n}", flush=True)


if __name__ == "__main__":
    main()
