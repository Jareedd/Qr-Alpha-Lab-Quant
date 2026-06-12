"""H7 borrow collector: known-answer parsing, no network anywhere.

The collector's value is schema stability over months of unattended
cycles -- so the parser is pinned against the verified 2026-06-12 file
format including its quirks ('>' capped availability, '#'-prefixed meta
lines, bond symbols, malformed lines counted not fatal).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import borrow

FIXTURE = """#BOF|2026.06.12|16:07:43
#SYM|CUR|NAME|CON|ISIN|REBATERATE|FEERATE|AVAILABLE|FIGI|
AAPL|USD|APPLE INC|265598|US0378331005|3.8300|0.2500|>10000000|BBG000B9XRY4|
GME|USD|GAMESTOP CORP|36285627|US36467W1099|-12.5000|16.7500|450000|BBG000BB5BF6|
049323AB4|USD|CB ATLAS FINL HLDGS IN D04-14-22 06.625% 27|557991763|XXXXXXX3AB46|3.3700|0.2500|300000|BBG017QTG7P1|
BROKENLINE|USD|TOO|FEW
ZRX|USD|NO FEE GIVEN|111|US111|||0|BBG111|
#EOF
"""


def test_parse_known_answer():
    stamp, frame = borrow.parse_ibkr_short_file(FIXTURE)
    assert stamp == "2026-06-12 16:07:43"
    assert len(frame) == 4  # broken line skipped, counted
    assert frame.attrs["n_skipped"] == 1

    aapl = frame.loc["AAPL"]
    assert aapl["fee_rate"] == pytest.approx(0.25)
    assert aapl["rebate_rate"] == pytest.approx(3.83)
    assert aapl["available"] == 10_000_000 and bool(aapl["available_capped"])

    gme = frame.loc["GME"]
    assert gme["fee_rate"] == pytest.approx(16.75)
    assert gme["rebate_rate"] == pytest.approx(-12.5)  # negative rebate = hard
    assert gme["available"] == 450_000 and not gme["available_capped"]

    zrx = frame.loc["ZRX"]  # blank fee fields -> None, not crash, not zero
    assert zrx["fee_rate"] is None or str(zrx["fee_rate"]) == "nan"


def test_snapshot_filters_to_universe_and_reports_aggregates():
    stamp, frame = borrow.parse_ibkr_short_file(FIXTURE)
    snap = borrow.build_snapshot(stamp, frame, ["AAPL", "GME", "NOT_IN_FILE"])
    assert snap["universe_size"] == 3
    assert snap["universe_covered"] == 2
    assert set(snap["names"]) == {"AAPL", "GME"}
    assert snap["n_symbols_in_file"] == 4
    assert snap["n_skipped_lines"] == 1
    assert snap["file_stamp"] == "2026-06-12 16:07:43"
    # JSON-serializable end to end (numpy types would break json.dump)
    import json

    json.dumps(snap)
    # aggregates exist and are sane
    p = snap["fee_rate_percentiles_full_file"]
    assert p["50"] <= p["90"] <= p["99"]


def test_empty_or_garbage_file_yields_empty_frame_not_crash():
    stamp, frame = borrow.parse_ibkr_short_file("")
    assert stamp == "" and frame.empty
    stamp, frame = borrow.parse_ibkr_short_file("garbage\nmore|garbage\n")
    assert frame.empty
    assert frame.attrs["n_skipped"] == 2
