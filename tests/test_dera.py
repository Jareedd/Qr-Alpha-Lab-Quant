"""Known-answer tests for the DERA bulk-fundamentals parsers + PIT join.
Fixtures are tiny inline tab-separated tables (no network, no GB downloads)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import dera

SUB = "\t".join(["adsh", "cik", "name", "form", "period", "fy", "fp", "filed"]) + "\n" + \
      "\n".join([
          "\t".join(["0001-22-01", "320193", "APPLE INC", "10-K", "20210930", "2021", "FY", "20211029"]),
          "\t".join(["0001-23-02", "320193", "APPLE INC", "10-K", "20220930", "2022", "FY", "20221028"]),
          "\t".join(["0009-22-09", "718877", "ACTIVISION BLIZZARD", "10-K", "20211231", "2021", "FY", "20220223"]),
      ])

NUM = "\t".join(["adsh", "tag", "version", "ddate", "qtrs", "uom", "value"]) + "\n" + \
      "\n".join([
          "\t".join(["0001-22-01", "Assets", "us-gaap/2021", "20210930", "0", "USD", "351002000000"]),
          "\t".join(["0001-23-02", "Assets", "us-gaap/2022", "20220930", "0", "USD", "352755000000"]),
          "\t".join(["0009-22-09", "Assets", "us-gaap/2021", "20211231", "0", "USD", "25056000000"]),
      ])


def test_parse_sub_and_num():
    sub = dera.parse_sub(SUB)
    num = dera.parse_num(NUM)
    assert set(sub["cik"].dropna().astype(int)) == {320193, 718877}
    assert sub["filed"].dt.year.tolist() == [2021, 2022, 2022]
    assert num.loc[num["tag"] == "Assets", "value"].max() == 352755000000.0


def test_pit_value_uses_latest_filing_known_at_asof():
    sub, num = dera.parse_sub(SUB), dera.parse_num(NUM)
    # only the 2021 10-K is public by 2022-06-30 -> its Assets
    assert dera.pit_value(sub, num, 320193, "Assets", "2022-06-30") == 351002000000.0
    # by 2023-01-01 the 2022 10-K is public -> the fresher figure
    assert dera.pit_value(sub, num, 320193, "Assets", "2023-01-01") == 352755000000.0


def test_pit_value_none_before_any_filing_and_for_dead_name_after_death():
    sub, num = dera.parse_sub(SUB), dera.parse_num(NUM)
    assert dera.pit_value(sub, num, 320193, "Assets", "2020-01-01") is None
    # a dead filer (Activision, CIK persists) is still resolvable by CIK -- the
    # survivorship-safe property DERA gives the fundamentals side.
    assert dera.pit_value(sub, num, 718877, "Assets", "2022-06-30") == 25056000000.0


def test_pit_value_unknown_tag_is_none():
    sub, num = dera.parse_sub(SUB), dera.parse_num(NUM)
    assert dera.pit_value(sub, num, 320193, "NoSuchTag", "2023-01-01") is None


def test_filer_ciks():
    assert dera.filer_ciks(dera.parse_sub(SUB)) == {320193, 718877}
