import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import ff_factors

FF5_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "scratch_refute",
    "F-F_Research_Data_5_Factors_2x3.csv",
)
pytestmark = pytest.mark.skipif(not os.path.exists(FF5_PATH), reason="FF file absent")


def test_load_ff_factors_monthly_known_values():
    df = ff_factors.load_ff_factors_monthly(FF5_PATH)
    assert df.columns.tolist() == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert df.index[0] == pd.Timestamp("1963-07-31")
    assert df.loc["1963-07", "Mkt-RF"].iloc[0] == pytest.approx(-0.0039, abs=1e-12)
    assert df.index[-1] == pd.Timestamp("2026-04-30")
    assert df.loc["2026-04", "Mkt-RF"].iloc[0] == pytest.approx(0.0994, abs=1e-12)
    assert len(df) == 754  # monthly rows only
    assert df.index.is_monotonic_increasing and df.index.is_unique
    # annual section absent: 1964 annual Mkt-RF was 12.59% => 0.1259, never a row
    assert pd.Timestamp("1964-12-31") in df.index  # the 196412 MONTH, not annual
    assert not (df["Mkt-RF"] == pytest.approx(0.1259)).any()


def test_load_ff_factors_daily_flags():
    with pytest.raises(FileNotFoundError, match="separate Ken French download"):
        ff_factors.load_ff_factors_daily(
            os.path.join(os.path.dirname(FF5_PATH), "does_not_exist_daily.CSV")
        )
