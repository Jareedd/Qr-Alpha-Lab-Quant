"""Thin loader for Ken French research-factor CSVs (numpy/pandas only).

The monthly 5-factor file (F-F_Research_Data_5_Factors_2x3.csv) has 3 header
text lines, a blank line, then the header ',Mkt-RF,SMB,HML,RMW,CMA,RF', then a
monthly block keyed by 6-digit YYYYMM, then a blank line, then an
' Annual Factors: January-December' section keyed by 4-digit years, then a
copyright line. We read ONLY the monthly block. The DAILY 5-factor file is a
SEPARATE Ken French download; the monthly file cannot drive a daily pipeline.
"""

from __future__ import annotations

import io
import os
import re

import pandas as pd

FF5_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
DAILY_FF_REQUIRED = (
    "F-F_Research_Data_5_Factors_2x3_daily.CSV is a separate Ken French "
    "download; the monthly file cannot drive the daily equity pipeline."
)
_MONTHLY_KEY = re.compile(r"^\s*\d{6}\s*,")


def load_ff_factors_monthly(path: str | os.PathLike) -> pd.DataFrame:
    """Load the FF 5-factor MONTHLY CSV. Returns a tidy DataFrame in DECIMAL
    (file values / 100), DatetimeIndex at month-END, columns FF5_COLUMNS, index
    name 'date'. Stops at the Annual Factors section / first non-6-digit key. RF
    is divided by 100 too (all columns are percent)."""
    with open(path, "r", newline="") as fh:
        lines = fh.read().splitlines()
    hdr = next(i for i, ln in enumerate(lines) if ln.lstrip().startswith(",Mkt-RF"))
    body = []
    for ln in lines[hdr + 1 :]:
        if _MONTHLY_KEY.match(ln):
            body.append(ln)
        elif body:  # first non-monthly line AFTER data starts => stop
            break
    df = pd.read_csv(
        io.StringIO("\n".join([lines[hdr]] + body)),
        skipinitialspace=True,
    )
    df = df.rename(columns={df.columns[0]: "key"})
    df["key"] = df["key"].astype(str).str.strip()
    df = df[df["key"].str.fullmatch(r"\d{6}")]  # belt-and-suspenders
    idx = pd.PeriodIndex(df["key"], freq="M").to_timestamp("M")
    idx.name = "date"
    out = df[FF5_COLUMNS].astype(float).to_numpy() / 100.0
    return pd.DataFrame(out, index=idx, columns=FF5_COLUMNS)


def load_ff_factors_daily(path: str | os.PathLike) -> pd.DataFrame:
    """The daily FF 5-factor file is a separate download (law #7 — never
    fabricate). Raises FileNotFoundError with guidance if absent; flag, do not
    block the monthly path."""
    if not os.path.exists(path):
        raise FileNotFoundError(DAILY_FF_REQUIRED + f" (looked for {path!r})")
    raise NotImplementedError(
        "Daily FF parsing not implemented; " + DAILY_FF_REQUIRED
    )
