"""Walk-forward validation with embargo (leakage-aware splitting).

Standard k-fold CV is invalid for overlapping financial labels: a label at
date t contains returns through t + horizon, so training dates within
``embargo`` days of the test window must be dropped (cf. Lopez de Prado,
'Advances in Financial Machine Learning', ch. 7: purged k-fold).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class WalkForwardSplitter:
    """Expanding-window walk-forward splits over a DatetimeIndex.

    train: [start, test_start - embargo)
    test:  [test_start, test_start + test_days)
    then the window rolls forward by ``test_days``.
    """

    min_train_days: int = 756  # 3 years
    test_days: int = 126       # 6 months
    embargo_days: int = 21     # >= label horizon

    def split(self, dates: pd.DatetimeIndex):
        dates = dates.sort_values().unique()
        n = len(dates)
        start = self.min_train_days
        while start + self.test_days <= n:
            test_idx = dates[start : start + self.test_days]
            train_end = start - self.embargo_days
            if train_end <= 0:
                break
            train_idx = dates[:train_end]
            yield train_idx, test_idx
            start += self.test_days

    def n_splits(self, dates: pd.DatetimeIndex) -> int:
        return sum(1 for _ in self.split(dates))
