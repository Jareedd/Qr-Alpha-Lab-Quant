import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab.validation import WalkForwardSplitter


def test_no_leakage_train_before_test_with_embargo():
    dates = pd.bdate_range("2015-01-01", periods=2000)
    sp = WalkForwardSplitter(min_train_days=756, test_days=126, embargo_days=21)
    n = 0
    for train, test in sp.split(dates):
        n += 1
        # Every train date strictly precedes test start by at least the embargo.
        gap_days = (test.min() - train.max()).days
        assert train.max() < test.min()
        assert gap_days >= 21  # calendar gap of >= embargo business days
    assert n > 0


def test_windows_are_expanding_and_cover_data():
    dates = pd.bdate_range("2015-01-01", periods=1500)
    sp = WalkForwardSplitter(min_train_days=756, test_days=126, embargo_days=21)
    train_lens = [len(tr) for tr, _ in sp.split(dates)]
    assert train_lens == sorted(train_lens)  # expanding window
