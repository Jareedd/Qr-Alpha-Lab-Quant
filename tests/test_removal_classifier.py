"""H8 removal-reason classifier: known answers on real Wikipedia phrasings.

The census drives a kill-at-zero-cost power gate, so the classifier's
failure mode that matters is silently mis-bucketing migrations or M&A as
'discretionary' (inflating the event count past the gate). Unknowns must
stay unknown -- they are shown, never guessed.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab.universe import classify_removal_reason


def test_corporate_actions():
    assert classify_removal_reason("Acquired by Microsoft.") == "corporate_action"
    assert classify_removal_reason("Merged with Exxon to form ExxonMobil") == "corporate_action"
    assert classify_removal_reason("Taken private by KKR") == "corporate_action"
    assert classify_removal_reason("Spun off from parent") == "corporate_action"


def test_distress():
    assert classify_removal_reason("Filed for bankruptcy.") == "distress"
    assert classify_removal_reason("Chapter 11 filing") == "distress"
    assert classify_removal_reason("Delisted from NYSE") == "distress"
    assert (classify_removal_reason("The FDIC placed Signature Bank into "
                                    "FDIC Receivership.") == "distress")


def test_real_wikipedia_phrasings_from_first_census():
    # Phrasings the first census run left as 'unknown' -- pinned so the
    # classifier's coverage cannot silently regress.
    assert classify_removal_reason("TGNA spins off Cars.com[125]") == "corporate_action"
    assert classify_removal_reason("PCL taken over by WY[154]") == "corporate_action"
    assert (classify_removal_reason("Arconic separated into 2 companies - "
                                    "Howmet remained on the index.")
            == "corporate_action")
    # 'X replaces Y' carries NO reason -- must stay unknown, never guessed.
    assert classify_removal_reason("Ulta replaces Tenet[149]") == "unknown"


def test_migration_is_not_discretionary():
    # Greenwood-Sammon: migrations drove much of the index effect's
    # disappearance -- conflating them with committee deletions would
    # corrupt H8's event set in the optimistic direction.
    assert classify_removal_reason("Moved to S&P MidCap 400") == "migration"
    assert classify_removal_reason("S&P 500/400 constituent swap") == "migration"
    assert classify_removal_reason("Moved to S&P SmallCap 600") == "migration"


def test_discretionary():
    assert classify_removal_reason("Market capitalization change.") == "discretionary"
    assert (classify_removal_reason("No longer representative of the US economy")
            == "discretionary")


def test_unknowns_stay_unknown():
    assert classify_removal_reason(None) == "unknown"
    assert classify_removal_reason("") == "unknown"
    assert classify_removal_reason("Reasons unclear.") == "unknown"
