"""Name->CIK crosswalk parsers: known-answer tests on real cik-lookup lines.

The fixture lines are verbatim from the live cik-lookup-data.txt probe, so these
pin the exact dead-name recoveries (Celgene, Monsanto, Xilinx) that ticker-keyed
sources miss or mis-map -- without any network.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import cik_crosswalk as cx


def test_normalize_name_drops_qualifiers_and_suffixes():
    assert cx.normalize_name("CELGENE CORP /DE/") == "CELGENE"
    assert cx.normalize_name("Celgene") == "CELGENE"
    assert cx.normalize_name("MONSANTO CO /NEW/") == "MONSANTO"
    assert cx.normalize_name("Xilinx, Inc.") == "XILINX"
    assert cx.normalize_name("Red Hat Inc") == "RED HAT"
    assert cx.normalize_name("The Williams Companies") == "WILLIAMS"


_LOOKUP = """\
CELGENE ALPINE INVESTMENT CO., LLC:0001581774:
CELGENE CORP /DE/:0000816284:
MONSANTO CO:0000067686:
MONSANTO CO /NEW/:0001110783:
XILINX INC:0000743988:
XILINX, INCORPORATED:0000743988:
RED HAT INC:0001087423:
APPLE INC:0000320193:
"""


def test_parse_cik_lookup_normalized_index():
    idx = cx.parse_cik_lookup(_LOOKUP)
    assert idx["CELGENE"] == ["0000816284"]           # the operating co
    assert "CELGENE ALPINE INVESTMENT" in idx          # the financing entity, separate key
    assert idx["MONSANTO"] == ["0000067686", "0001110783"]  # both Monsantos, sorted
    assert idx["XILINX"] == ["0000743988"]             # two names -> one CIK, deduped
    assert idx["APPLE"] == ["0000320193"]


def test_match_name_exact_and_recovers_dead_names():
    idx = cx.parse_cik_lookup(_LOOKUP)
    # the recoveries ticker-keyed maps miss / mis-map:
    assert cx.match_name("Celgene", idx) == ["0000816284"]
    assert "0001110783" in cx.match_name("Monsanto", idx)   # NOT the SPAC FMP returned
    assert cx.match_name("Xilinx, Inc.", idx) == ["0000743988"]
    assert cx.match_name("Red Hat", idx) == ["0001087423"]


def test_match_name_ambiguous_prefix_refused():
    # "CELGENE" prefix also matches "CELGENE ALPINE INVESTMENT"; an exact key
    # exists so that wins, but a bare ambiguous prefix with no exact key returns
    # nothing rather than a false link.
    idx = {"CELGENE ALPINE INVESTMENT": ["0001581774"],
           "CELGENE EUROPEAN INVESTMENT": ["0001577650"]}
    assert cx.match_name("Celgene", idx) == []         # 2 distinct prefix names -> refuse


def test_match_name_unique_prefix_resolves():
    idx = {"WELLCARE HEALTH PLANS": ["0001279363"]}
    assert cx.match_name("WellCare Health", idx) == ["0001279363"]  # unique prefix
