"""Registration enforcement (quantlab.registry): law #3 as an exit code."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantlab import registry

ROOT = os.path.join(os.path.dirname(__file__), "..")

FIXTURE = """# Pre-registered hypotheses

### H1: something proposed
- Status: PROPOSED — blocked on a data-source decision
- Economic prior: ...

### H2: something already run
- Status: RUN (trial #8)

### H7: collection-only
- Status: REGISTERED 2026-06-12 with owner sign-off, COLLECTION-ONLY

### H9: dead idea
- Status: ABANDONED (killed by the census)

### H10: malformed block with no status line
- Economic prior: ...
"""


def test_registration_status_parsing():
    assert registry.registration_status(FIXTURE, "H1") == "PROPOSED"
    assert registry.registration_status(FIXTURE, "H2") == "RUN"
    assert registry.registration_status(FIXTURE, "H7") == "REGISTERED"
    assert registry.registration_status(FIXTURE, "H9") == "ABANDONED"
    assert registry.registration_status(FIXTURE, "H10") == "UNKNOWN"
    assert registry.registration_status(FIXTURE, "H99") is None
    # 'H1' must not match 'H10' by prefix
    assert registry.registration_status(FIXTURE, "H1") != "UNKNOWN"


def test_require_runnable_registration(tmp_path):
    path = tmp_path / "reg.md"
    path.write_text(FIXTURE, encoding="utf-8")

    registry.require_runnable_registration("H1", str(path))  # PROPOSED: ok

    with pytest.raises(RuntimeError, match="not registered"):
        registry.require_runnable_registration("H99", str(path))
    with pytest.raises(RuntimeError, match="status RUN"):
        registry.require_runnable_registration("H2", str(path))
    with pytest.raises(RuntimeError, match="status REGISTERED"):
        registry.require_runnable_registration("H7", str(path))  # collection != run
    with pytest.raises(RuntimeError, match="status ABANDONED"):
        registry.require_runnable_registration("H9", str(path))
    with pytest.raises(RuntimeError, match="unreadable"):
        registry.require_runnable_registration("H1", str(tmp_path / "missing.md"))


def test_against_the_real_registry():
    # The real file must authorize exactly the hypotheses the project
    # believes are runnable today: H2/H3/H4 PROPOSED (runnable on
    # sign-off), H7 REGISTERED collection-only (NOT runnable as a trial).
    path = os.path.join(ROOT, registry.REGISTRY_PATH)
    with open(path, encoding="utf-8") as f:
        md = f.read()
    assert registry.registration_status(md, "H2") == "PROPOSED"
    assert registry.registration_status(md, "H7") == "REGISTERED"
    registry.require_runnable_registration("H2", path)
    with pytest.raises(RuntimeError):
        registry.require_runnable_registration("H7", path)
