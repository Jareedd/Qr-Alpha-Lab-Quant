"""H1 / trial #12 entry point — registration-gated. COMMAND OF RECORD.

    python scripts/run_fundamentals.py --hypothesis H1 --source free_xwalk

This is a THIN DELEGATION to the registered graded construction in
``run_h1_trial`` (the two-arm RAW-vs-NEUTRAL CBOP/A book the 2026-06-16 /
2026-06-24 amendments freeze). The forbidden placeholder that used to live here —
a GP/A − accruals BLEND, EQUAL-weight book, MONTHLY rebalance — is NEUTRALIZED:
it is not the registration and must never be what the command of record runs. The
single source of truth for the graded run is ``run_h1_trial`` (registration gate
-> machinery gate -> DATA GATE -> two arms -> 4-config PBO/MDE -> verdict).

Order of operations (delegated to run_h1_trial.main):
1. Registration gate: H1 must be PROPOSED (law #3).
2. Machinery gate: synthetic planted_quality recovered, null_quality rejected
   (paired). Proves the harness can tell a quality premium from its absence.
3. DATA GATE: a graded trial requires a SURVIVORSHIP-SAFE source. The free SEC
   source (--source free_sec) is current-ticker-only (~73% coverage) — running H1
   on it would re-commit trial #1's original sin, so it is REFUSED (no trial).

Sources:
- free_sec    : free SEC fundamentals via current-only ticker->CIK map +
                Tiingo prices. SURVIVORSHIP-BLOCKED — fails the DATA GATE.
- compustat   : WRDS slot — survivorship-safe, awaiting access.
- free_xwalk  : SEC fundamentals via the name->CIK crosswalk (recovers dead/
                renamed names) + Tiingo prices — the FREE survivorship-safe path.
                Routes to the registered construction; a graded trial still
                requires explicit sign-off (N stays put until then).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# The registered graded run lives in run_h1_trial. Import it as the single source
# of truth so --source free_xwalk routes through the registered construction and
# the forbidden placeholder below is unreachable.
sys.path.insert(0, os.path.dirname(__file__))
import run_h1_trial


def _run_trial(source, n_trials: int) -> None:
    """Thin delegation to the registered graded two-arm run (B1).

    NOT a placeholder. The forbidden GP/A−accruals-blend / equal-weight / monthly
    construction that previously lived here is removed; this forwards to
    ``run_h1_trial._run_trial`` so there is exactly ONE graded construction in the
    repo — the registered one."""
    run_h1_trial._run_trial(source, n_trials)


def main() -> None:
    """Delegate entirely to run_h1_trial.main — the registered entry point. Kept
    as a stable command-of-record alias (`run_fundamentals.py`) so existing docs /
    RESUME commands continue to work, while the graded logic has a single home."""
    run_h1_trial.main()


if __name__ == "__main__":
    main()
