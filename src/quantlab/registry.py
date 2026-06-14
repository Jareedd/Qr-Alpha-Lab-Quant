"""Pre-registration enforcement: culture, turned into code.

Law #3 and the registration protocol live in prose (CLAUDE.md,
preregistered_hypotheses.md). Prose can be forgotten at 11pm; an exit
code cannot. `run_pipeline` refuses any real-data run that is not either
(a) an explicitly named, currently-PROPOSED registration — a new trial —
or (b) an explicitly stated reproduction of existing logged work. The
synthetic modes (planted/noise/...) need nothing: synthetic data is free
by law.

This module only READS the registration file; editing registrations
remains a human act with git history as the audit trail.
"""

from __future__ import annotations

import re

REGISTRY_PATH = "writeup/preregistered_hypotheses.md"


def registration_status(md_text: str, name: str) -> str | None:
    """Status word of registration ``name`` ('H2'), or None if absent.

    A registration is a '### H<n>:' heading; its status is the first
    '- Status:' line in the block (first WORD kept: PROPOSED / REGISTERED
    / RUN / ABANDONED).
    """
    for block in re.split(r"^###\s+", md_text, flags=re.MULTILINE)[1:]:
        head, _, body = block.partition("\n")
        m = re.match(rf"{re.escape(name)}\s*:", head.strip())
        if not m:
            continue
        # Tolerate markdown bold anywhere around the label OR the value
        # ('- Status: **RUN ...' must parse as RUN, not UNKNOWN).
        sm = re.search(r"-\s*\**Status\**\s*:\s*\**\s*([A-Z]+)", body)
        return sm.group(1) if sm else "UNKNOWN"
    return None


def require_runnable_registration(name: str, registry_path: str = REGISTRY_PATH) -> None:
    """Raise (with a message worth reading) unless ``name`` exists and is
    PROPOSED — the only status that authorizes spending a trial. RUN means
    it was already spent (re-running it is a reproduction, use
    --reproduce); ABANDONED and collection-only registrations authorize
    nothing."""
    try:
        with open(registry_path, encoding="utf-8") as f:
            md = f.read()
    except OSError as exc:
        raise RuntimeError(
            f"registration file {registry_path} unreadable ({exc}) -- a "
            "real-data run without a readable registry is not authorized"
        ) from exc
    status = registration_status(md, name)
    if status is None:
        raise RuntimeError(
            f"{name} is not registered in {registry_path}. Register it "
            "(hypothesis, exact config, success AND kill criteria, paired "
            "controls) BEFORE the run -- that is the difference between "
            "testing a hypothesis and mining one."
        )
    if status != "PROPOSED":
        raise RuntimeError(
            f"{name} has status {status}, which does not authorize a new "
            "trial run. RUN = already spent (use --reproduce for a re-run "
            "of logged work); REGISTERED/collection-only = no analysis "
            "authorized; ABANDONED = dead."
        )
