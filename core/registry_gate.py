"""Deployment authority = the registry, mechanized (law #3 extended to live).

``quantlab.registry`` gates SPENDING a trial (PROPOSED is the only runnable
status). This module gates DEPLOYING a strategy: only a hypothesis whose
registry status is GRADUATED may be wrapped in a StrategyAgent and
registered with the PortfolioController.

As of 2026-07-08 nothing has graduated (N=13, zero graduations), so this
gate authorizes ZERO agents. That is the honest state of the research, not
a framework gap — the framework exists so the FIRST graduation has a
disciplined place to land.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:  # same per-file convention the test suite uses
    sys.path.insert(0, _SRC)

from quantlab.registry import REGISTRY_PATH, registration_status  # noqa: E402

#: The only status word that authorizes wrapping a hypothesis in an agent.
DEPLOYABLE_STATUS = "GRADUATED"


def require_deployable(hypothesis_id: str, registry_path: str | None = None) -> None:
    """Raise unless ``hypothesis_id`` exists in the registry with status
    GRADUATED. Reuses quantlab.registry's parser so there is exactly one
    definition of "what the registry says" in the codebase."""
    path = registry_path or os.path.join(_REPO_ROOT, REGISTRY_PATH)
    try:
        with open(path, encoding="utf-8") as f:
            md = f.read()
    except OSError as exc:
        raise RuntimeError(
            f"registry {path} unreadable ({exc}) — no deployment authority "
            "without a readable registry"
        ) from exc
    status = registration_status(md, hypothesis_id)
    if status is None:
        raise RuntimeError(
            f"{hypothesis_id} is not registered in {path}; an unregistered "
            "strategy cannot be deployed (law #3)."
        )
    if status != DEPLOYABLE_STATUS:
        raise RuntimeError(
            f"{hypothesis_id} has status {status}; only {DEPLOYABLE_STATUS} "
            "authorizes deployment. A RUN null or a PROPOSED idea is not a "
            "tradable strategy."
        )
