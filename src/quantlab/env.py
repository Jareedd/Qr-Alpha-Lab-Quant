"""Minimal .env loader -- secrets stay out of code, git, and chat logs.

Reads KEY=VALUE lines (``#`` comments and blanks ignored) and injects them
into ``os.environ`` without overwriting variables that are already set, so a
real environment always wins over the file. Deliberately ~20 lines instead of
a python-dotenv dependency (project rule: no new dependencies without
justification, and this needs none).

Used from Phase 6 onward for the Alpaca paper-trading keys. ``.env`` is
gitignored; ``.env.example`` documents the expected keys.
"""

from __future__ import annotations

import os


def load_env(path: str = ".env") -> dict[str, str]:
    """Load KEY=VALUE pairs from ``path`` into os.environ (non-destructive)."""
    loaded: dict[str, str] = {}
    if not os.path.exists(path):
        return loaded
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            loaded[key] = value
            os.environ.setdefault(key, value)
    return loaded


def alpaca_credentials(path: str = ".env") -> tuple[str, str]:
    """Return (key_id, secret) for Alpaca paper trading, or raise with help."""
    load_env(path)
    key = os.environ.get("ALPACA_API_KEY_ID", "")
    secret = os.environ.get("ALPACA_API_SECRET_KEY", "")
    if not key or not secret:
        raise RuntimeError(
            "Alpaca credentials missing. Copy .env.example to .env and fill "
            "ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY (paper-trading keys "
            "from https://app.alpaca.markets, free account)."
        )
    return key, secret
