from __future__ import annotations

import os

from core_service.app import app
from core_service.configurations import resolve_spa_file_path

# TODO: Remove this compatibility shim after all runtime scripts and package paths use core_service.app:app directly.

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def public_host_mode_enabled() -> bool:
    """Compatibility behavior for legacy tests and callers using FASTAPI_HOST."""
    host = (os.getenv("FASTAPI_HOST") or "").strip().lower()
    if not host:
        host = (os.getenv("CORE_SERVICE_HOST") or "").strip().lower()
    if not host:
        return False
    return host not in LOOPBACK_HOSTS


__all__ = ["app", "public_host_mode_enabled", "resolve_spa_file_path"]
