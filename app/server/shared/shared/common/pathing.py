from __future__ import annotations

from pathlib import Path


def server_root() -> str:
    return str(Path(__file__).resolve().parents[3])
