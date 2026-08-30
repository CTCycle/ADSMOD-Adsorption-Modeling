from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OPENAPI_ROOT = Path("app/backend/openapi")


def _read_schema(name: str) -> dict[str, Any]:
    return json.loads((OPENAPI_ROOT / name).read_text(encoding="utf-8"))


def test_core_openapi_snapshot_covers_core_surface() -> None:
    schema = _read_schema("core.json")
    assert schema["openapi"] == "3.1.0"
    paths = schema["paths"]
    assert "/api/v1/datasets" in paths
    assert "/api/v1/fitting/models" in paths
    assert not any(path.startswith("/api/training") for path in paths)


def test_ml_openapi_snapshot_covers_training_surface() -> None:
    schema = _read_schema("ml.json")
    assert schema["openapi"] == "3.1.0"
    paths = schema["paths"]
    assert "/api/v1/training/datasets" in paths
    assert "/api/v1/training/configuration" in paths
    assert not any(path.startswith("/api/v1/datasets") for path in paths)
