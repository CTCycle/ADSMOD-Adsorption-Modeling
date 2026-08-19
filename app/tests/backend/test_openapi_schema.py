from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "shared" / "openapi.json"

###############################################################################
def test_shared_openapi_schema_covers_unified_api_surface() -> None:
    schema: dict[str, Any] = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert schema["openapi"] == "3.1.0"
    paths = schema["paths"]
    assert "/api/datasets" in paths
    assert "/api/training/datasets" in paths
