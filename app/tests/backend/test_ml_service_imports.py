from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "app/server/core_service"))
sys.path.insert(0, str(ROOT / "app/server/ml_service"))
sys.path.insert(0, str(ROOT / "app/server/shared"))

def test_ml_service_import_and_routes() -> None:
    from ml_service.app import app

    assert app is not None
    assert any(path.startswith('/api/training') for path in {route.path for route in app.routes})

