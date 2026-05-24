from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "app/server/core_service"))
sys.path.insert(0, str(ROOT / "app/server/ml_service"))
sys.path.insert(0, str(ROOT / "app/server/shared"))
from fastapi.testclient import TestClient  # noqa: E402


def test_core_routes_exclude_training() -> None:
    from core_service.app import app

    client = TestClient(app)
    assert client.get('/api/training/status').status_code == 404


def test_core_health_route() -> None:
    from core_service.app import app

    client = TestClient(app)
    response = client.get('/api/health')
    assert response.status_code == 200

