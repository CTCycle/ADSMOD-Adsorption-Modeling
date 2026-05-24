from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "app/server/core_service"))
sys.path.insert(0, str(ROOT / "app/server/ml_service"))
sys.path.insert(0, str(ROOT / "app/server/shared"))
from fastapi.testclient import TestClient  # noqa: E402


def test_ml_routes_include_training() -> None:
    from ml_service.app import app

    client = TestClient(app)
    assert client.get('/api/training/status').status_code in {200, 500}
    assert client.get('/api/training/checkpoints').status_code in {200, 500}
    assert client.get('/api/training/datasets').status_code in {200, 500}


def test_ml_health_route() -> None:
    from ml_service.app import app

    client = TestClient(app)
    response = client.get('/api/health')
    assert response.status_code == 200

