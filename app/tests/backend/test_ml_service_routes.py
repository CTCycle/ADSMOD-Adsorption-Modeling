from __future__ import annotations

from fastapi.testclient import TestClient


###############################################################################
def test_ml_health_route() -> None:
    from ml_service.app import app

    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
