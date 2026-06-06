from __future__ import annotations

from fastapi.testclient import TestClient


def test_core_routes_exclude_training() -> None:
    from core_service.app import app

    client = TestClient(app)
    assert client.get("/api/training/status").status_code == 404


def test_core_health_route() -> None:
    from core_service.app import app

    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200


def test_unified_app_exposes_core_routes_without_direct_training_ownership() -> None:
    from app.server.app import app

    paths = {route.path for route in app.routes}
    assert "/api/health" in paths
    assert any(path.startswith("/api/datasets") for path in paths)
    assert any(path.startswith("/api/fitting") for path in paths)
    assert any(path.startswith("/api/nist") for path in paths)
    assert any(path.startswith("/api/training") for path in paths)

