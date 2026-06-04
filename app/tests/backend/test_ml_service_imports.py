from __future__ import annotations

def test_ml_service_import_and_routes() -> None:
    from ml_service.app import app

    assert app is not None
    assert any(
        path.startswith("/api/training") for path in {route.path for route in app.routes}
    )

