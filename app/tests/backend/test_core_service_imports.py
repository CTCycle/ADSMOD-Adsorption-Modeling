from __future__ import annotations

import sys


###############################################################################
def test_core_service_import_and_routes() -> None:
    from core_service.app import app

    assert app is not None
    assert not any(
        path.startswith("/api/training") for path in {route.path for route in app.routes}
    )


###############################################################################
def test_core_service_does_not_import_ml_libs() -> None:
    before = set(sys.modules)
    __import__("core_service.app")
    loaded = set(sys.modules) - before
    assert "torch" not in loaded
    assert "keras" not in loaded
    assert "sklearn" not in loaded
