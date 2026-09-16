from __future__ import annotations

import inspect
from pathlib import Path

from server.services.ml_container import MlServiceContainer

###############################################################################
def test_ml_container_consumes_in_process_snapshot_access() -> None:
    parameters = inspect.signature(MlServiceContainer).parameters
    assert "snapshot_access" in parameters
    assert "internal_token" not in parameters

###############################################################################
def test_ml_extension_has_no_standalone_fastapi_server_or_core_http_client() -> None:
    root = Path("app/server")
    assert (root / "app.py").exists()
    assert (root / "cli.py").exists()
    assert not (root / "http" / "entrypoint.py").exists()
    assert not (root / "clients" / "core_client.py").exists()

###############################################################################
def test_ml_source_has_no_backend_to_backend_http_boundary() -> None:
    root = Path("app/server")
    source_roots = [root / "app.py", root / "cli.py"] + [
        root / name
        for name in (
            "api",
            "common",
            "configurations",
            "domain",
            "models",
            "repositories",
            "services",
        )
    ]
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for source_root in source_roots
        for path in ([source_root] if source_root.is_file() else source_root.rglob("*.py"))
    )
    assert "CoreSnapshotClient" not in combined
    assert "core_base_url" not in combined
    assert "X-ADSMOD-Internal-Token" not in combined
