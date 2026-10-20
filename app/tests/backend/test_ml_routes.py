# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient
from server.configurations.settings import AdsmodConfig, DatabaseConfig, UserSettings
from server.app import create_app

###############################################################################
def _config(tmp_path: Path):
    return AdsmodConfig.assemble(
        database=DatabaseConfig(embedded_database=True, sqlite_path="data/test.db"),
        storage_root=tmp_path,
        settings=UserSettings(),
    )

###############################################################################
def test_ml_routes_are_mounted_on_the_single_backend(tmp_path: Path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        assert client.get("/health/live").json()["service"] == "backend"
        assert client.get("/api/v1/training/configuration").status_code == 200
        assert client.get("/api/v1/training/status").status_code == 200
        assert client.get("/api/v1/system/configuration").status_code == 200

###############################################################################
def test_standalone_ml_server_entrypoints_are_removed() -> None:
    root = Path(__file__).resolve().parents[2] / "server"
    assert (root / "app.py").exists()
    assert (root / "cli.py").exists()
    assert not (root / "http" / "entrypoint.py").exists()
    assert not (root / "clients" / "core_client.py").exists()
