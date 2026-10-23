# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations
from pathlib import Path
from fastapi.testclient import TestClient
from server.configurations.settings import AdsmodConfig, DatabaseConfig, UserSettings
from server.app import create_app


# Core routes


###############################################################################
def _config(tmp_path: Path):  # type: ignore[no-untyped-def]
    return AdsmodConfig.assemble(
        database=DatabaseConfig(embedded_database=True, sqlite_path="data/test.db"),
        storage_root=tmp_path,
        settings=UserSettings(),
    )


###############################################################################
def test_unified_routes_expose_core_optional_ml_and_public_data_surface(
    tmp_path: Path,
) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        assert client.get("/health/live").json()["service"] == "backend"
        assert client.get("/health/ready").json()["state"] == "ready"
        capabilities = client.get("/api/v1/system/capabilities").json()
        assert capabilities["features"]["datasets"] is True
        assert capabilities["features"]["machine_learning"] is True, (
            client.app.state.runtime.machine_learning_reason
        )
        assert client.get("/api/v1/datasets").status_code == 200
        dataset_configuration = client.get("/api/v1/datasets/supported-units")
        assert dataset_configuration.status_code == 200
        assert dataset_configuration.json()["allowed_extensions"] == [
            ".csv",
            ".xls",
            ".xlsx",
        ]
        assert client.get("/api/v1/training/configuration").status_code == 200
        sources = client.get("/api/v1/public-data/sources")
        assert sources.status_code == 200
        assert {item["key"] for item in sources.json()["sources"]} == {
            "nist",
            "pubchem",
            "cod",
        }
        assert all(
            item["health_status"] == "unknown" for item in sources.json()["sources"]
        )
        assert client.get("/api/health").status_code == 404


###############################################################################
def test_unified_openapi_surface_contains_public_data_routes(tmp_path: Path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        paths = client.get("/openapi.json").json()["paths"]
        assert "/api/v1/datasets" in paths
        assert "/api/v1/fitting/models" in paths
        assert "/api/v1/training/configuration" in paths
        assert "/api/v1/public-data/sources" in paths
        assert "/api/v1/public-data/sources/{source_key}/health-check" in paths
        assert "/api/v1/public-data/sync-runs" in paths
        assert "/api/v1/public-data/sync-runs/{run_id}" in paths
        assert "/api/v1/public-data/sync-runs/{run_id}/cancel" in paths
        assert "/api/v1/public-data/sync-runs/{run_id}/resume" in paths
        assert "/api/v1/public-data/adsorption" in paths
        assert "/api/v1/public-data/structures/search" in paths
        assert "/api/v1/public-data/sources/nist/collections" in paths
        assert "/api/v1/public-data/materials/{adsorbent_id}" in paths
        assert "put" in paths["/api/v1/public-data/structures/{structure_id}/material"]
        assert "/api/v1/public-data/structures/{structure_id}/file" in paths
        assert "post" in paths["/api/v1/public-data/sync-runs/{run_id}/cancel"]
        assert "post" in paths["/api/v1/public-data/sync-runs/{run_id}/resume"]
        # Acquisition runs only through durable sync runs.
        assert "/api/v1/public-data/chemicals/resolve" not in paths
        assert "/api/v1/public-data/structures/import" not in paths
        assert "delete" not in paths["/api/v1/public-data/sync-runs/{run_id}"]
        assert not any(path.startswith("/api/v1/nist") for path in paths)


# Ml routes


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
