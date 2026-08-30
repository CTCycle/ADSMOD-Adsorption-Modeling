from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from adsmod_common.config import StorageConfig, load_config
from adsmod_core.app import create_app


CONFIG_PATH = Path("app/resources/adsmod.json")


def _config(tmp_path: Path):
    base = load_config(CONFIG_PATH)
    return base.model_copy(
        update={
            "storage": StorageConfig(root=tmp_path),
            "application": base.application.model_copy(
                update={
                    "database": base.application.database.model_copy(
                        update={"sqlite_path": "data/test.db"}
                    )
                }
            ),
        }
    )


def test_core_routes_use_versioned_api_and_exclude_training(tmp_path: Path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200
        assert client.get("/api/v1/system/capabilities").status_code == 200
        assert client.get("/api/v1/datasets").status_code == 200
        assert client.get("/api/v1/training/status").status_code == 404
        assert client.get("/api/health").status_code == 404


def test_core_openapi_surface_has_no_training_routes(tmp_path: Path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        schema = client.get("/openapi.json").json()
        assert "/api/v1/datasets" in schema["paths"]
        assert "/api/v1/fitting/models" in schema["paths"]
        assert not any(path.startswith("/api/v1/training") for path in schema["paths"])
