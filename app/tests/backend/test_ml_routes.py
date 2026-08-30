from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from adsmod_common.config import StorageConfig, load_config
from adsmod_ml.app import create_app


CONFIG_PATH = Path("app/resources/adsmod.json")


def _config(tmp_path: Path):
    base = load_config(CONFIG_PATH)
    runtime = base.runtime.model_copy(
        update={"mode": "core-ml", "ml_restart_attempts": 1}
    )
    return base.model_copy(
        update={
            "runtime": runtime,
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


def test_ml_routes_use_versioned_training_api(tmp_path: Path) -> None:
    with TestClient(create_app(_config(tmp_path), internal_token="secret")) as client:
        assert client.get("/health/live").json()["service"] == "ml"
        assert client.get("/health/ready").json()["state"] == "ready"
        assert client.get("/api/v1/system/capabilities").status_code == 200
        assert client.get("/api/v1/training/configuration").status_code == 200
        assert client.get("/api/v1/training/status").status_code == 200
        assert client.get("/api/v1/system/configuration").status_code == 404
        assert client.get("/api/health").status_code == 404
