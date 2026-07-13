from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from adsmod_common.config import StorageConfig, load_config
from adsmod_core.app import create_app, create_app_from_path


CONFIG_PATH = Path("settings/adsmod.json")

###############################################################################
def test_core_runtime_contracts() -> None:
    with TestClient(create_app(load_config(CONFIG_PATH))) as client:
        assert client.get("/health/live").json()["state"] == "ready"
        assert client.get("/health/ready").json()["state"] == "ready"
        capabilities = client.get("/api/v1/system/capabilities").json()
        assert capabilities["configured_mode"] == "core"
        assert capabilities["features"]["training"] is False
        assert capabilities["services"]["ml"]["readiness"] == "not-configured"

###############################################################################
def test_core_factory_requires_explicit_config() -> None:
    application = create_app_from_path(CONFIG_PATH)
    assert application.state.config.runtime.mode == "core"

###############################################################################
def test_snapshot_api_requires_token_and_preserves_hash() -> None:
    with TemporaryDirectory(dir="assets/QA") as directory:
        config = load_config(CONFIG_PATH).model_copy(
            update={"storage": StorageConfig(root=Path(directory), database="core.db")}
        )
        with TestClient(create_app(config, internal_token="secret")) as client:
            rows = [{"id": 1, "value": "alpha"}, {"id": 2, "value": "beta"}]
            unauthorized = client.post("/api/v1/internal/snapshots", json={"rows": rows})
            assert unauthorized.status_code == 401
            created = client.post(
                "/api/v1/internal/snapshots",
                headers={"X-ADSMOD-Internal-Token": "secret"},
                json={"rows": rows},
            )
            assert created.status_code == 200
            metadata = created.json()
            rows[0]["value"] = "mutated-after-create"
            page = client.get(
                f"/api/v1/internal/snapshots/{metadata['snapshot_id']}?page=1&page_size=1",
                headers={"X-ADSMOD-Internal-Token": "secret"},
            )
            assert page.status_code == 200
            payload = page.json()
            assert payload["total_rows"] == 2
            assert payload["rows"] == [{"id": 1, "value": "alpha"}]
            assert payload["content_hash"] == metadata["content_hash"]

###############################################################################
def test_snapshot_page_not_found() -> None:
    with TemporaryDirectory(dir="assets/QA") as directory:
        config = load_config(CONFIG_PATH).model_copy(
            update={"storage": StorageConfig(root=Path(directory), database="core.db")}
        )
        with TestClient(create_app(config, internal_token="secret")) as client:
            response = client.get(
                "/api/v1/internal/snapshots/not-a-snapshot",
                headers={"X-ADSMOD-Internal-Token": "secret"},
            )
            assert response.status_code == 404