from pathlib import Path

from fastapi.testclient import TestClient

from adsmod_common.config import load_config
from adsmod_core.app import create_app, create_app_from_path


CONFIG_PATH = Path("settings/adsmod.json")


def test_core_runtime_contracts() -> None:
    client = TestClient(create_app(load_config(CONFIG_PATH)))

    assert client.get("/health/live").json()["state"] == "ready"
    assert client.get("/health/ready").json()["state"] == "ready"

    capabilities = client.get("/api/v1/system/capabilities").json()
    assert capabilities["configured_mode"] == "core"
    assert capabilities["features"]["training"] is False
    assert capabilities["services"]["ml"]["readiness"] == "not-configured"


def test_core_factory_requires_explicit_config() -> None:
    application = create_app_from_path(CONFIG_PATH)
    assert application.state.config.runtime.mode == "core"