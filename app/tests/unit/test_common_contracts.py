import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from scripts.generate_openapi import build_openapi_schema
from server.domain.capabilities import CapabilitiesResponse
from server.configurations.settings import AdsmodConfig, load_config

CONFIG_PATH = Path("resources/adsmod.json")

###############################################################################
def test_canonical_config_loads_single_backend_runtime() -> None:
    config = load_config(CONFIG_PATH)
    assert config.version == "3.0.0"
    assert config.runtime.backend_port != config.runtime.frontend_port
    assert not hasattr(config.runtime, "mode")
    assert not hasattr(config.runtime, "ml_port")

###############################################################################
def test_legacy_dual_backend_runtime_keys_are_rejected() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["runtime"]["mode"] = "core-ml"
    with pytest.raises(ValidationError):
        AdsmodConfig.model_validate(payload)

###############################################################################
def test_duplicate_backend_and_frontend_ports_are_rejected() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["runtime"]["frontend_port"] = payload["runtime"]["backend_port"]
    with pytest.raises(ValidationError):
        AdsmodConfig.model_validate(payload)

###############################################################################
def test_capability_contract_is_strict() -> None:
    response = CapabilitiesResponse.model_validate({
        "version": "3.0.0",
        "features": {
            "datasets": True,
            "nist": True,
            "fitting": True,
            "machine_learning": False,
            "training": False,
            "checkpoints": False,
        },
    })
    assert response.features.machine_learning is False


###############################################################################
def test_openapi_generation_fails_closed_without_ml_profile(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    application = SimpleNamespace(
        state=SimpleNamespace(
            runtime=SimpleNamespace(machine_learning_available=False)
        )
    )
    monkeypatch.setattr("scripts.generate_openapi.create_app", lambda config: application)

    with pytest.raises(RuntimeError, match="requires the ML-enabled backend profile"):
        build_openapi_schema(CONFIG_PATH)
