# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from scripts.generate_openapi import build_openapi_schema
from server.domain.capabilities import CapabilitiesResponse
from server.configurations.settings import (
    AdsmodConfig,
    DatabaseConfig,
    RuntimeConfig,
    UserSettings,
)

###############################################################################
def test_typed_settings_define_one_backend_runtime() -> None:
    settings = UserSettings()
    assert settings.runtime.backend_port != settings.runtime.frontend_port
    assert not hasattr(settings.runtime, "mode")
    assert not hasattr(settings.runtime, "ml_port")

###############################################################################
def test_legacy_dual_backend_runtime_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate({"mode": "core-ml"})

###############################################################################
def test_duplicate_backend_and_frontend_ports_are_rejected() -> None:
    with pytest.raises(ValidationError):
        RuntimeConfig(backend_port=6045, frontend_port=6045)

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
def test_openapi_generation_fails_closed_without_ml_profile(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    application = SimpleNamespace(
        state=SimpleNamespace(
            runtime=SimpleNamespace(machine_learning_available=False)
        )
    )
    monkeypatch.setattr("scripts.generate_openapi.create_app", lambda config: application)
    monkeypatch.setattr("scripts.generate_openapi.load_database_config", DatabaseConfig)
    monkeypatch.setattr("scripts.generate_openapi.load_storage_root", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="requires the ML-enabled backend profile"):
        build_openapi_schema()
