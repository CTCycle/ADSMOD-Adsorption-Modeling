# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from scripts.generate_openapi import build_openapi_schema
from server.domain.capabilities import CapabilitiesResponse
from server.configurations.settings import (
    RuntimeConfig,
    UserSettings,
)


###############################################################################
def test_runtime_environment_settings_are_separate_from_user_settings() -> None:
    settings = UserSettings()
    runtime = RuntimeConfig()
    assert runtime.backend_port != runtime.frontend_port
    assert runtime.host == "127.0.0.1"
    assert not hasattr(settings, "runtime")


###############################################################################
def test_unsupported_runtime_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate({"mode": "core-ml"})


###############################################################################
def test_duplicate_backend_and_frontend_ports_are_rejected() -> None:
    with pytest.raises(ValidationError):
        RuntimeConfig(backend_port=6045, frontend_port=6045)


###############################################################################
def test_capability_contract_is_strict() -> None:
    response = CapabilitiesResponse.model_validate(
        {
            "version": "3.0.0",
            "features": {
                "datasets": True,
                "nist": True,
                "fitting": True,
                "machine_learning": False,
                "training": False,
                "checkpoints": False,
            },
        }
    )
    assert response.features.machine_learning is False


###############################################################################
def test_openapi_generation_fails_closed_without_ml_profile(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    application = SimpleNamespace(
        state=SimpleNamespace(runtime=SimpleNamespace(machine_learning_available=False))
    )
    settings_store = SimpleNamespace(database=SimpleNamespace(dispose=lambda: None))
    monkeypatch.setattr(
        "scripts.generate_openapi.create_app", lambda config, **kwargs: application
    )
    monkeypatch.setattr(
        "scripts.generate_openapi.load_runtime_configuration",
        lambda **kwargs: SimpleNamespace(
            config=object(), settings_store=settings_store
        ),
    )

    with pytest.raises(RuntimeError, match="requires the ML-enabled backend profile"):
        build_openapi_schema()
