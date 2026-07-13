import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from adsmod_common.capabilities import CapabilitiesResponse
from adsmod_common.config import AdsmodConfig, load_config


CONFIG_PATH = Path("settings/adsmod.json")


###############################################################################
def test_canonical_core_config_loads() -> None:
    config = load_config(CONFIG_PATH)
    assert config.version == "3.0.0"
    assert config.runtime.mode == "core"


###############################################################################
def test_core_ml_config_loads() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["runtime"]["mode"] = "core-ml"
    payload["runtime"]["ml_restart_attempts"] = 1
    assert AdsmodConfig.model_validate(payload).runtime.mode == "core-ml"


###############################################################################
def test_missing_canonical_sections_are_rejected() -> None:
    config_path = Path("settings/_test_invalid_config.json")
    config_path.write_text(json.dumps({"version": "3.0.0"}), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="missing required sections"):
            load_config(config_path)
    finally:
        config_path.unlink(missing_ok=True)


###############################################################################
def test_unknown_and_legacy_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        AdsmodConfig.model_validate({"mode": "both"})


###############################################################################
def test_duplicate_ports_are_rejected() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["runtime"]["ml_port"] = payload["runtime"]["core_port"]
    with pytest.raises(ValidationError):
        AdsmodConfig.model_validate(payload)


###############################################################################
def test_capability_contract_is_strict() -> None:
    response = CapabilitiesResponse.model_validate({
        "configured_mode": "core",
        "version": "3.0.0",
        "features": {"datasets": True, "nist": True, "fitting": True, "training": False, "checkpoints": False},
        "services": {"ml": {"configured": False, "health": "unavailable", "readiness": "not-configured"}},
    })
    assert response.features.training is False
