from __future__ import annotations

import json
from pathlib import Path

from core_service.configurations.management import ConfigurationManager
from core_service.domain.settings import ServerSettings
from ml_service.configurations.management import (
    ConfigurationManager as MlConfigurationManager,
)
from shared.common.paths import CORE_CONFIGURATION_FILE, ML_CONFIGURATION_FILE


###############################################################################
def write_config(path: str, payload: dict[str, object]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


###############################################################################
def build_payload() -> dict[str, object]:
    return {
        "database": {"embedded_database": True},
        "datasets": {
            "allowed_extensions": [".csv", ".xlsx"],
            "column_detection_cutoff": 0.65,
        },
        "nist": {"parallel_tasks": 4, "pubchem_parallel_tasks": 2},
        "fitting": {"default_max_iterations": 1000, "preview_row_limit": 5},
        "jobs": {"polling_interval": 1.0},
        "training": {"use_jit": False, "dataloader_workers": 0},
    }


###############################################################################
def test_manager_load_and_accessors(tmp_path) -> None:
    config_path = tmp_path / "configurations.json"
    write_config(str(config_path), build_payload())

    manager = ConfigurationManager(config_path=str(config_path))
    manager.load()

    block = manager.get_block("training")
    assert block.get("use_jit") is False
    assert manager.get_value("jobs", "polling_interval") == 1.0
    assert manager.get_value("jobs", "missing", 9) == 9


###############################################################################
def test_manager_reload_reflects_file_changes(tmp_path) -> None:
    config_path = tmp_path / "configurations.json"
    payload = build_payload()
    write_config(str(config_path), payload)

    manager = ConfigurationManager(config_path=str(config_path))
    manager.load()

    payload["jobs"] = {"polling_interval": 2.5}
    write_config(str(config_path), payload)
    reloaded = manager.reload()

    assert reloaded.jobs.polling_interval == 2.5


###############################################################################
def test_manager_update_persists_and_reloads(tmp_path) -> None:
    config_path = tmp_path / "configurations.json"
    write_config(str(config_path), build_payload())

    manager = ConfigurationManager(config_path=str(config_path))
    manager.load()
    updated = manager.update({"jobs": {"polling_interval": 3.0}})

    assert updated.jobs.polling_interval == 3.0
    assert manager.get_value("jobs", "polling_interval") == 3.0


###############################################################################
def test_manager_to_server_settings(tmp_path) -> None:
    config_path = tmp_path / "configurations.json"
    write_config(str(config_path), build_payload())

    manager = ConfigurationManager(config_path=str(config_path))
    server_settings = manager.to_server_settings()

    assert isinstance(server_settings, ServerSettings)
    assert server_settings.datasets.column_detection_cutoff == 0.65


###############################################################################
def test_core_manager_defaults_to_core_runtime_file() -> None:
    manager = ConfigurationManager()
    assert manager.config_path == CORE_CONFIGURATION_FILE


###############################################################################
def test_ml_manager_defaults_to_ml_runtime_file() -> None:
    manager = MlConfigurationManager()
    assert manager.config_path == ML_CONFIGURATION_FILE


