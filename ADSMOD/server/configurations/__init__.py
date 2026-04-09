from __future__ import annotations

from ADSMOD.server.configurations.bootstrap import ensure_environment_loaded
from ADSMOD.server.configurations.base import ensure_mapping, load_configuration_data
from ADSMOD.server.configurations.server import (
    AppSettings,
    DatabaseSettings,
    DatasetSettings,
    FittingSettings,
    JobSettings,
    NISTSettings,
    ServerSettings,
    TrainingSettings,
    build_database_settings,
    build_dataset_settings,
    build_fitting_settings,
    build_job_settings,
    build_nist_settings,
    build_server_settings,
    build_training_settings,
    get_app_settings,
    get_server_settings,
    reload_settings_for_tests,
    server_settings,
)


ensure_environment_loaded()

# Backward-compatible alias used by existing tests/imports.
load_configurations = load_configuration_data

__all__ = [
    "ensure_environment_loaded",
    "ensure_mapping",
    "load_configuration_data",
    "load_configurations",
    "AppSettings",
    "DatabaseSettings",
    "DatasetSettings",
    "FittingSettings",
    "JobSettings",
    "NISTSettings",
    "ServerSettings",
    "TrainingSettings",
    "build_database_settings",
    "build_dataset_settings",
    "build_fitting_settings",
    "build_job_settings",
    "build_nist_settings",
    "build_server_settings",
    "build_training_settings",
    "get_app_settings",
    "get_server_settings",
    "reload_settings_for_tests",
    "server_settings",
]
