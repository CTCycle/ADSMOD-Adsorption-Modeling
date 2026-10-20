# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from server.configurations.bootstrap import (
    load_database_config,
    load_storage_root,
    resolve_settings_database_path,
)
from server.configurations.settings import AdsmodConfig, DatabaseConfig, UserSettings
from server.configurations.store import ApplicationSettingsStore
from server.repositories.database.initializer import prepare_database_for_startup


@dataclass(frozen=True)
class RuntimeConfiguration:
    config: AdsmodConfig
    settings_store: ApplicationSettingsStore


def load_runtime_configuration(
    *,
    database: DatabaseConfig | None = None,
    storage_root: Path | None = None,
    initialize_database: bool = True,
) -> RuntimeConfiguration:
    bootstrap_database = database or load_database_config()
    resolved_storage_root = (storage_root or load_storage_root()).resolve()
    resolved_storage_root.mkdir(parents=True, exist_ok=True)
    if initialize_database:
        prepare_database_for_startup(
            bootstrap_database,
            storage_root=resolved_storage_root,
        )
    settings_store = ApplicationSettingsStore(
        resolve_settings_database_path(bootstrap_database, resolved_storage_root)
    )
    user_settings = settings_store.initialize(UserSettings())
    return RuntimeConfiguration(
        config=AdsmodConfig.assemble(
            database=bootstrap_database,
            storage_root=resolved_storage_root,
            settings=user_settings,
        ),
        settings_store=settings_store,
    )


__all__ = ["RuntimeConfiguration", "load_runtime_configuration"]
