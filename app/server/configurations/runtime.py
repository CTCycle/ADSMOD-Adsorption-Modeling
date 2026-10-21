# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from server.configurations.bootstrap import (
    load_database_configuration,
    load_runtime_settings,
    load_storage_root,
    persist_database_configuration,
)
from server.configurations.settings import AdsmodConfig, DatabaseConfig, UserSettings
from server.configurations.store import ApplicationSettingsStore
from server.repositories.database.initializer import prepare_database_for_startup
from server.repositories.database.manager import DatabaseManager
from server.common.utils.logger import logger


###############################################################################
@dataclass(frozen=True)
class RuntimeConfiguration:
    config: AdsmodConfig
    settings_store: ApplicationSettingsStore
    database_ready: bool = False


###############################################################################
def load_runtime_configuration(
    *,
    database: DatabaseConfig | None = None,
    storage_root: Path | None = None,
    initialize_database: bool = True,
) -> RuntimeConfiguration:
    resolved_storage_root = (storage_root or load_storage_root()).resolve()
    resolved_storage_root.mkdir(parents=True, exist_ok=True)
    runtime_settings = load_runtime_settings()
    database_load = (
        None
        if database is not None
        else load_database_configuration(storage_root=resolved_storage_root)
    )
    bootstrap_database = database or database_load.config
    manager = DatabaseManager(bootstrap_database, storage_root=resolved_storage_root)
    database_ready = False
    if initialize_database:
        try:
            prepare_database_for_startup(
                bootstrap_database, storage_root=resolved_storage_root
            )
            database_ready = True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Configured database is unavailable during startup (%s); the Database settings page remains available.",
                type(exc).__name__,
            )
    settings_store = ApplicationSettingsStore(manager)
    try:
        defaults = UserSettings()
        if database_ready:
            try:
                user_settings = settings_store.initialize(defaults)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Application settings could not be loaded from the configured database (%s).",
                    type(exc).__name__,
                )
                user_settings = defaults
                database_ready = False
        else:
            user_settings = defaults
        if database_load is not None and database_load.needs_persist and database_ready:
            persist_database_configuration(bootstrap_database, resolved_storage_root)
        return RuntimeConfiguration(
            config=AdsmodConfig.assemble(
                database=bootstrap_database,
                storage_root=resolved_storage_root,
                settings=user_settings,
                runtime=runtime_settings,
            ),
            settings_store=settings_store,
            database_ready=database_ready,
        )
    except Exception:
        manager.dispose()
        raise


__all__ = ["RuntimeConfiguration", "load_runtime_configuration"]
