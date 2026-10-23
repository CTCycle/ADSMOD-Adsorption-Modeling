# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import pytest
import os
import server.runtime_config as runtime_config
from pydantic import ValidationError
from scripts.generate_openapi import build_openapi_schema
from server.domain.capabilities import CapabilitiesResponse
from server.configurations.settings import (
    RuntimeConfig,
    UserSettings,
    AdsmodConfig,
    DatabaseConfig,
    DatasetConfig,
    FittingConfig,
    JobConfig,
    PublicDataConfig,
    TrainingConfig,
)
from server.configurations.bootstrap import (
    DatabaseConfigurationLoad,
    RETIRED_DATABASE_KEYS,
    load_database_configuration,
)
from server.configurations.database_config import database_config_path
from server.configurations.runtime import load_runtime_configuration
from server.runtime_config import build_runtime_record
from server.repositories.database.manager import resolve_sqlite_path
from server.common.path import resolve_database_path


# Common contracts


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


# Configurations


###############################################################################
def test_application_settings_defaults_and_runtime_defaults_are_separate() -> None:
    settings = UserSettings()
    runtime = RuntimeConfig()

    assert runtime.host == "127.0.0.1"
    assert runtime.backend_port == 6045
    assert runtime.frontend_port == 5173
    assert runtime.cors_origins == ("http://127.0.0.1:5173", "http://localhost:5173")
    assert not hasattr(settings, "runtime")
    assert settings.datasets.allowed_extensions == (".csv", ".xls", ".xlsx")
    assert settings.datasets.column_detection_cutoff == 0.6
    assert settings.public_data.pubchem_parallel_requests == 2
    assert settings.public_data.cod_max_interactive_results == 250
    assert settings.public_data.sync_max_records_per_run == 200
    assert settings.public_data.auto_sync_enabled is False
    assert settings.fitting.preview_row_limit == 20
    assert settings.fitting.best_model_metric == "aicc"
    assert settings.training.persistent_workers is False


###############################################################################
def test_settings_validation_checks_dependent_fields_together() -> None:
    with pytest.raises(ValidationError, match="distinct"):
        RuntimeConfig(backend_port=6045, frontend_port=6045)
    with pytest.raises(ValidationError, match="at least one NIST collection"):
        PublicDataConfig(auto_sync_enabled=True)
    with pytest.raises(ValidationError, match="between the minimum and maximum"):
        FittingConfig(default_parameter_min=2, default_parameter_initial=1)
    with pytest.raises(ValidationError, match="greater than"):
        FittingConfig(
            default_parameter_min=2,
            default_parameter_max=2,
            default_parameter_initial=2,
        )
    with pytest.raises(ValidationError, match="greater than zero"):
        TrainingConfig(persistent_workers=True, dataloader_workers=0)
    with pytest.raises(ValidationError):
        FittingConfig.model_validate({"best_model_metric": "unknown"})


###############################################################################
def test_database_bootstrap_is_separate_from_user_settings() -> None:
    database = DatabaseConfig()
    settings = UserSettings()
    config = AdsmodConfig.assemble(
        database=database,
        storage_root=Path("/tmp/adsmod"),
        settings=settings,
    )

    assert database.embedded_database is True
    assert database.sqlite_path == "data/database.db"
    assert config.user_settings() == settings
    assert not hasattr(database, "insert_batch_size")


###############################################################################
def test_training_settings_are_typed_and_validated() -> None:
    training = TrainingConfig.model_validate(
        {
            "use_jit": True,
            "jit_backend": "cudagraphs",
            "use_mixed_precision": True,
            "dataloader_workers": 4,
            "persistent_workers": True,
        }
    )

    assert training.use_jit is True
    assert training.jit_backend == "cudagraphs"
    assert training.use_mixed_precision is True
    assert training.dataloader_workers == 4
    assert training.persistent_workers is True


###############################################################################
def test_public_data_policy_is_bounded_and_categories_are_checked() -> None:
    config = PublicDataConfig()

    assert config.request_timeout_seconds == 20.0
    assert config.retry_attempts == 3
    assert config.pubchem_parallel_requests == 2
    assert config.cod_max_interactive_results == 250
    assert config.health_cache_ttl_seconds == 900
    assert config.sync_batch_size == 50
    assert config.sync_max_records_per_run == 200
    assert config.sync_max_concurrent_jobs == 1
    assert config.sync_resume_on_startup is False

    scheduled = PublicDataConfig(
        auto_sync_enabled=True,
        auto_sync_categories=("guest", "host"),
        auto_sync_interval_hours=12,
        auto_sync_fraction=0.5,
    )
    assert scheduled.auto_sync_categories == ("guest", "host")
    assert scheduled.auto_sync_interval_hours == 12


###############################################################################
def test_supported_extensions_and_job_polling_are_validated() -> None:
    assert DatasetConfig(allowed_extensions=("csv", ".XLSX")).allowed_extensions == (
        ".csv",
        ".xlsx",
    )
    with pytest.raises(ValidationError):
        DatasetConfig(allowed_extensions=())
    with pytest.raises(ValidationError):
        JobConfig(polling_interval=0)
    assert JobConfig(polling_interval=0.5).polling_interval == 0.5


# Runtime config


###############################################################################
def test_runtime_endpoint_record_uses_environment_endpoints_and_database_bootstrap(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    database = DatabaseConfig(
        embedded_database=True,
        sqlite_path=":memory:",
    )
    endpoints = RuntimeConfig(host="::1", backend_port=6046, frontend_port=5174)
    monkeypatch.setattr(runtime_config, "load_runtime_settings", lambda: endpoints)
    monkeypatch.setattr(runtime_config, "load_storage_root", lambda: tmp_path)
    monkeypatch.setattr(
        runtime_config,
        "load_database_configuration",
        lambda **_: DatabaseConfigurationLoad(config=database, needs_persist=False),
    )
    monkeypatch.setattr(
        runtime_config,
        "resolve_sqlite_path",
        lambda *_args, **_kwargs: tmp_path / "database.db",
    )

    record = build_runtime_record()

    assert record["host"] == "::1"
    assert record["backend_port"] == 6046
    assert record["frontend_port"] == 5174
    assert record["backend_url"] == "http://[::1]:6046"
    assert record["frontend_url"] == "http://[::1]:5174"
    assert record["database_backend"] == "sqlite"
    assert record["database_path"] is None


###############################################################################
def test_first_runtime_startup_seeds_defaults_and_persists_current_config(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    env_path = tmp_path / ".env"
    env_path.write_text(
        "DATABASE_HOST=retired-host\nDATABASE_PASSWORD=retired-secret\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "server.configurations.runtime.load_database_configuration",
        lambda **kwargs: load_database_configuration(**kwargs, dotenv_path=env_path),
    )
    monkeypatch.setenv("DATABASE_PASSWORD", "process-secret")

    runtime = load_runtime_configuration(storage_root=tmp_path)
    try:
        assert runtime.database_ready is True
        assert runtime.settings_store.get() == UserSettings()
        loaded = load_database_configuration(
            dotenv_path=env_path, storage_root=tmp_path
        )
        assert loaded.needs_persist is False
        assert loaded.config == runtime.config.application.database
        assert "DATABASE_" not in env_path.read_text(encoding="utf-8")
        assert all(key not in os.environ for key in RETIRED_DATABASE_KEYS)
        assert database_config_path(tmp_path).is_file()
    finally:
        runtime.settings_store.database.dispose()


###############################################################################
def test_openapi_runtime_configuration_does_not_create_partial_database(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "not-created.db"
    runtime = load_runtime_configuration(
        database=DatabaseConfig(embedded_database=True, sqlite_path=str(database_path)),
        storage_root=tmp_path,
        initialize_database=False,
    )
    try:
        assert runtime.database_ready is False
        assert runtime.config.user_settings() == UserSettings()
        assert not database_path.exists()
        assert not database_config_path(tmp_path).exists()
    finally:
        runtime.settings_store.database.dispose()


# Data paths


###############################################################################
def test_relative_database_path_follows_the_canonical_storage_root(
    tmp_path: Path,
) -> None:
    database = DatabaseConfig(embedded_database=True, sqlite_path="data/database.db")

    assert (
        resolve_sqlite_path(database, storage_root=tmp_path)
        == (tmp_path / "data" / "database.db").resolve()
    )

    config = AdsmodConfig.assemble(
        database=database,
        storage_root=tmp_path,
        settings=UserSettings(),
    )
    assert (
        resolve_database_path(config) == (tmp_path / "data" / "database.db").resolve()
    )
