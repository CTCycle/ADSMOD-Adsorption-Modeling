# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from server.configurations.settings import (
    AdsmodConfig,
    DatabaseConfig,
    DatasetConfig,
    FittingConfig,
    JobConfig,
    PublicDataConfig,
    RuntimeConfig,
    TrainingConfig,
    UserSettings,
)


def test_application_settings_defaults_cover_runtime_behaviors() -> None:
    settings = UserSettings()

    assert settings.runtime.backend_port == 6045
    assert settings.runtime.frontend_port == 5173
    assert settings.datasets.allowed_extensions == (".csv", ".xls", ".xlsx")
    assert settings.datasets.column_detection_cutoff == 0.6
    assert settings.public_data.pubchem_parallel_requests == 2
    assert settings.public_data.cod_max_interactive_results == 250
    assert settings.public_data.sync_max_records_per_run == 200
    assert settings.public_data.auto_sync_enabled is False
    assert settings.fitting.preview_row_limit == 20
    assert settings.fitting.best_model_metric == "aicc"
    assert settings.training.persistent_workers is False


def test_settings_validation_checks_dependent_fields_together() -> None:
    with pytest.raises(ValidationError, match="distinct"):
        RuntimeConfig(backend_port=6045, frontend_port=6045)
    with pytest.raises(ValidationError, match="at least one NIST collection"):
        PublicDataConfig(auto_sync_enabled=True)
    with pytest.raises(ValidationError, match="between the minimum and maximum"):
        FittingConfig(default_parameter_min=2, default_parameter_initial=1)
    with pytest.raises(ValidationError, match="greater than"):
        FittingConfig(default_parameter_min=2, default_parameter_max=2, default_parameter_initial=2)
    with pytest.raises(ValidationError, match="greater than zero"):
        TrainingConfig(persistent_workers=True, dataloader_workers=0)
    with pytest.raises(ValidationError):
        FittingConfig.model_validate({"best_model_metric": "unknown"})


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
