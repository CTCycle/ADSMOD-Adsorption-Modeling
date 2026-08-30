from __future__ import annotations

from pathlib import Path

from adsmod_common.config import TrainingConfig, load_config

CANONICAL_CONFIGURATION_FILE = Path("app/resources/adsmod.json")


###############################################################################
def test_json_training_configuration_projects_from_canonical_model() -> None:
    config = load_config(CANONICAL_CONFIGURATION_FILE)

    assert config.application.training.persistent_workers is False
    assert config.application.datasets.allowed_extensions == (
        ".csv",
        ".xls",
        ".xlsx",
    )


###############################################################################
def test_training_configuration_values_are_validated_by_canonical_model() -> None:
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
def test_canonical_training_defaults_are_single_source() -> None:
    training = TrainingConfig.model_validate({"persistent_workers": False})

    assert training.use_jit is False
    assert training.jit_backend == "inductor"
    assert training.use_mixed_precision is False
    assert training.dataloader_workers == 0
    assert training.persistent_workers is False


###############################################################################
def test_canonical_runtime_configuration_validates() -> None:
    config = load_config(Path(CANONICAL_CONFIGURATION_FILE))

    assert config.version == "3.0.0"
    assert config.application.datasets.allowed_extensions
    assert config.application.jobs.polling_interval > 0
