import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from shared.common.utils.security import resolve_checkpoint_path
from core_service.configurations import resolve_spa_file_path
from core_service.domain.fitting import DatasetPayload
from ml_service.domain.training import TrainingConfigRequest

###############################################################################
def test_resolve_spa_file_path_rejects_traversal() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = Path(temp_dir) / "index.html"
        with index_path.open("w", encoding="utf-8") as handle:
            handle.write("ok")

        assert resolve_spa_file_path(temp_dir, "index.html") == str(index_path.resolve())
        assert resolve_spa_file_path(temp_dir, "..\\secrets.txt") is None
        assert resolve_spa_file_path(Path(temp_dir), "../secrets.txt") is None

###############################################################################
def test_resolve_checkpoint_path_accepts_path_inputs() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        base_path = Path(temp_dir)
        resolved = resolve_checkpoint_path(base_path, "checkpoint_01")

        assert resolved == str((base_path / "checkpoint_01").resolve())

###############################################################################
def test_dataset_payload_rejects_unsafe_dataset_name() -> None:
    with pytest.raises(ValidationError):
        DatasetPayload(
            dataset_name="../../etc/passwd",
            columns=["temperature"],
            records=[{"temperature": 300}],
        )

###############################################################################
def test_training_config_rejects_invalid_dataset_hash() -> None:
    with pytest.raises(ValidationError):
        TrainingConfigRequest(dataset_hash="not_a_sha256")
