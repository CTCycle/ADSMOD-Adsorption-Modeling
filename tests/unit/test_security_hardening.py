import os
import tempfile

import pytest
from pydantic import ValidationError

from ADSMOD.server.app import cloud_mode_enabled, resolve_spa_file_path
from ADSMOD.server.entities.fitting import DatasetPayload
from ADSMOD.server.entities.training import TrainingConfigRequest


def test_resolve_spa_file_path_rejects_traversal() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = os.path.join(temp_dir, "index.html")
        with open(index_path, "w", encoding="utf-8") as handle:
            handle.write("ok")

        assert resolve_spa_file_path(temp_dir, "index.html") == os.path.abspath(
            index_path
        )
        assert resolve_spa_file_path(temp_dir, "..\\secrets.txt") is None


def test_cloud_mode_enabled_detects_non_loopback_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTAPI_HOST", "0.0.0.0")
    assert cloud_mode_enabled() is True

    monkeypatch.setenv("FASTAPI_HOST", "127.0.0.1")
    assert cloud_mode_enabled() is False


def test_dataset_payload_rejects_unsafe_dataset_name() -> None:
    with pytest.raises(ValidationError):
        DatasetPayload(
            dataset_name="../../etc/passwd",
            columns=["temperature"],
            records=[{"temperature": 300}],
        )


def test_training_config_rejects_invalid_dataset_hash() -> None:
    with pytest.raises(ValidationError):
        TrainingConfigRequest(dataset_hash="not_a_sha256")
