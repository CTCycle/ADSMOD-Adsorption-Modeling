from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pandas as pd
import pytest
from pydantic import ValidationError

from adsmod_ml.clients.core_client import CoreSnapshotClient
from adsmod_ml.contracts.training import TrainingMetadata
from adsmod_ml.learning.serialization.model import ModelSerializer
from adsmod_ml.learning.serialization.training import TrainingDataSerializer


def create_basis_metadata(**kwargs: object) -> TrainingMetadata:
    defaults: dict[str, object] = {
        "sample_size": 1.0,
        "validation_size": 0.2,
        "min_measurements": 1,
        "max_measurements": 30,
        "smile_sequence_size": 20,
        "max_pressure": 10000.0,
        "max_uptake": 20.0,
        "smile_vocabulary": {"C": 1, "H": 2, "O": 3},
        "adsorbent_vocabulary": {"MOF-1": 1, "ZEOLITE-X": 2},
        "normalization_stats": {"pressure_mean": 5000.0, "pressure_std": 2000.0},
    }
    defaults.update(kwargs)
    return TrainingMetadata(**defaults)


def _serializer(
    tmp_path: Path,
    *,
    response_rows: list[dict[str, object]] | None = None,
) -> tuple[TrainingDataSerializer, list[dict[str, object]]]:
    captured: list[dict[str, object]] = []
    rows = response_rows or [{"dataset_label": "default", "split": "train"}]
    content = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            payload = json.loads(request.content)
            captured.append(payload)
            return httpx.Response(
                200,
                json={
                    "snapshot_id": "snapshot-1",
                    "content_hash": content_hash,
                    "row_count": len(payload["rows"]),
                },
            )
        return httpx.Response(
            200,
            json={
                "snapshot_id": "snapshot-1",
                "content_hash": content_hash,
                "page": 1,
                "page_size": 1000,
                "total_rows": len(rows),
                "rows": rows,
            },
        )

    client = CoreSnapshotClient(
        "http://core",
        "secret",
        httpx.MockTransport(handler),
    )
    return TrainingDataSerializer(client, tmp_path / "artifacts"), captured


def test_validate_metadata_identical() -> None:
    assert TrainingDataSerializer.validate_metadata(
        create_basis_metadata(), create_basis_metadata()
    ) is True


def test_validate_metadata_rejects_parameter_and_vocabulary_changes() -> None:
    assert TrainingDataSerializer.validate_metadata(
        create_basis_metadata(sample_size=1.0),
        create_basis_metadata(sample_size=0.5),
    ) is False
    assert TrainingDataSerializer.validate_metadata(
        create_basis_metadata(smile_vocabulary={"A": 1}),
        create_basis_metadata(smile_vocabulary={"A": 1, "B": 2}),
    ) is False
    assert TrainingDataSerializer.validate_metadata(
        create_basis_metadata(smile_vocabulary={"A": 1, "B": 2}),
        create_basis_metadata(smile_vocabulary={"A": 2, "B": 1}),
    ) is False


def test_compute_metadata_hash_is_deterministic() -> None:
    first = create_basis_metadata(smile_vocabulary={"A": 1, "B": 2})
    second = create_basis_metadata(smile_vocabulary={"B": 2, "A": 1})
    assert TrainingDataSerializer.compute_metadata_hash(first) == (
        TrainingDataSerializer.compute_metadata_hash(second)
    )


def test_save_training_dataset_deduplicates_rows_and_publishes_snapshot(
    tmp_path: Path,
) -> None:
    serializer, captured = _serializer(tmp_path)
    dataset = pd.DataFrame(
        [
            {
                "dataset_name": "NIST ISODB",
                "split": "train",
                "temperature": 298.15,
                "pressure": "[1.0,2.0]",
                "adsorbed_amount": "[0.1,0.2]",
                "encoded_adsorbent": 1,
                "adsorbate_molecular_weight": 44.0,
                "adsorbate_encoded_SMILE": "[1,2,3]",
            },
            {
                "dataset_name": "NIST ISODB",
                "split": "train",
                "temperature": 298.15,
                "pressure": "[1.0,2.0]",
                "adsorbed_amount": "[0.1,0.2]",
                "encoded_adsorbent": 1,
                "adsorbate_molecular_weight": 44.0,
                "adsorbate_encoded_SMILE": "[1,2,3]",
            },
        ]
    )

    serializer.save_training_dataset(dataset, "small_dataset", "a" * 64)

    rows = captured[0]["rows"]
    assert isinstance(rows, list)
    assert len(rows) == 1
    assert rows[0]["pressure"] == [1.0, 2.0]
    assert rows[0]["adsorbed_amount"] == [0.1, 0.2]
    assert rows[0]["adsorbate_encoded_SMILE"] == [1, 2, 3]


def test_save_training_metadata_normalizes_json_mappings(tmp_path: Path) -> None:
    serializer, captured = _serializer(tmp_path)
    serializer.save_training_dataset(
        pd.DataFrame([{"split": "train"}]), "small_dataset", "a" * 64
    )
    metadata = pd.DataFrame(
        [
            {
                "dataset_label": "small_dataset",
                "dataset_hash": "a" * 64,
                "smile_vocabulary": '{"C": 1}',
                "adsorbent_vocabulary": '{"MOF-1": 0}',
                "normalization_stats": '{"pressure_mean": 1.0}',
            }
        ]
    )

    serializer.save_training_metadata(metadata, "small_dataset")

    manifest = json.loads(
        (tmp_path / "artifacts" / "training-manifest.json").read_text(encoding="utf-8")
    )
    saved = manifest["small_dataset"]["metadata"]
    assert saved["smile_vocabulary"] == {"C": 1}
    assert saved["adsorbent_vocabulary"] == {"MOF-1": 0}
    assert saved["normalization_stats"] == {"pressure_mean": 1.0}
    assert len(captured) == 1


def test_training_dataset_requires_canonical_hash(tmp_path: Path) -> None:
    serializer, _ = _serializer(tmp_path)

    with pytest.raises(ValueError, match="dataset_hash"):
        serializer.save_training_dataset(
            pd.DataFrame({"split": ["train"]}), "small_dataset", ""
        )


def test_training_metadata_rejects_legacy_fields() -> None:
    with pytest.raises(ValidationError):
        TrainingMetadata(hashcode="a" * 64)


def test_training_data_read_requires_canonical_columns(tmp_path: Path) -> None:
    serializer, _ = _serializer(tmp_path, response_rows=[{"split": "train"}])
    serializer._write_manifest(
        {
            "default": {
                "snapshot_id": "snapshot-1",
                "snapshot_hash": hashlib.sha256(
                    json.dumps(
                        [{"split": "train"}],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "dataset_hash": "a" * 64,
            }
        }
    )

    with pytest.raises(ValueError, match="dataset_label"):
        serializer.load_training_data("default")


def test_checkpoint_metadata_rejects_legacy_hash_alias(tmp_path: Path) -> None:
    configuration_dir = tmp_path / "configuration"
    configuration_dir.mkdir()
    (configuration_dir / "configuration.json").write_text("{}", encoding="utf-8")
    (configuration_dir / "metadata.json").write_text(
        '{"hash_code": "' + ("a" * 64) + '"}', encoding="utf-8"
    )
    (configuration_dir / "session_history.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValidationError):
        ModelSerializer(tmp_path).load_training_configuration(str(tmp_path))
