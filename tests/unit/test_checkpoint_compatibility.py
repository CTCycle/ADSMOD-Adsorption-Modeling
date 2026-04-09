from __future__ import annotations

import json
import logging

import pandas as pd

from ADSMOD.server.services.training import determine_checkpoint_compatibility
from ADSMOD.server.domain.training import TrainingMetadata
from ADSMOD.server.repositories.serialization.training import TrainingDataSerializer


class TrainingMetadataQueries:
    def __init__(self, metadata_frame: pd.DataFrame) -> None:
        self.metadata_frame = metadata_frame

    def load_training_metadata(self, limit=None, offset=None):  # noqa: ANN001
        return self.metadata_frame


def build_metadata(dataset_hash: str | None) -> TrainingMetadata:
    return TrainingMetadata(
        dataset_hash=dataset_hash,
        total_samples=10,
        smile_vocabulary={"C": 1},
        adsorbent_vocabulary={"MOF": 1},
        normalization_stats={"pressure_mean": 5.0},
        normalization={"pressure_mean": 5.0},
    )


def test_checkpoint_compatibility_single_dataset_match() -> None:
    hash_one = "1" * 64
    metadata = build_metadata(hash_one)
    assert (
        determine_checkpoint_compatibility("checkpoint-a", metadata, {hash_one})
        is True
    )


def test_checkpoint_compatibility_multiple_datasets_one_matches() -> None:
    hash_one = "1" * 64
    hash_two = "2" * 64
    hash_three = "3" * 64
    metadata = build_metadata(hash_two)
    dataset_hashes = {hash_one, hash_two, hash_three}
    assert (
        determine_checkpoint_compatibility("checkpoint-b", metadata, dataset_hashes)
        is True
    )


def test_checkpoint_compatibility_multiple_datasets_none_match() -> None:
    hash_one = "1" * 64
    hash_two = "2" * 64
    hash_four = "4" * 64
    metadata = build_metadata(hash_four)
    dataset_hashes = {hash_one, hash_two}
    assert (
        determine_checkpoint_compatibility("checkpoint-c", metadata, dataset_hashes)
        is False
    )


def test_checkpoint_compatibility_missing_metadata_logs_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="ADSMOD"):
        result = determine_checkpoint_compatibility(
            "checkpoint-missing", None, {"hash-one"}
        )
    assert result is False
    assert any(
        "metadata missing or invalid" in record.message for record in caplog.records
    )


def test_collect_dataset_hashes_skips_uncomputable_hash(caplog) -> None:
    metadata_df = pd.DataFrame(
        [
            {
                "dataset_label": "default",
                "dataset_hash": None,
                "total_samples": 0,
                "train_samples": 0,
                "validation_samples": 0,
                "sample_size": 1.0,
                "validation_size": 0.2,
                "min_measurements": 1,
                "max_measurements": 30,
                "smile_sequence_size": 20,
                "max_pressure": 10000.0,
                "max_uptake": 20.0,
                "smile_vocabulary": json.dumps({"C": 1}),
                "adsorbent_vocabulary": json.dumps({"MOF": 1}),
                "normalization_stats": json.dumps({"pressure_mean": 5.0}),
                "created_at": "2026-01-27T00:00:00",
            }
        ]
    )

    serializer = TrainingDataSerializer(queries=TrainingMetadataQueries(metadata_df))
    with caplog.at_level(logging.WARNING, logger="ADSMOD"):
        hashes = serializer.collect_dataset_hashes()
    assert hashes == set()
    assert any("unable to compute hash" in record.message for record in caplog.records)


def test_collect_dataset_hashes_computes_missing_hash() -> None:
    metadata_df = pd.DataFrame(
        [
            {
                "dataset_label": "alpha",
                "dataset_hash": None,
                "total_samples": 100,
                "train_samples": 80,
                "validation_samples": 20,
                "sample_size": 1.0,
                "validation_size": 0.2,
                "min_measurements": 1,
                "max_measurements": 30,
                "smile_sequence_size": 20,
                "max_pressure": 10000.0,
                "max_uptake": 20.0,
                "smile_vocabulary": json.dumps({"C": 1, "O": 2}),
                "adsorbent_vocabulary": json.dumps({"MOF": 1}),
                "normalization_stats": json.dumps({"pressure_mean": 5.0}),
                "created_at": "2026-01-27T00:00:00",
            }
        ]
    )

    expected_metadata = TrainingMetadata(
        created_at="2026-01-27T00:00:00",
        dataset_hash=None,
        sample_size=1.0,
        validation_size=0.2,
        min_measurements=1,
        max_measurements=30,
        smile_sequence_size=20,
        max_pressure=10000.0,
        max_uptake=20.0,
        total_samples=100,
        train_samples=80,
        validation_samples=20,
        smile_vocabulary={"C": 1, "O": 2},
        adsorbent_vocabulary={"MOF": 1},
        normalization_stats={"pressure_mean": 5.0},
        normalization={"pressure_mean": 5.0},
    )
    expected_hash = TrainingDataSerializer.compute_metadata_hash(expected_metadata)

    serializer = TrainingDataSerializer(queries=TrainingMetadataQueries(metadata_df))
    hashes = serializer.collect_dataset_hashes()
    assert hashes == {expected_hash}
