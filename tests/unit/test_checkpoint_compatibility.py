from __future__ import annotations

import json
import logging

import pandas as pd

from ADSMOD.server.routes.training import determine_checkpoint_compatibility
from ADSMOD.server.entities.training import TrainingMetadata
from ADSMOD.server.repositories.serialization import serializer as serializer_module
from ADSMOD.server.repositories.serialization.training import TrainingDataSerializer


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
    metadata = build_metadata("hash-one")
    assert (
        determine_checkpoint_compatibility("checkpoint-a", metadata, {"hash-one"})
        is True
    )


def test_checkpoint_compatibility_multiple_datasets_one_matches() -> None:
    metadata = build_metadata("hash-two")
    dataset_hashes = {"hash-one", "hash-two", "hash-three"}
    assert (
        determine_checkpoint_compatibility("checkpoint-b", metadata, dataset_hashes)
        is True
    )


def test_checkpoint_compatibility_multiple_datasets_none_match() -> None:
    metadata = build_metadata("hash-four")
    dataset_hashes = {"hash-one", "hash-two"}
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


def test_collect_dataset_hashes_skips_uncomputable_hash(monkeypatch, caplog) -> None:
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

    def load_from_database(table_name: str, limit=None, offset=None):
        assert table_name == "training_metadata"
        return metadata_df

    monkeypatch.setattr(
        serializer_module.database, "load_from_database", load_from_database
    )

    serializer = TrainingDataSerializer()
    with caplog.at_level(logging.WARNING, logger="ADSMOD"):
        hashes = serializer.collect_dataset_hashes()
    assert hashes == set()
    assert any("unable to compute hash" in record.message for record in caplog.records)


def test_collect_dataset_hashes_computes_missing_hash(monkeypatch) -> None:
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

    def load_from_database(table_name: str, limit=None, offset=None):
        assert table_name == "training_metadata"
        return metadata_df

    monkeypatch.setattr(
        serializer_module.database, "load_from_database", load_from_database
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

    serializer = TrainingDataSerializer()
    hashes = serializer.collect_dataset_hashes()
    assert hashes == {expected_hash}
