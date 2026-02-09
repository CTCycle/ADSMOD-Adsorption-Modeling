from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from ADSMOD.server.entities.training import TrainingMetadata
from ADSMOD.server.repositories.serialization.training import TrainingDataSerializer


###############################################################################
@dataclass
class SyntheticDatasetSpec:
    sample_count: int
    series_length: int
    smile_length: int
    adsorbent_count: int
    smile_vocab_size: int
    validation_fraction: float
    seed: int
    dataset_label: str


# -------------------------------------------------------------------------
def build_smile_vocabulary(size: int) -> dict[str, int]:
    tokens = [f"T{index}" for index in range(size)]
    return {token: index + 1 for index, token in enumerate(tokens)}


# -------------------------------------------------------------------------
def build_adsorbent_vocabulary(count: int) -> dict[str, int]:
    return {f"adsorbent_{index}": index for index in range(count)}


# -------------------------------------------------------------------------
def create_synthetic_training_frame(
    spec: SyntheticDatasetSpec,
) -> tuple[pd.DataFrame, TrainingMetadata]:
    rng = np.random.default_rng(spec.seed)
    smile_vocab = build_smile_vocabulary(spec.smile_vocab_size)
    adsorbent_vocab = build_adsorbent_vocabulary(spec.adsorbent_count)

    validation_count = int(spec.sample_count * spec.validation_fraction)
    if spec.sample_count <= 1:
        validation_count = 0
    elif validation_count <= 0:
        validation_count = 1
    elif validation_count >= spec.sample_count:
        validation_count = spec.sample_count - 1

    indices = rng.permutation(spec.sample_count)
    validation_indices = set(indices[:validation_count])

    rows: list[dict[str, object]] = []
    max_pressure_value = 0.0
    max_uptake_value = 0.0

    for index in range(spec.sample_count):
        pressure_values = rng.uniform(0.05, 1.0, size=spec.series_length).astype(
            np.float32
        )
        pressure_values.sort()
        uptake_values = pressure_values * rng.uniform(0.4, 1.2)

        smile_values = rng.integers(
            1,
            spec.smile_vocab_size + 1,
            size=spec.smile_length,
            dtype=np.int32,
        )

        max_pressure_value = max(max_pressure_value, float(pressure_values.max()))
        max_uptake_value = max(max_uptake_value, float(uptake_values.max()))

        row = {
            "dataset_label": spec.dataset_label,
            "dataset_name": spec.dataset_label,
            "split": "validation" if index in validation_indices else "train",
            "temperature": float(rng.uniform(250.0, 500.0)),
            "pressure": json.dumps(pressure_values.tolist()),
            "adsorbed_amount": json.dumps(uptake_values.tolist()),
            "adsorbate_encoded_SMILE": json.dumps(smile_values.tolist()),
            "adsorbate_molecular_weight": float(rng.uniform(10.0, 200.0)),
            "encoded_adsorbent": int(rng.integers(0, spec.adsorbent_count)),
        }
        rows.append(row)

    dataset = pd.DataFrame(rows)
    train_samples = spec.sample_count - validation_count

    metadata = TrainingMetadata(
        created_at=datetime.now().isoformat(),
        sample_size=1.0,
        validation_size=spec.validation_fraction,
        min_measurements=1,
        max_measurements=spec.series_length,
        smile_sequence_size=spec.smile_length,
        max_pressure=max_pressure_value,
        max_uptake=max_uptake_value,
        total_samples=spec.sample_count,
        train_samples=train_samples,
        validation_samples=validation_count,
        smile_vocabulary=smile_vocab,
        adsorbent_vocabulary=adsorbent_vocab,
        normalization_stats={},
        normalization={},
    )
    metadata.dataset_hash = TrainingDataSerializer.compute_metadata_hash(metadata)

    return dataset, metadata


# -------------------------------------------------------------------------
def save_synthetic_training_dataset(
    dataset: pd.DataFrame,
    metadata: TrainingMetadata,
    dataset_label: str,
) -> None:
    serializer = TrainingDataSerializer()
    serializer.save_training_dataset(dataset, dataset_label)

    metadata_df = pd.DataFrame(
        [
            {
                "dataset_label": dataset_label,
                "created_at": metadata.created_at,
                "dataset_hash": metadata.dataset_hash,
                "sample_size": metadata.sample_size,
                "validation_size": metadata.validation_size,
                "min_measurements": metadata.min_measurements,
                "max_measurements": metadata.max_measurements,
                "smile_sequence_size": metadata.smile_sequence_size,
                "max_pressure": metadata.max_pressure,
                "max_uptake": metadata.max_uptake,
                "total_samples": metadata.total_samples,
                "train_samples": metadata.train_samples,
                "validation_samples": metadata.validation_samples,
                "smile_vocabulary": json.dumps(metadata.smile_vocabulary),
                "adsorbent_vocabulary": json.dumps(metadata.adsorbent_vocabulary),
                "normalization_stats": json.dumps(metadata.normalization_stats),
            }
        ]
    )

    serializer.save_training_metadata(metadata_df, dataset_label)


# -------------------------------------------------------------------------
def create_and_save_synthetic_training_dataset(
    spec: SyntheticDatasetSpec,
) -> TrainingMetadata:
    dataset, metadata = create_synthetic_training_frame(spec)
    save_synthetic_training_dataset(dataset, metadata, spec.dataset_label)
    return metadata


# -------------------------------------------------------------------------
def clear_synthetic_training_dataset(dataset_label: str) -> None:
    serializer = TrainingDataSerializer()
    serializer.clear_training_dataset(dataset_label)
