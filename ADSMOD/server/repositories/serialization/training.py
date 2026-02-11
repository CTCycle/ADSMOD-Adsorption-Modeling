from __future__ import annotations

from typing import Any

import hashlib
import json

import pandas as pd

from ADSMOD.server.entities.training import TrainingMetadata
from ADSMOD.server.common.utils.logger import logger
from ADSMOD.server.repositories.queries.training import TrainingRepositoryQueries


###############################################################################
class TrainingDataSerializer:
    dataset_table = "training_dataset"
    metadata_table = "training_metadata"
    dataset_label_column = "name"
    dataset_source_column = "source_dataset"
    metadata_hash_column = "hashcode"
    sample_key_column = "sample_key"
    training_hash_column = "training_hashcode"
    series_columns = ["pressure", "adsorbed_amount", "adsorbate_encoded_SMILE"]

    def __init__(self, queries: TrainingRepositoryQueries | None = None) -> None:
        self.queries = queries or TrainingRepositoryQueries()

    # -------------------------------------------------------------------------
    @classmethod
    def prepare_dataset_for_storage(cls, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset.copy().rename(
            columns={
                "dataset_label": cls.dataset_label_column,
                "dataset_name": cls.dataset_source_column,
                "adsorbate_encoded_SMILE": "adsorbate_encoded_smile",
            }
        )

    # -------------------------------------------------------------------------
    @classmethod
    def restore_dataset_from_storage(cls, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset.copy().rename(
            columns={
                cls.dataset_label_column: "dataset_label",
                cls.dataset_source_column: "dataset_name",
                "adsorbate_encoded_smile": "adsorbate_encoded_SMILE",
            }
        )

    # -------------------------------------------------------------------------
    @classmethod
    def prepare_metadata_for_storage(cls, metadata: pd.DataFrame) -> pd.DataFrame:
        return metadata.copy().rename(
            columns={"dataset_hash": cls.metadata_hash_column}
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_dataset_label(dataset_label: str | None) -> str:
        if not dataset_label:
            return "default"
        normalized = str(dataset_label).strip()
        return normalized or "default"

    # -------------------------------------------------------------------------
    @classmethod
    def build_sample_key(cls, row: pd.Series) -> str:
        payload = {
            cls.dataset_label_column: row.get(cls.dataset_label_column),
            cls.dataset_source_column: row.get(cls.dataset_source_column),
            "split": row.get("split"),
            "temperature": row.get("temperature"),
            "pressure": row.get("pressure"),
            "adsorbed_amount": row.get("adsorbed_amount"),
            "encoded_adsorbent": row.get("encoded_adsorbent"),
            "adsorbate_molecular_weight": row.get("adsorbate_molecular_weight"),
            "adsorbate_encoded_smile": row.get("adsorbate_encoded_smile"),
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------------
    def save_training_dataset(
        self, dataset: pd.DataFrame, dataset_label: str = "default"
    ) -> None:
        dataset_label = self.normalize_dataset_label(dataset_label)
        storage_dataset = self.prepare_dataset_for_storage(dataset)
        if self.dataset_label_column not in storage_dataset.columns:
            storage_dataset[self.dataset_label_column] = dataset_label
        if self.training_hash_column not in storage_dataset.columns:
            storage_dataset[self.training_hash_column] = pd.NA
        storage_dataset[self.sample_key_column] = storage_dataset.apply(
            self.build_sample_key, axis=1
        )
        duplicate_mask = storage_dataset[self.sample_key_column].duplicated(keep="last")
        duplicate_count = int(duplicate_mask.sum())
        if duplicate_count > 0:
            logger.warning(
                "Dropping %d duplicate training rows by sample_key before upsert.",
                duplicate_count,
            )
            storage_dataset = storage_dataset.loc[~duplicate_mask].copy()
        existing = self.queries.load_training_dataset()
        if not existing.empty and self.dataset_label_column in existing.columns:
            retained = existing[existing[self.dataset_label_column] != dataset_label]
            self.queries.save_training_dataset(retained)
        self.queries.upsert_training_dataset(storage_dataset)

    # -------------------------------------------------------------------------
    def save_training_metadata(
        self, metadata: pd.DataFrame, dataset_label: str = "default"
    ) -> None:
        dataset_label = self.normalize_dataset_label(dataset_label)
        storage_metadata = self.prepare_metadata_for_storage(metadata)
        if "dataset_label" not in storage_metadata.columns:
            storage_metadata["dataset_label"] = dataset_label

        existing = self.queries.load_training_metadata()
        if not existing.empty:
            metadata_hash = None
            if self.metadata_hash_column in storage_metadata.columns:
                hash_values = (
                    storage_metadata[self.metadata_hash_column]
                    .dropna()
                    .astype("string")
                    .str.strip()
                )
                metadata_hash = hash_values.iloc[0] if not hash_values.empty else None
            if metadata_hash and self.metadata_hash_column in existing.columns:
                existing = existing[
                    existing[self.metadata_hash_column] != metadata_hash
                ]
            elif "dataset_label" in existing.columns:
                existing = existing[existing["dataset_label"] != dataset_label]

        combined = pd.concat([existing, storage_metadata], ignore_index=True)
        self.queries.save_training_metadata(combined)
        metadata_hash = None
        if self.metadata_hash_column in storage_metadata.columns:
            hash_values = (
                storage_metadata[self.metadata_hash_column]
                .dropna()
                .astype("string")
                .str.strip()
            )
            metadata_hash = hash_values.iloc[0] if not hash_values.empty else None
        if metadata_hash:
            training_data = self.queries.load_training_dataset()
            if (
                not training_data.empty
                and self.dataset_label_column in training_data.columns
            ):
                updated = training_data.copy()
                updated.loc[
                    updated[self.dataset_label_column] == dataset_label,
                    self.training_hash_column,
                ] = metadata_hash
                self.queries.save_training_dataset(updated)

    # -------------------------------------------------------------------------
    def clear_training_dataset(self, dataset_label: str | None = None) -> None:
        if dataset_label is None:
            empty_df = pd.DataFrame()
            self.queries.save_training_dataset(empty_df)
            self.queries.save_training_metadata(empty_df)
            return

        dataset_label = self.normalize_dataset_label(dataset_label)
        existing_data = self.queries.load_training_dataset()
        existing_meta = self.queries.load_training_metadata()

        if (
            not existing_data.empty
            and self.dataset_label_column in existing_data.columns
        ):
            filtered_data = existing_data[
                existing_data[self.dataset_label_column] != dataset_label
            ]
            self.queries.save_training_dataset(filtered_data)

        if not existing_meta.empty and "dataset_label" in existing_meta.columns:
            filtered_meta = existing_meta[
                existing_meta["dataset_label"] != dataset_label
            ]
            self.queries.save_training_metadata(filtered_meta)

    # -------------------------------------------------------------------------
    @staticmethod
    def parse_sequence_value(value: Any) -> list[Any]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, str):
            trimmed = value.strip()
            if not trimmed:
                return []
            try:
                parsed = json.loads(trimmed)
            except json.JSONDecodeError:
                return [x.strip() for x in trimmed.split(",") if x.strip()]
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return list(parsed.values())
            return [parsed]
        if isinstance(value, pd.Series):
            return value.tolist()
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            return list(tolist())
        return [value]

    # -------------------------------------------------------------------------
    def coerce_sequence_columns(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if dataset.empty:
            return dataset
        normalized = dataset.copy()
        for column in self.series_columns:
            if column in normalized.columns:
                normalized[column] = normalized[column].apply(
                    TrainingDataSerializer.parse_sequence_value
                )
        return normalized

    # -------------------------------------------------------------------------
    @staticmethod
    def _parse_json(value: Any) -> dict[str, Any]:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {}
        if isinstance(value, dict):
            return value
        return {}

    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_dataset_hash(dataset_hash_value: Any) -> str | None:
        if pd.notna(dataset_hash_value) and str(dataset_hash_value).strip():
            return str(dataset_hash_value).strip()
        return None

    # -------------------------------------------------------------------------
    def _select_metadata_row(
        self, metadata_df: pd.DataFrame, dataset_label: str
    ) -> pd.Series | None:
        if "dataset_label" in metadata_df.columns:
            filtered = metadata_df[metadata_df["dataset_label"] == dataset_label]
            if filtered.empty:
                return None
            return filtered.iloc[0]
        return metadata_df.iloc[0]

    # -------------------------------------------------------------------------
    def _build_training_metadata(self, row: pd.Series) -> TrainingMetadata:
        smile_vocabulary = self._parse_json(row.get("smile_vocabulary"))
        adsorbent_vocabulary = self._parse_json(row.get("adsorbent_vocabulary"))
        max_smile_index = max(smile_vocabulary.values()) if smile_vocabulary else 0
        smile_vocab_size = int(max_smile_index) + 1
        normalization_stats = self._parse_json(row.get("normalization_stats"))
        dataset_hash_value = row.get(self.metadata_hash_column)
        if dataset_hash_value is None:
            dataset_hash_value = row.get("dataset_hash")

        return TrainingMetadata(
            created_at=str(row.get("created_at", "")),
            dataset_hash=self._normalize_dataset_hash(dataset_hash_value),
            sample_size=float(row.get("sample_size", 1.0)),
            validation_size=float(row.get("validation_size", 0.2)),
            min_measurements=int(row.get("min_measurements", 1)),
            max_measurements=int(row.get("max_measurements", 30)),
            smile_sequence_size=int(row.get("smile_sequence_size", 20)),
            max_pressure=float(row.get("max_pressure", 10000.0)),
            max_uptake=float(row.get("max_uptake", 20.0)),
            total_samples=int(row.get("total_samples", 0)),
            train_samples=int(row.get("train_samples", 0)),
            validation_samples=int(row.get("validation_samples", 0)),
            smile_vocabulary=smile_vocabulary,
            adsorbent_vocabulary=adsorbent_vocabulary,
            normalization_stats=normalization_stats,
            normalization=normalization_stats,
            smile_vocabulary_size=smile_vocab_size,
            adsorbent_vocabulary_size=len(adsorbent_vocabulary),
            SMILE_sequence_size=int(row.get("smile_sequence_size", 20)),
            SMILE_vocabulary=smile_vocabulary,
            SMILE_vocabulary_size=smile_vocab_size,
        )

    # -------------------------------------------------------------------------
    def load_training_metadata(
        self, dataset_label: str = "default"
    ) -> TrainingMetadata:
        dataset_label = self.normalize_dataset_label(dataset_label)
        metadata_df = self.queries.load_training_metadata()
        if metadata_df.empty:
            return TrainingMetadata()

        row = self._select_metadata_row(metadata_df, dataset_label)
        if row is None:
            return TrainingMetadata()
        return self._build_training_metadata(row)

    # -------------------------------------------------------------------------
    def collect_dataset_hashes(self) -> set[str]:
        metadata_df = self.queries.load_training_metadata()
        if metadata_df.empty:
            return set()

        dataset_labels: set[str] = set()
        if "dataset_label" in metadata_df.columns:
            for label in metadata_df["dataset_label"].tolist():
                dataset_labels.add(self.normalize_dataset_label(label))
        else:
            dataset_labels.add("default")

        dataset_hashes: set[str] = set()
        for dataset_label in sorted(dataset_labels):
            metadata = self.load_training_metadata(dataset_label)
            dataset_hash = metadata.dataset_hash
            if not dataset_hash:
                if not metadata or metadata.total_samples == 0:
                    logger.warning(
                        "Training metadata missing or empty for dataset '%s'; unable to compute hash.",
                        dataset_label,
                    )
                    continue
                computed_hash = TrainingDataSerializer.compute_metadata_hash(metadata)
                if not computed_hash:
                    logger.warning(
                        "Training metadata hash could not be computed for dataset '%s'.",
                        dataset_label,
                    )
                    continue
                dataset_hash = computed_hash
            dataset_hashes.add(dataset_hash)

        return dataset_hashes

    # -------------------------------------------------------------------------
    def load_training_data(
        self, dataset_label: str = "default", only_metadata: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame, TrainingMetadata] | TrainingMetadata:
        dataset_label = self.normalize_dataset_label(dataset_label)
        metadata = self.load_training_metadata(dataset_label)
        if only_metadata:
            return metadata

        training_data = self.queries.load_training_dataset()
        if training_data.empty:
            return training_data, training_data, metadata

        if self.dataset_label_column in training_data.columns:
            training_data = training_data[
                training_data[self.dataset_label_column] == dataset_label
            ]

        training_data = self.restore_dataset_from_storage(training_data)
        training_data = self.coerce_sequence_columns(training_data)

        train_data = training_data[training_data["split"] == "train"]
        val_data = training_data[training_data["split"] == "validation"]

        return train_data, val_data, metadata

    # -------------------------------------------------------------------------
    @staticmethod
    def list_processed_datasets() -> list[dict[str, Any]]:
        metadata_df = TrainingRepositoryQueries().load_training_metadata()
        if metadata_df.empty:
            return []

        datasets = []
        for _, row in metadata_df.iterrows():
            dataset_hash_value = row.get(TrainingDataSerializer.metadata_hash_column)
            if dataset_hash_value is None:
                dataset_hash_value = row.get("dataset_hash")
            datasets.append(
                {
                    "dataset_label": str(row.get("dataset_label", "default")),
                    "dataset_hash": str(dataset_hash_value).strip()
                    if pd.notna(dataset_hash_value) and str(dataset_hash_value).strip()
                    else None,
                    "train_samples": int(row.get("train_samples", 0)),
                    "validation_samples": int(row.get("validation_samples", 0)),
                    "created_at": str(row.get("created_at", "")),
                }
            )

        return datasets

    # -------------------------------------------------------------------------
    @staticmethod
    def compute_metadata_hash(metadata: TrainingMetadata) -> str:
        if not metadata:
            return ""

        payload = {
            "sample_size": metadata.sample_size,
            "validation_size": metadata.validation_size,
            "min_measurements": metadata.min_measurements,
            "max_measurements": metadata.max_measurements,
            "smile_sequence_size": metadata.smile_sequence_size,
            "max_pressure": metadata.max_pressure,
            "max_uptake": metadata.max_uptake,
            "smile_vocabulary": sorted(metadata.smile_vocabulary.items())
            if metadata.smile_vocabulary
            else [],
            "adsorbent_vocabulary": sorted(metadata.adsorbent_vocabulary.items())
            if metadata.adsorbent_vocabulary
            else [],
            "normalization_stats": metadata.normalization_stats,
        }

        serialized = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------------
    @staticmethod
    def validate_metadata(
        metadata: TrainingMetadata, target_metadata: TrainingMetadata
    ) -> bool:
        if not metadata or not target_metadata:
            logger.warning("Metadata validation failed: missing metadata")
            return False

        hash_a = TrainingDataSerializer.compute_metadata_hash(metadata)
        hash_b = TrainingDataSerializer.compute_metadata_hash(target_metadata)
        if hash_a != hash_b:
            logger.debug(
                "Metadata mismatch: Content hash mismatch (%s != %s)",
                hash_a,
                hash_b,
            )
            return False

        return True


###############################################################################
