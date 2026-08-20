from __future__ import annotations

from typing import Any

import hashlib
import json

import pandas as pd

from ml_service.contracts.training import TrainingMetadata
from shared.common.utils.logger import logger
from shared.repositories.queries.training import TrainingRepositoryQueries

###############################################################################
class TrainingDataSerializer:
    dataset_label_column = "dataset_label"
    dataset_hash_column = "dataset_hash"
    dataset_name_column = "dataset_name"
    sample_key_column = "sample_key"
    series_columns = ["pressure", "adsorbed_amount", "adsorbate_encoded_SMILE"]

    # -------------------------------------------------------------------------
    def __init__(self, queries: TrainingRepositoryQueries | None = None) -> None:
        self.queries = queries or TrainingRepositoryQueries()

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
            cls.dataset_name_column: row.get(cls.dataset_name_column),
            "split": row.get("split"),
            "temperature": row.get("temperature"),
            "pressure": row.get("pressure"),
            "adsorbed_amount": row.get("adsorbed_amount"),
            "encoded_adsorbent": row.get("encoded_adsorbent"),
            "adsorbate_molecular_weight": row.get("adsorbate_molecular_weight"),
            "adsorbate_encoded_SMILE": row.get("adsorbate_encoded_SMILE"),
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------------
    @staticmethod
    def build_archived_label(dataset_label: str) -> str:
        timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d%H%M%S%f")
        return f"archived::{dataset_label}::{timestamp}"

    # -------------------------------------------------------------------------
    @staticmethod
    def build_archived_hash(archived_label: str) -> str:
        return hashlib.sha256(archived_label.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------------
    def archive_training_dataset_rows(self, dataset_label: str | None = None) -> None:
        existing_data = self.queries.load_training_dataset()
        if existing_data.empty:
            return
        self._require_columns(
            existing_data,
            {self.dataset_label_column, self.dataset_hash_column},
            "Stored training data",
        )

        archived_data = existing_data.copy()
        if dataset_label is None:
            mask = pd.Series(True, index=archived_data.index)
            label_seed = "all"
        else:
            mask = archived_data[self.dataset_label_column] == dataset_label
            label_seed = dataset_label
        if not bool(mask.any()):
            return

        archived_label = self.build_archived_label(self.normalize_dataset_label(label_seed))
        archived_data.loc[mask, self.dataset_label_column] = archived_label
        archived_data.loc[mask, self.dataset_hash_column] = self.build_archived_hash(
            archived_label
        )
        self.queries.upsert_training_dataset(archived_data)

    # -------------------------------------------------------------------------
    def archive_training_metadata_rows(self, dataset_label: str | None = None) -> None:
        existing_meta = self.queries.load_training_metadata()
        if existing_meta.empty:
            return
        self._require_columns(
            existing_meta,
            {self.dataset_label_column, self.dataset_hash_column},
            "Stored training metadata",
        )

        archived_meta = existing_meta.copy()
        if dataset_label is None:
            mask = pd.Series(True, index=archived_meta.index)
            label_seed = "all"
        else:
            mask = archived_meta[self.dataset_label_column] == dataset_label
            label_seed = dataset_label
        if not bool(mask.any()):
            return

        archived_label = self.build_archived_label(self.normalize_dataset_label(label_seed))
        archived_meta.loc[mask, self.dataset_label_column] = archived_label
        archived_meta.loc[mask, self.dataset_hash_column] = self.build_archived_hash(
            archived_label
        )
        self.queries.save_training_metadata(archived_meta)

    # -------------------------------------------------------------------------
    def save_training_dataset(
        self, dataset: pd.DataFrame, dataset_label: str = "default", dataset_hash: str = ""
    ) -> None:
        dataset_label = self.normalize_dataset_label(dataset_label)
        dataset_hash = self.require_dataset_hash(dataset_hash)
        normalized_dataset = self.coerce_sequence_columns(dataset)
        storage_dataset = normalized_dataset.copy()
        if self.dataset_label_column not in storage_dataset.columns:
            storage_dataset[self.dataset_label_column] = dataset_label
        storage_dataset[self.dataset_hash_column] = dataset_hash
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

        self.archive_training_dataset_rows(dataset_label)
        self.queries.upsert_training_dataset(storage_dataset)

    # -------------------------------------------------------------------------
    def save_training_metadata(
        self, metadata: pd.DataFrame, dataset_label: str = "default"
    ) -> None:
        dataset_label = self.normalize_dataset_label(dataset_label)
        storage_metadata = metadata.copy()
        if "dataset_label" not in storage_metadata.columns:
            storage_metadata["dataset_label"] = dataset_label
        if self.dataset_hash_column not in storage_metadata.columns:
            raise ValueError("Training metadata requires dataset_hash.")
        storage_metadata[self.dataset_hash_column] = storage_metadata[
            self.dataset_hash_column
        ].apply(self.require_dataset_hash)
        for column in ("smile_vocabulary", "adsorbent_vocabulary", "normalization_stats"):
            if column in storage_metadata.columns:
                storage_metadata[column] = storage_metadata[column].apply(
                    self._parse_json
                )

        self.archive_training_metadata_rows(dataset_label)
        self.queries.save_training_metadata(storage_metadata)

    # -------------------------------------------------------------------------
    def clear_training_dataset(self, dataset_label: str | None = None) -> None:
        if dataset_label is None:
            self.archive_training_dataset_rows(None)
            self.archive_training_metadata_rows(None)
            return

        dataset_label = self.normalize_dataset_label(dataset_label)
        self.archive_training_dataset_rows(dataset_label)
        self.archive_training_metadata_rows(dataset_label)

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
    @staticmethod
    def _require_columns(
        frame: pd.DataFrame, required_columns: set[str], context: str
    ) -> None:
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{context} is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

    # -------------------------------------------------------------------------
    @staticmethod
    def require_dataset_hash(dataset_hash: Any) -> str:
        if dataset_hash is None or pd.isna(dataset_hash):
            normalized = ""
        else:
            normalized = str(dataset_hash).strip()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in normalized
        ):
            raise ValueError("dataset_hash must be a 64-character hexadecimal digest.")
        return normalized

    # -------------------------------------------------------------------------
    def _select_metadata_row(
        self, metadata_df: pd.DataFrame, dataset_label: str
    ) -> pd.Series | None:
        self._require_columns(
            metadata_df,
            {self.dataset_label_column},
            "Training metadata",
        )
        filtered = metadata_df[metadata_df[self.dataset_label_column] == dataset_label]
        if filtered.empty:
            return None
        return filtered.iloc[0]

    # -------------------------------------------------------------------------
    def _build_training_metadata(self, row: pd.Series) -> TrainingMetadata:
        smile_vocabulary = self._parse_json(row.get("smile_vocabulary"))
        adsorbent_vocabulary = self._parse_json(row.get("adsorbent_vocabulary"))
        max_smile_index = max(smile_vocabulary.values()) if smile_vocabulary else 0
        smile_vocab_size = int(max_smile_index) + 1
        normalization_stats = self._parse_json(row.get("normalization_stats"))
        dataset_hash_value = row.get(self.dataset_hash_column)

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
            smile_vocabulary_size=smile_vocab_size,
            adsorbent_vocabulary_size=len(adsorbent_vocabulary),
        )

    # -------------------------------------------------------------------------
    def load_training_metadata(
        self, dataset_label: str = "default"
    ) -> TrainingMetadata:
        dataset_label = self.normalize_dataset_label(dataset_label)
        metadata_df = self.queries.load_training_metadata()
        if metadata_df.empty:
            return TrainingMetadata()

        self._require_columns(
            metadata_df,
            {self.dataset_label_column, self.dataset_hash_column},
            "Training metadata",
        )
        row = self._select_metadata_row(metadata_df, dataset_label)
        if row is None:
            return TrainingMetadata()
        return self._build_training_metadata(row)

    # -------------------------------------------------------------------------
    def collect_dataset_hashes(self) -> set[str]:
        metadata_df = self.queries.load_training_metadata()
        if metadata_df.empty:
            return set()

        self._require_columns(
            metadata_df,
            {self.dataset_label_column, self.dataset_hash_column},
            "Training metadata",
        )
        dataset_labels: set[str] = set()
        for label in metadata_df[self.dataset_label_column].tolist():
            dataset_labels.add(self.normalize_dataset_label(label))

        dataset_hashes: set[str] = set()
        for dataset_label in sorted(dataset_labels):
            metadata = self.load_training_metadata(dataset_label)
            dataset_hash = metadata.dataset_hash
            if not dataset_hash:
                logger.warning(
                    "Training metadata missing dataset_hash for dataset '%s'.",
                    dataset_label,
                )
                continue
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

        self._require_columns(
            training_data,
            {self.dataset_label_column, self.dataset_hash_column},
            "Training data",
        )
        training_data = training_data[
            training_data[self.dataset_label_column] == dataset_label
        ]

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

        TrainingDataSerializer._require_columns(
            metadata_df,
            {
                TrainingDataSerializer.dataset_label_column,
                TrainingDataSerializer.dataset_hash_column,
            },
            "Training metadata",
        )
        datasets = []
        for _, row in metadata_df.iterrows():
            dataset_hash_value = row.get(TrainingDataSerializer.dataset_hash_column)
            datasets.append(
                {
                    "dataset_label": str(row[TrainingDataSerializer.dataset_label_column]),
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



