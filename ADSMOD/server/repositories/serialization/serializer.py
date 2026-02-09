from __future__ import annotations

from typing import Any

import json
import hashlib
import os
import shutil
from datetime import datetime

import pandas as pd
from ADSMOD.server.repositories.database.backend import database
from ADSMOD.server.repositories.queries.data import DataRepositoryQueries
from ADSMOD.server.repositories.queries.training import TrainingRepositoryQueries
from ADSMOD.server.entities.models import MODEL_SCHEMAS
from ADSMOD.server.common.constants import (
    CHECKPOINTS_PATH,
    COLUMN_ADSORBATE,
    COLUMN_ADSORBENT,
    COLUMN_BEST_MODEL,
    COLUMN_DATASET_NAME,
    COLUMN_EXPERIMENT,
    COLUMN_EXPERIMENT_NAME,
    COLUMN_ID,
    COLUMN_MAX_PRESSURE,
    COLUMN_MAX_UPTAKE,
    COLUMN_MEASUREMENT_COUNT,
    COLUMN_MIN_PRESSURE,
    COLUMN_MIN_UPTAKE,
    COLUMN_PRESSURE_PA,
    COLUMN_TEMPERATURE_K,
    COLUMN_UPTAKE_MOL_G,
    COLUMN_WORST_MODEL,
)
from ADSMOD.server.learning.metrics import (
    MaskedMeanSquaredError,
    MaskedRSquared,
)
from ADSMOD.server.entities.training import TrainingMetadata
from ADSMOD.server.learning.models.embeddings import MolecularEmbedding
from ADSMOD.server.learning.models.encoders import (
    PressureSerierEncoder,
    QDecoder,
    StateEncoder,
)
from ADSMOD.server.learning.models.transformers import (
    AddNorm,
    FeedForward,
    TransformerEncoder,
)
from ADSMOD.server.learning.training.scheduler import LinearDecayLRScheduler
from ADSMOD.server.common.utils.logger import logger


###############################################################################
class DataSerializer:
    raw_table = "adsorption_data"
    processed_table = "adsorption_processed_data"
    best_fit_table = "adsorption_best_fit"
    raw_name_column = "name"
    fitting_name_column = "name"
    processed_key_column = "processed_key"
    material_aliases = {
        COLUMN_ADSORBATE: [COLUMN_ADSORBATE],
        COLUMN_ADSORBENT: [COLUMN_ADSORBENT],
    }
    table_aliases = {
        "ADSORPTION_DATA": raw_table,
        "ADSORPTION_PROCESSED_DATA": processed_table,
        "ADSORPTION_BEST_FIT": best_fit_table,
        "ADSORPTION_LANGMUIR": "adsorption_langmuir",
        "ADSORPTION_SIPS": "adsorption_sips",
        "ADSORPTION_FREUNDLICH": "adsorption_freundlich",
        "ADSORPTION_TEMKIN": "adsorption_temkin",
        "ADSORPTION_TOTH": "adsorption_toth",
        "ADSORPTION_DUBININ_RADUSHKEVICH": "adsorption_dubinin_radushkevich",
        "ADSORPTION_DUAL_SITE_LANGMUIR": "adsorption_dual_site_langmuir",
        "ADSORPTION_REDLICH_PETERSON": "adsorption_redlich_peterson",
        "ADSORPTION_JOVANOVIC": "adsorption_jovanovic",
        "NIST_SINGLE_COMPONENT_ADSORPTION": "nist_single_component_adsorption",
        "NIST_BINARY_MIXTURE_ADSORPTION": "nist_binary_mixture_adsorption",
        "ADSORBATES": "adsorbates",
        "ADSORBENTS": "adsorbents",
        "TRAINING_DATASET": "training_dataset",
        "TRAINING_METADATA": "training_metadata",
    }
    experiment_columns = [
        fitting_name_column,
        COLUMN_ADSORBENT,
        COLUMN_ADSORBATE,
        COLUMN_TEMPERATURE_K,
        COLUMN_PRESSURE_PA,
        COLUMN_UPTAKE_MOL_G,
        COLUMN_MEASUREMENT_COUNT,
        COLUMN_MIN_PRESSURE,
        COLUMN_MAX_PRESSURE,
        COLUMN_MIN_UPTAKE,
        COLUMN_MAX_UPTAKE,
    ]

    def __init__(self, queries: DataRepositoryQueries | None = None) -> None:
        self.queries = queries or DataRepositoryQueries()

    # -------------------------------------------------------------------------
    @classmethod
    def normalize_table_name(cls, table_name: str) -> str:
        return cls.table_aliases.get(table_name, table_name)

    # -------------------------------------------------------------------------
    @classmethod
    def fitting_tables(cls) -> set[str]:
        return {cls.processed_table, cls.best_fit_table}.union(
            {schema["table"] for schema in MODEL_SCHEMAS.values()}
        )

    # -------------------------------------------------------------------------
    @classmethod
    def normalize_material_columns(cls, dataset: pd.DataFrame) -> pd.DataFrame:
        normalized = dataset.copy()
        rename_map: dict[str, str] = {}
        for target, aliases in cls.material_aliases.items():
            if target in normalized.columns:
                continue
            for alias in aliases:
                if alias in normalized.columns:
                    rename_map[alias] = target
                    break
        if rename_map:
            normalized = normalized.rename(columns=rename_map)
        for target in cls.material_aliases:
            if target not in normalized.columns:
                normalized[target] = ""
        return normalized

    # -------------------------------------------------------------------------
    @classmethod
    def build_processed_key(cls, row: pd.Series) -> str:
        payload = {
            COLUMN_ADSORBENT: row.get(COLUMN_ADSORBENT, ""),
            COLUMN_ADSORBATE: row.get(COLUMN_ADSORBATE, ""),
            COLUMN_TEMPERATURE_K: row.get(COLUMN_TEMPERATURE_K),
            COLUMN_PRESSURE_PA: row.get(COLUMN_PRESSURE_PA),
            COLUMN_UPTAKE_MOL_G: row.get(COLUMN_UPTAKE_MOL_G),
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------------
    @classmethod
    def add_processed_keys(cls, dataset: pd.DataFrame) -> pd.DataFrame:
        normalized = dataset.copy()
        if normalized.empty:
            normalized[cls.processed_key_column] = pd.Series(dtype="string")
            return normalized
        normalized[cls.processed_key_column] = normalized.apply(
            cls.build_processed_key, axis=1
        )
        return normalized

    # -------------------------------------------------------------------------
    @classmethod
    def prepare_for_storage(
        cls, dataset: pd.DataFrame, table_name: str
    ) -> pd.DataFrame:
        normalized = cls.normalize_table_name(table_name)
        storage = dataset.copy()
        if normalized == cls.raw_table:
            storage = cls.normalize_material_columns(storage)
            if (
                COLUMN_DATASET_NAME in storage.columns
                and cls.raw_name_column in storage.columns
            ):
                storage = storage.drop(columns=[COLUMN_DATASET_NAME], errors="ignore")
            else:
                storage = storage.rename(
                    columns={COLUMN_DATASET_NAME: cls.raw_name_column}
                )
        elif normalized in cls.fitting_tables():
            if (
                COLUMN_EXPERIMENT_NAME in storage.columns
                and cls.fitting_name_column in storage.columns
            ):
                storage = storage.drop(
                    columns=[COLUMN_EXPERIMENT_NAME], errors="ignore"
                )
            else:
                storage = storage.rename(
                    columns={COLUMN_EXPERIMENT_NAME: cls.fitting_name_column}
                )
            if normalized == cls.processed_table:
                storage = cls.normalize_material_columns(storage)
                storage = storage.drop(columns=[COLUMN_EXPERIMENT], errors="ignore")
                storage = cls.add_processed_keys(storage)
        return storage

    # -------------------------------------------------------------------------
    @classmethod
    def restore_from_storage(
        cls, dataset: pd.DataFrame, table_name: str
    ) -> pd.DataFrame:
        return dataset.copy()

    # -------------------------------------------------------------------------
    def save_raw_dataset(self, dataset: pd.DataFrame) -> None:
        table_name = self.raw_table
        storage_dataset = self.prepare_for_storage(dataset, table_name)
        self.queries.upsert_table(storage_dataset, table_name)

    # -------------------------------------------------------------------------
    def delete_raw_dataset(self, dataset_name: str) -> bool:
        dataset_name = str(dataset_name or "").strip()
        if not dataset_name:
            return False

        existing = self.queries.load_table(self.raw_table)
        if existing.empty or self.raw_name_column not in existing.columns:
            return False

        filtered = existing[existing[self.raw_name_column] != dataset_name].copy()
        if len(filtered) == len(existing):
            return False

        self.queries.save_table(filtered, self.raw_table)
        return True

    # -------------------------------------------------------------------------
    def load_table(
        self,
        table_name: str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> pd.DataFrame:
        normalized = self.normalize_table_name(table_name)
        loaded = self.queries.load_table(normalized, limit=limit, offset=offset)
        return self.restore_from_storage(loaded, normalized)

    # -------------------------------------------------------------------------
    def upsert_table(self, dataset: pd.DataFrame, table_name: str) -> None:
        normalized = self.normalize_table_name(table_name)
        storage_dataset = self.prepare_for_storage(dataset, normalized)
        self.queries.upsert_table(storage_dataset, normalized)

    # -------------------------------------------------------------------------
    def save_processed_dataset(self, dataset: pd.DataFrame) -> None:
        self.upsert_table(dataset, self.processed_table)

    # -------------------------------------------------------------------------
    def save_fitting_results(self, dataset: pd.DataFrame) -> None:
        if dataset.empty:
            return
        experiments = self.build_experiment_frame(dataset)
        self.upsert_table(experiments, self.processed_table)
        for schema in MODEL_SCHEMAS.values():
            model_frame = self.build_model_frame(dataset, schema)
            if model_frame is None:
                continue
            self.upsert_table(model_frame, schema["table"])

    # -------------------------------------------------------------------------
    def load_fitting_results(self) -> pd.DataFrame:
        experiments = self.load_table(self.processed_table)
        if experiments.empty:
            return experiments
        combined = experiments.rename(
            columns={self.fitting_name_column: COLUMN_EXPERIMENT_NAME}
        )
        if COLUMN_EXPERIMENT_NAME not in combined.columns:
            return combined
        for schema in MODEL_SCHEMAS.values():
            model_frame = self.load_table(schema["table"])
            if model_frame.empty:
                continue
            model_frame = model_frame.rename(
                columns={self.fitting_name_column: COLUMN_EXPERIMENT_NAME}
            )
            if COLUMN_EXPERIMENT_NAME not in model_frame.columns:
                continue
            renamed = self.rename_model_columns(model_frame, schema)
            combined = combined.merge(renamed, how="left", on=COLUMN_EXPERIMENT_NAME)
        return combined

    # -------------------------------------------------------------------------
    def save_best_fit(self, dataset: pd.DataFrame) -> None:
        if dataset.empty:
            return
        if COLUMN_EXPERIMENT_NAME not in dataset.columns:
            raise ValueError("Missing experiment name column for best fit results.")
        best = pd.DataFrame()
        best[COLUMN_EXPERIMENT_NAME] = dataset.get(COLUMN_EXPERIMENT_NAME)
        best[COLUMN_BEST_MODEL] = dataset.get(COLUMN_BEST_MODEL)
        best[COLUMN_WORST_MODEL] = dataset.get(COLUMN_WORST_MODEL)
        self.upsert_table(best, self.best_fit_table)

    # -------------------------------------------------------------------------
    def load_best_fit(self) -> pd.DataFrame:
        best = self.load_table(self.best_fit_table)
        if best.empty:
            return best
        best = best.rename(columns={self.fitting_name_column: COLUMN_EXPERIMENT_NAME})
        if COLUMN_EXPERIMENT_NAME not in best.columns:
            return best
        experiments = self.load_table(self.processed_table)
        if experiments.empty:
            return pd.DataFrame()
        experiments = experiments.rename(
            columns={self.fitting_name_column: COLUMN_EXPERIMENT_NAME}
        )
        drop_columns = [COLUMN_ID]
        merged = experiments.merge(
            best.drop(columns=drop_columns, errors="ignore"),
            how="left",
            on=COLUMN_EXPERIMENT_NAME,
        )
        return merged

    # -------------------------------------------------------------------------
    def build_experiment_frame(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if COLUMN_EXPERIMENT_NAME not in dataset.columns:
            raise ValueError("Missing experiment name column for fitting results.")
        experiments = dataset.copy()
        experiments[self.fitting_name_column] = experiments[COLUMN_EXPERIMENT_NAME]
        for column in self.experiment_columns:
            if column not in experiments.columns:
                experiments[column] = pd.NA
        experiments = experiments.loc[:, self.experiment_columns].copy()
        return experiments

    # -------------------------------------------------------------------------
    def resolve_dataset_column(
        self, prefix: str, suffix: str, columns: list[str] | pd.Index
    ) -> str | None:
        target = f"{prefix} {suffix}".lower()
        for column in columns:
            if str(column).lower() == target:
                return column
        return None

    # -------------------------------------------------------------------------
    def build_model_frame(
        self,
        dataset: pd.DataFrame,
        schema: dict[str, Any],
    ) -> pd.DataFrame | None:
        resolved = {
            field: self.resolve_dataset_column(
                schema["prefix"], suffix, dataset.columns
            )
            for field, suffix in schema["fields"].items()
        }
        if all(column is None for column in resolved.values()):
            return None
        if COLUMN_EXPERIMENT_NAME not in dataset.columns:
            raise ValueError("Missing experiment name column for fitting results.")
        model_frame = pd.DataFrame()
        model_frame[self.fitting_name_column] = dataset.get(COLUMN_EXPERIMENT_NAME)
        for field, column in resolved.items():
            target = schema["fields"][field]
            if column is None:
                model_frame[target] = pd.NA
            else:
                model_frame[target] = dataset[column]
        return model_frame

    # -------------------------------------------------------------------------
    def rename_model_columns(
        self, model_frame: pd.DataFrame, schema: dict[str, Any]
    ) -> pd.DataFrame:
        rename_map = {
            column_name: f"{schema['prefix']} {column_name}"
            for column_name in schema["fields"].values()
        }
        trimmed = model_frame.rename(columns=rename_map)
        drop_columns = [COLUMN_ID, "experiment_id"]
        return trimmed.drop(columns=drop_columns, errors="ignore")


###############################################################################
class TrainingDataSerializer:
    dataset_table = "training_dataset"
    metadata_table = "training_metadata"
    dataset_label_column = "name"
    dataset_source_column = "source_dataset"
    metadata_hash_column = "hashcode"
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
    def save_training_dataset(
        self, dataset: pd.DataFrame, dataset_label: str = "default"
    ) -> None:
        dataset_label = self.normalize_dataset_label(dataset_label)
        storage_dataset = self.prepare_dataset_for_storage(dataset)
        if self.dataset_label_column not in storage_dataset.columns:
            storage_dataset[self.dataset_label_column] = dataset_label
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
class ModelSerializer:
    def __init__(self, model_name: str = "SCADS") -> None:
        self.model_name = model_name

    # -------------------------------------------------------------------------
    def create_checkpoint_folder(self) -> str:
        today_datetime = datetime.now().strftime("%Y%m%dT%H%M%S")
        checkpoint_path = os.path.join(
            CHECKPOINTS_PATH, f"{self.model_name}_{today_datetime}"
        )
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(os.path.join(checkpoint_path, "configuration"), exist_ok=True)
        logger.debug("Created checkpoint folder at %s", checkpoint_path)

        return checkpoint_path

    # -------------------------------------------------------------------------
    def delete_checkpoint(self, checkpoint_name: str) -> bool:
        checkpoint_path = os.path.join(CHECKPOINTS_PATH, checkpoint_name)
        if not os.path.exists(checkpoint_path):
            return False
        try:
            shutil.rmtree(checkpoint_path)
            logger.info("Deleted checkpoint: %s", checkpoint_name)
            return True
        except Exception as e:
            logger.error("Failed to delete checkpoint %s: %s", checkpoint_name, e)
            return False

    # -------------------------------------------------------------------------
    def save_pretrained_model(self, model: Any, path: str) -> None:
        model_files_path = os.path.join(path, "saved_model.keras")
        model.save(model_files_path)
        logger.info(
            "Training session is over. Model %s has been saved", os.path.basename(path)
        )

    # -------------------------------------------------------------------------
    def save_training_configuration(
        self,
        path: str,
        history: dict,
        configuration: dict[str, Any],
        metadata: TrainingMetadata,
    ) -> None:
        config_path = os.path.join(path, "configuration", "configuration.json")
        metadata_path = os.path.join(path, "configuration", "metadata.json")
        history_path = os.path.join(path, "configuration", "session_history.json")

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(configuration, f, indent=4, default=str)
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(metadata.model_dump_json(indent=4))
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, default=str)

        logger.debug(
            "Model configuration, session history and metadata saved for %s",
            os.path.basename(path),
        )

    # -------------------------------------------------------------------------
    def scan_checkpoints_folder(self) -> list[str]:
        model_folders: list[str] = []
        if not os.path.exists(CHECKPOINTS_PATH):
            return model_folders
        for entry in os.scandir(CHECKPOINTS_PATH):
            if entry.is_dir():
                has_keras = any(
                    f.name.endswith(".keras") and f.is_file()
                    for f in os.scandir(entry.path)
                )
                if has_keras:
                    model_folders.append(entry.name)

        return sorted(model_folders)

    # -------------------------------------------------------------------------
    def load_training_configuration(
        self, path: str
    ) -> tuple[dict, TrainingMetadata, dict]:
        config_path = os.path.join(path, "configuration", "configuration.json")
        metadata_path = os.path.join(path, "configuration", "metadata.json")
        history_path = os.path.join(path, "configuration", "session_history.json")
        with open(config_path, encoding="utf-8") as f:
            configuration = json.load(f)
        with open(metadata_path, encoding="utf-8") as f:
            metadata_dict = json.load(f)
            if "dataset_hash" not in metadata_dict:
                alias_value = metadata_dict.get("hashcode") or metadata_dict.get(
                    "hash_code"
                )
                if alias_value:
                    metadata_dict["dataset_hash"] = alias_value
            metadata = TrainingMetadata(**metadata_dict)
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)

        return configuration, metadata, history

    # -------------------------------------------------------------------------
    def load_checkpoint(
        self, checkpoint: str
    ) -> tuple[Any, dict, TrainingMetadata, dict, str]:
        from keras.models import load_model

        custom_objects = {
            "MaskedMeanSquaredError": MaskedMeanSquaredError,
            "MaskedRSquared": MaskedRSquared,
            "LinearDecayLRScheduler": LinearDecayLRScheduler,
            "MolecularEmbedding": MolecularEmbedding,
            "StateEncoder": StateEncoder,
            "PressureSerierEncoder": PressureSerierEncoder,
            "QDecoder": QDecoder,
            "TransformerEncoder": TransformerEncoder,
            "AddNorm": AddNorm,
            "FeedForward": FeedForward,
        }

        checkpoint_path = os.path.join(CHECKPOINTS_PATH, checkpoint)
        model_path = os.path.join(checkpoint_path, "saved_model.keras")
        model = load_model(
            model_path,
            custom_objects=custom_objects,
            compile=True,
        )
        configuration, metadata, session = self.load_training_configuration(
            checkpoint_path
        )

        return model, configuration, metadata, session, checkpoint_path
