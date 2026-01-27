from __future__ import annotations

from typing import Any

import json
import hashlib
import os
from datetime import datetime

import pandas as pd
from keras import Model
from keras.models import load_model

from ADSMOD.server.database.database import database
from ADSMOD.server.schemas.models import MODEL_SCHEMAS
from ADSMOD.server.utils.constants import (
    CHECKPOINTS_PATH,
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
from ADSMOD.server.utils.learning.metrics import (
    MaskedMeanSquaredError,
    MaskedRSquared,
)
from ADSMOD.server.schemas.training import TrainingMetadata
from ADSMOD.server.utils.learning.models.embeddings import MolecularEmbedding
from ADSMOD.server.utils.learning.models.encoders import (
    PressureSerierEncoder,
    QDecoder,
    StateEncoder,
)
from ADSMOD.server.utils.learning.models.transformers import (
    AddNorm,
    FeedForward,
    TransformerEncoder,
)
from ADSMOD.server.utils.learning.training.scheduler import LinearDecayLRScheduler
from ADSMOD.server.utils.logger import logger


###############################################################################
class DataSerializer:
    processed_table = "ADSORPTION_PROCESSED_DATA"
    best_fit_table = "ADSORPTION_BEST_FIT"
    experiment_columns = [
        COLUMN_EXPERIMENT,
        COLUMN_EXPERIMENT_NAME,
        COLUMN_TEMPERATURE_K,
        COLUMN_PRESSURE_PA,
        COLUMN_UPTAKE_MOL_G,
        COLUMN_MEASUREMENT_COUNT,
        COLUMN_MIN_PRESSURE,
        COLUMN_MAX_PRESSURE,
        COLUMN_MIN_UPTAKE,
        COLUMN_MAX_UPTAKE,
    ]

    # -------------------------------------------------------------------------
    def save_raw_dataset(self, dataset: pd.DataFrame) -> None:
        try:
            database.upsert_into_database(dataset, "ADSORPTION_DATA")
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Upsert failed for ADSORPTION_DATA, falling back to merge: %s",
                exc,
            )

        existing = database.load_from_database("ADSORPTION_DATA")
        if existing.empty:
            database.save_into_database(dataset, "ADSORPTION_DATA")
            return

        key_columns = [
            COLUMN_DATASET_NAME,
            COLUMN_EXPERIMENT,
            COLUMN_TEMPERATURE_K,
            COLUMN_PRESSURE_PA,
        ]
        available_keys = [
            col
            for col in key_columns
            if col in dataset.columns and col in existing.columns
        ]
        merged = pd.concat([existing, dataset], ignore_index=True)
        if available_keys:
            merged = merged.drop_duplicates(subset=available_keys, keep="last")
        database.save_into_database(merged, "ADSORPTION_DATA")

    # -------------------------------------------------------------------------
    def load_table(
        self,
        table_name: str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> pd.DataFrame:
        return database.load_from_database(table_name, limit=limit, offset=offset)

    # -------------------------------------------------------------------------
    def upsert_table(self, dataset: pd.DataFrame, table_name: str) -> None:
        try:
            database.upsert_into_database(dataset, table_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Upsert failed for %s, falling back to overwrite: %s",
                table_name,
                exc,
            )
            database.save_into_database(dataset, table_name)

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
        combined = experiments.copy()
        if COLUMN_EXPERIMENT_NAME not in combined.columns:
            return combined
        for schema in MODEL_SCHEMAS.values():
            model_frame = self.load_table(schema["table"])
            if model_frame.empty:
                continue
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
        if COLUMN_EXPERIMENT_NAME not in best.columns:
            return best
        experiments = self.load_table(self.processed_table)
        if experiments.empty:
            return pd.DataFrame()
        drop_columns = [COLUMN_ID]
        merged = experiments.merge(
            best.drop(columns=drop_columns, errors="ignore"),
            how="left",
            on=COLUMN_EXPERIMENT_NAME,
        )
        return merged

    # -------------------------------------------------------------------------
    def build_experiment_frame(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if COLUMN_EXPERIMENT not in dataset.columns:
            raise ValueError("Missing experiment column for fitting results.")
        if COLUMN_EXPERIMENT_NAME not in dataset.columns:
            raise ValueError("Missing experiment name column for fitting results.")
        experiments = dataset.copy()
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
        model_frame[COLUMN_EXPERIMENT_NAME] = dataset.get(COLUMN_EXPERIMENT_NAME)
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
    series_columns = ["pressure", "adsorbed_amount", "adsorbate_encoded_SMILE"]

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
        # Load existing data
        existing = database.load_from_database("TRAINING_DATASET")

        # Filter out rows with the same dataset_label
        if not existing.empty and "dataset_label" in existing.columns:
            existing = existing[existing["dataset_label"] != dataset_label]

        # Append new dataset
        combined = pd.concat([existing, dataset], ignore_index=True)
        database.save_into_database(combined, "TRAINING_DATASET")

    # -------------------------------------------------------------------------
    def save_training_metadata(
        self, metadata: pd.DataFrame, dataset_label: str = "default"
    ) -> None:
        dataset_label = self.normalize_dataset_label(dataset_label)
        # Load existing metadata
        existing = database.load_from_database("TRAINING_METADATA")

        # Filter out row with the same dataset_label (upsert logic)
        if not existing.empty and "dataset_label" in existing.columns:
            existing = existing[existing["dataset_label"] != dataset_label]

        # Append new metadata row
        combined = pd.concat([existing, metadata], ignore_index=True)
        database.save_into_database(combined, "TRAINING_METADATA")

    # -------------------------------------------------------------------------
    def clear_training_dataset(self, dataset_label: str | None = None) -> None:
        if dataset_label is None:
            # Clear all datasets (backward compatibility)
            empty_df = pd.DataFrame()
            database.save_into_database(empty_df, "TRAINING_DATASET")
            database.save_into_database(empty_df, "TRAINING_METADATA")
        else:
            dataset_label = self.normalize_dataset_label(dataset_label)
            # Clear only the specified dataset
            existing_data = database.load_from_database("TRAINING_DATASET")
            existing_meta = database.load_from_database("TRAINING_METADATA")

            if not existing_data.empty and "dataset_label" in existing_data.columns:
                filtered_data = existing_data[
                    existing_data["dataset_label"] != dataset_label
                ]
                database.save_into_database(filtered_data, "TRAINING_DATASET")

            if not existing_meta.empty and "dataset_label" in existing_meta.columns:
                filtered_meta = existing_meta[
                    existing_meta["dataset_label"] != dataset_label
                ]
                database.save_into_database(filtered_meta, "TRAINING_METADATA")

    # -------------------------------------------------------------------------
    def deserialize_series(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Deprecated: JSONSequence type handles this. 
        Kept briefly for compatibility if called explicitly, but effectively a pass-through 
        unless raw strings are still encountered (which JSONSequence also handles).
        """
        if data.empty:
            return data
        # We trust the ORM or the JSONSequence type to have done the work.
        return data

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    @staticmethod
    def _deserialize_value(value: Any) -> Any:
        # Legacy helper, likely unused now
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

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
    def load_training_metadata(
        self, dataset_label: str = "default"
    ) -> TrainingMetadata:
        dataset_label = self.normalize_dataset_label(dataset_label)
        metadata_df = database.load_from_database("TRAINING_METADATA")
        if metadata_df.empty:
            return TrainingMetadata()

        # Filter by dataset_label if column exists
        if "dataset_label" in metadata_df.columns:
            filtered = metadata_df[metadata_df["dataset_label"] == dataset_label]
            if filtered.empty:
                return TrainingMetadata()
            row = filtered.iloc[0]
        else:
            # Backward compatibility: use first row if no dataset_label column
            row = metadata_df.iloc[0]

        smile_vocabulary = self._parse_json(row.get("smile_vocabulary"))
        adsorbent_vocabulary = self._parse_json(row.get("adsorbent_vocabulary"))
        max_smile_index = max(smile_vocabulary.values()) if smile_vocabulary else 0
        smile_vocab_size = int(max_smile_index) + 1
        normalization_stats = self._parse_json(row.get("normalization_stats"))

        metadata = TrainingMetadata(
            created_at=str(row.get("created_at", "")),
            dataset_hash=str(row.get("dataset_hash"))
            if row.get("dataset_hash")
            else None,
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

        return metadata

    # -------------------------------------------------------------------------
    def collect_dataset_hashes(self) -> set[str]:
        metadata_df = database.load_from_database("TRAINING_METADATA")
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

        training_data = database.load_from_database("TRAINING_DATASET")
        if training_data.empty:
            return training_data, training_data, metadata

        # Filter by dataset_label if column exists
        if "dataset_label" in training_data.columns:
            training_data = training_data[
                training_data["dataset_label"] == dataset_label
            ]

        # The JSONSequence type handles deserialization automatically for list columns
        # However, if we need to ensure specific formatting or type coercion for 'split', etc.
        # we can do it here. For now, we assume data comes back mostly correct.
        
        train_data = training_data[training_data["split"] == "train"]
        val_data = training_data[training_data["split"] == "validation"]

        return train_data, val_data, metadata

    # -------------------------------------------------------------------------
    @staticmethod
    def list_processed_datasets() -> list[dict[str, Any]]:
        """
        Returns a list of all processed datasets with their metadata.
        Each entry contains: dataset_label, dataset_hash, train_samples, validation_samples, created_at
        """
        metadata_df = database.load_from_database("TRAINING_METADATA")
        if metadata_df.empty:
            return []

        datasets = []
        for _, row in metadata_df.iterrows():
            datasets.append(
                {
                    "dataset_label": str(row.get("dataset_label", "default")),
                    "dataset_hash": str(row.get("dataset_hash"))
                    if row.get("dataset_hash")
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
        """
        Computes a SHA256 hash of the metadata to ensure strict compatibility.
        Includes all configuration parameters, vocabularies (keys + indices), and statistics.
        """
        if not metadata:
            return ""

        # specialized serialization for hashing
        payload = {
            "sample_size": metadata.sample_size,
            "validation_size": metadata.validation_size,
            "min_measurements": metadata.min_measurements,
            "max_measurements": metadata.max_measurements,
            "smile_sequence_size": metadata.smile_sequence_size,
            "max_pressure": metadata.max_pressure,
            "max_uptake": metadata.max_uptake,
            # Sort dictionaries to ensure deterministic hashing
            # Vocabularies must include both keys and values (indices)
            "smile_vocabulary": sorted(metadata.smile_vocabulary.items())
            if metadata.smile_vocabulary
            else [],
            "adsorbent_vocabulary": sorted(metadata.adsorbent_vocabulary.items())
            if metadata.adsorbent_vocabulary
            else [],
            # normalization stats
            "normalization_stats": metadata.normalization_stats,
        }

        # Serialize to JSON with sort_keys=True
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

        # Strict validation: Re-compute hashes for both and compare.
        # This ensures that even if the stored hash was manipulated or outdated,
        # the actual content compatibility is verified.
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
    def save_pretrained_model(self, model: Model, path: str) -> None:
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
                alias_value = (
                    metadata_dict.get("hashcode") or metadata_dict.get("hash_code")
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
    ) -> tuple[Model | Any, dict, TrainingMetadata, dict, str]:
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
        model = load_model(model_path, custom_objects=custom_objects)
        configuration, metadata, session = self.load_training_configuration(
            checkpoint_path
        )

        return model, configuration, metadata, session, checkpoint_path
