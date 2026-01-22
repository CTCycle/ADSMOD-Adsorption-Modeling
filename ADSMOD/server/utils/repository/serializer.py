from __future__ import annotations

from typing import Any

import json
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
from ADSMOD.server.utils.learning.models.embeddings import MolecularEmbedding
from ADSMOD.server.utils.learning.models.encoders import (
    PressureSerierEncoder,
    QDecoder,
    StateEncoder,
)
from ADSMOD.server.utils.learning.models.transformers import AddNorm, FeedForward, TransformerEncoder
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
    def load_table(self, table_name: str) -> pd.DataFrame:
        return database.load_from_database(table_name)

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
        encoded = self.convert_lists_to_strings(dataset)
        experiments = self.build_experiment_frame(encoded)
        self.upsert_table(experiments, self.processed_table)
        for schema in MODEL_SCHEMAS.values():
            model_frame = self.build_model_frame(encoded, schema)
            if model_frame is None:
                continue
            self.upsert_table(model_frame, schema["table"])

    # -------------------------------------------------------------------------
    def load_fitting_results(self) -> pd.DataFrame:
        experiments = self.load_table(self.processed_table)
        if experiments.empty:
            return experiments
        experiments = self.convert_strings_to_lists(experiments)
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
            combined = combined.merge(
                renamed, how="left", on=COLUMN_EXPERIMENT_NAME
            )
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
        experiments = self.convert_strings_to_lists(experiments)
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
            field: self.resolve_dataset_column(schema["prefix"], suffix, dataset.columns)
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

    # -------------------------------------------------------------------------
    def convert_list_to_string(self, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            parts: list[str] = []
            for element in value:
                if element is None:
                    continue
                text = str(element)
                if text:
                    parts.append(text)
            return ",".join(parts)
        return value

    # -------------------------------------------------------------------------
    def convert_string_to_list(self, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, list):
                    return parsed
            parts = [segment.strip() for segment in stripped.split(",")]
            converted: list[float] = []
            for part in parts:
                if not part:
                    continue
                try:
                    converted.append(float(part))
                except ValueError:
                    return value
            return converted
        return value

    # -------------------------------------------------------------------------
    def convert_lists_to_strings(self, dataset: pd.DataFrame) -> pd.DataFrame:
        converted = dataset.copy()
        for column in converted.columns:
            converted[column] = converted[column].apply(self.convert_list_to_string)
        return converted

    # -------------------------------------------------------------------------
    def convert_strings_to_lists(self, dataset: pd.DataFrame) -> pd.DataFrame:
        converted = dataset.copy()
        for column in converted.columns:
            if converted[column].dtype == object:
                converted[column] = converted[column].apply(self.convert_string_to_list)
        return converted


###############################################################################
class TrainingDataSerializer:
    series_columns = ["pressure", "adsorbed_amount", "adsorbate_encoded_SMILE"]

    # -------------------------------------------------------------------------
    def save_training_dataset(self, dataset: pd.DataFrame) -> None:
        database.save_into_database(dataset, "TRAINING_DATASET")

    # -------------------------------------------------------------------------
    def save_training_metadata(self, metadata: pd.DataFrame) -> None:
        database.save_into_database(metadata, "TRAINING_METADATA")

    # -------------------------------------------------------------------------
    def clear_training_dataset(self) -> None:
        empty_df = pd.DataFrame()
        database.save_into_database(empty_df, "TRAINING_DATASET")
        database.save_into_database(empty_df, "TRAINING_METADATA")

    # -------------------------------------------------------------------------
    def deserialize_series(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return data
        parsed = data.copy()
        for col in self.series_columns:
            if col not in parsed.columns:
                continue
            parsed[col] = parsed[col].apply(self._deserialize_value)
        return parsed

    # -------------------------------------------------------------------------
    @staticmethod
    def _deserialize_value(value: Any) -> Any:
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
    def load_training_metadata(self) -> dict[str, Any]:
        metadata_df = database.load_from_database("TRAINING_METADATA")
        if metadata_df.empty:
            return {}

        row = metadata_df.iloc[0]
        smile_vocabulary = self._parse_json(row.get("smile_vocabulary"))
        adsorbent_vocabulary = self._parse_json(row.get("adsorbent_vocabulary"))
        normalization_stats = self._parse_json(row.get("normalization_stats"))

        metadata = {
            "created_at": row.get("created_at", ""),
            "sample_size": row.get("sample_size", 1.0),
            "validation_size": row.get("validation_size", 0.2),
            "min_measurements": row.get("min_measurements", 1),
            "max_measurements": row.get("max_measurements", 30),
            "smile_sequence_size": row.get("smile_sequence_size", 20),
            "max_pressure": row.get("max_pressure", 10000.0),
            "max_uptake": row.get("max_uptake", 20.0),
            "total_samples": row.get("total_samples", 0),
            "train_samples": row.get("train_samples", 0),
            "validation_samples": row.get("validation_samples", 0),
            "smile_vocabulary": smile_vocabulary,
            "adsorbent_vocabulary": adsorbent_vocabulary,
            "normalization": normalization_stats,
            "normalization_stats": normalization_stats,
            "smile_vocabulary_size": len(smile_vocabulary),
            "adsorbent_vocabulary_size": len(adsorbent_vocabulary),
            "SMILE_sequence_size": row.get("smile_sequence_size", 20),
            "SMILE_vocabulary": smile_vocabulary,
            "SMILE_vocabulary_size": len(smile_vocabulary),
        }

        return metadata

    # -------------------------------------------------------------------------
    def load_training_data(
        self, only_metadata: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]] | dict[str, Any]:
        metadata = self.load_training_metadata()
        if only_metadata:
            return metadata

        training_data = database.load_from_database("TRAINING_DATASET")
        if training_data.empty:
            return training_data, training_data, metadata
        training_data = self.deserialize_series(training_data)
        train_data = training_data[training_data["split"] == "train"]
        val_data = training_data[training_data["split"] == "validation"]

        return train_data, val_data, metadata

    # -------------------------------------------------------------------------
    @staticmethod
    def validate_metadata(
        metadata: dict[str, Any] | Any, target_metadata: dict[str, Any]
    ) -> bool:
        if not metadata or not target_metadata:
            return False
        keys_to_compare = [k for k in metadata if k not in {"created_at"}]
        meta_current = {k: metadata.get(k) for k in keys_to_compare}
        meta_target = {k: target_metadata.get(k) for k in keys_to_compare}
        differences = {
            k: (meta_current[k], meta_target[k])
            for k in keys_to_compare
            if meta_current[k] != meta_target[k]
        }

        return False if differences else True


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
        self, path: str, history: dict, configuration: dict[str, Any], metadata: dict
    ) -> None:
        config_path = os.path.join(path, "configuration", "configuration.json")
        metadata_path = os.path.join(path, "configuration", "metadata.json")
        history_path = os.path.join(path, "configuration", "session_history.json")

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(configuration, f, indent=4, default=str)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, default=str)
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
    def load_training_configuration(self, path: str) -> tuple[dict, dict, dict]:
        config_path = os.path.join(path, "configuration", "configuration.json")
        metadata_path = os.path.join(path, "configuration", "metadata.json")
        history_path = os.path.join(path, "configuration", "session_history.json")
        with open(config_path, encoding="utf-8") as f:
            configuration = json.load(f)
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)

        return configuration, metadata, history

    # -------------------------------------------------------------------------
    def load_checkpoint(
        self, checkpoint: str
    ) -> tuple[Model | Any, dict, dict, dict, str]:
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
