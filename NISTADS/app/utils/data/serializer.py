from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import pandas as pd
from keras import Model
from keras.models import load_model
from keras.utils import plot_model

from NISTADS.app.constants import CHECKPOINT_PATH, PROCESS_METADATA_FILE
from NISTADS.app.logger import logger
from NISTADS.app.utils.data.database import database
from NISTADS.app.utils.learning.metrics import MaskedMeanSquaredError, MaskedRSquared
from NISTADS.app.utils.learning.training.scheduler import LinearDecayLRScheduler


# [DATA SERIALIZATION]
###############################################################################
class DataSerializer:
    def __init__(self) -> None:
        self.P_COL = "pressure"
        self.Q_COL = "adsorbed_amount"
        self.series_cols = [self.P_COL, self.Q_COL, "adsorbate_encoded_SMILE"]

    # -------------------------------------------------------------------------
    def validate_metadata(
        self, metadata: dict[str, Any] | Any, target_metadata: dict[str, Any]
    ) -> bool:
        keys_to_compare = [k for k in metadata if k != "date"]
        meta_current = {k: metadata.get(k) for k in keys_to_compare}
        meta_target = {k: target_metadata.get(k) for k in keys_to_compare}
        differences = {
            k: (meta_current[k], meta_target[k])
            for k in keys_to_compare
            if meta_current[k] != meta_target[k]
        }

        return False if differences else True

    # -------------------------------------------------------------------------
    def serialize_series(self, data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        for col in columns:
            data[col] = data[col].apply(
                lambda x: " ".join(map(str, x))
                if isinstance(x, list)
                else [float(i) for i in x.split()]
                if isinstance(x, str)
                else x
            )

        return data

    # -------------------------------------------------------------------------
    def load_adsorption_datasets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        adsorption_data = database.load_from_database("SINGLE_COMPONENT_ADSORPTION")
        guest_data = database.load_from_database("ADSORBATES")
        host_data = database.load_from_database("ADSORBENTS")

        return adsorption_data, guest_data, host_data

    # -------------------------------------------------------------------------
    def load_training_data(
        self, only_metadata: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict] | dict:
        # load metadata from file
        with open(PROCESS_METADATA_FILE) as file:
            metadata = json.load(file)

        if not only_metadata:
            # load preprocessed data from database and convert joint strings to list
            training_data = database.load_from_database("TRAINING_DATASET")
            training_data = self.serialize_series(training_data, self.series_cols)
            train_data = training_data[training_data["split"] == "train"]
            val_data = training_data[training_data["split"] == "validation"]

            return train_data, val_data, metadata

        return metadata

    # -------------------------------------------------------------------------
    def load_inference_dataset(self) -> pd.DataFrame:
        dataset = database.load_from_database("PREDICTED_ADSORPTION")
        return dataset

    # -------------------------------------------------------------------------
    def save_training_data(
        self,
        data: pd.DataFrame,
        configuration: dict[str, Any],
        smile_vocabulary: dict[str, Any],
        ads_vocabulary: dict[str, Any],
        normalization_stats: dict[str, Any] | Any = {},
    ) -> None:
        # convert list to joint string and save preprocessed data to database
        validated_data = self.serialize_series(data, self.series_cols)
        database.save_into_database(validated_data, "TRAINING_DATASET")
        metadata = {
            "seed": configuration.get("seed", 42),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "sample_size": configuration.get("sample_size", 1.0),
            "validation_size": configuration.get("validation_size", 0.2),
            "split_seed": configuration.get("split_seed", 42),
            "max_measurements": configuration.get("max_measurements", 1000),
            "SMILE_sequence_size": configuration.get("SMILE_sequence_size", 30),
            "SMILE_vocabulary_size": len(smile_vocabulary),
            "adsorbent_vocabulary_size": len(ads_vocabulary),
            "normalization": {
                self.P_COL: float(normalization_stats[self.P_COL]),
                self.Q_COL: float(normalization_stats[self.Q_COL]),
                "temperature": float(normalization_stats["temperature"]),
                "adsorbate_molecular_weight": float(
                    normalization_stats["adsorbate_molecular_weight"]
                ),
            },
            "SMILE_vocabulary": smile_vocabulary,
            "adsorbent_vocabulary": ads_vocabulary,
        }

        with open(PROCESS_METADATA_FILE, "w") as file:
            json.dump(metadata, file, indent=4)

    # -------------------------------------------------------------------------
    def save_materials_datasets(self, guest_data=None, host_data=None) -> None:
        if guest_data:
            database.upsert_into_database(guest_data, "ADSORBATES")
        if host_data:
            database.upsert_into_database(host_data, "ADSORBENTS")

    # -------------------------------------------------------------------------
    def save_adsorption_datasets(
        self, single_component: pd.DataFrame, binary_mixture: pd.DataFrame
    ):
        database.upsert_into_database(single_component, "SINGLE_COMPONENT_ADSORPTION")
        database.upsert_into_database(binary_mixture, "BINARY_MIXTURE_ADSORPTION")

    # -------------------------------------------------------------------------
    def save_predictions_dataset(self, data: pd.DataFrame) -> None:
        database.save_into_database(data, "PREDICTED_ADSORPTION")

    # -------------------------------------------------------------------------
    def save_checkpoints_summary(self, data: pd.DataFrame) -> None:
        database.upsert_into_database(data, "CHECKPOINTS_SUMMARY")


# [MODEL SERIALIZATION]
###############################################################################
class ModelSerializer:
    def __init__(self) -> None:
        self.model_name = "SCADS"

    # function to create a folder where to save model checkpoints
    # -------------------------------------------------------------------------
    def create_checkpoint_folder(self) -> str:
        today_datetime = datetime.now().strftime("%Y%m%dT%H%M%S")
        checkpoint_path = os.path.join(
            CHECKPOINT_PATH, f"{self.model_name}_{today_datetime}"
        )
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(os.path.join(checkpoint_path, "configuration"), exist_ok=True)
        logger.debug(f"Created checkpoint folder at {checkpoint_path}")

        return checkpoint_path

    # -------------------------------------------------------------------------
    def save_pretrained_model(self, model: Model, path) -> None:
        model_files_path = os.path.join(path, "saved_model.keras")
        model.save(model_files_path)
        logger.info(
            f"Training session is over. Model {os.path.basename(path)} has been saved"
        )

    # -------------------------------------------------------------------------
    def save_training_configuration(
        self, path, history: dict, configuration: dict[str, Any], metadata: dict
    ) -> None:
        config_path = os.path.join(path, "configuration", "configuration.json")
        metadata_path = os.path.join(path, "configuration", "metadata.json")
        history_path = os.path.join(path, "configuration", "session_history.json")

        # Save training and model configuration
        with open(config_path, "w") as f:
            json.dump(configuration, f)
        # Save metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        # Save session history
        with open(history_path, "w") as f:
            json.dump(history, f)

        logger.debug(
            f"Model configuration, session history and metadata saved for {os.path.basename(path)}"
        )

    # -------------------------------------------------------------------------
    def scan_checkpoints_folder(self) -> list[str]:
        model_folders = []
        for entry in os.scandir(CHECKPOINT_PATH):
            if entry.is_dir():
                # Check if the folder contains at least one .keras file
                has_keras = any(
                    f.name.endswith(".keras") and f.is_file()
                    for f in os.scandir(entry.path)
                )
                if has_keras:
                    model_folders.append(entry.name)

        return model_folders

    # -------------------------------------------------------------------------
    def load_training_configuration(self, path: str) -> tuple[dict, dict, dict]:
        config_path = os.path.join(path, "configuration", "configuration.json")
        metadata_path = os.path.join(path, "configuration", "metadata.json")
        history_path = os.path.join(path, "configuration", "session_history.json")
        # Load training and model configuration
        with open(config_path) as f:
            configuration = json.load(f)
        # Load metadata
        with open(metadata_path) as f:
            metadata = json.load(f)
        # Load session history
        with open(history_path) as f:
            history = json.load(f)

        return configuration, metadata, history

    # -------------------------------------------------------------------------
    def save_model_plot(self, model: Model, path: str) -> None:
        try:
            plot_path = os.path.join(path, "model_layout.png")
            plot_model(
                model,
                to_file=plot_path,
                show_shapes=True,
                show_layer_names=True,
                show_layer_activations=True,
                expand_nested=True,
                rankdir="TB",
                dpi=400,
            )
            logger.debug(f"Model architecture plot generated as {plot_path}")
        except (OSError, FileNotFoundError, ImportError):
            logger.warning(
                "Could not generate model architecture plot (graphviz/pydot not correctly installed)"
            )

    # -------------------------------------------------------------------------
    def load_checkpoint(
        self, checkpoint: str
    ) -> tuple[Model | Any, dict, dict, dict, str]:
        custom_objects = {
            "MaskedSparseCategoricalCrossentropy": MaskedMeanSquaredError,
            "MaskedAccuracy": MaskedRSquared,
            "LinearDecayLRScheduler": LinearDecayLRScheduler,
        }

        checkpoint_path = os.path.join(CHECKPOINT_PATH, checkpoint)
        model_path = os.path.join(checkpoint_path, "saved_model.keras")
        model = load_model(model_path, custom_objects=custom_objects)
        configuration, metadata, session = self.load_training_configuration(
            checkpoint_path
        )

        return model, configuration, metadata, session, checkpoint_path
