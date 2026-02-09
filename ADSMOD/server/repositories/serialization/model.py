from __future__ import annotations

from typing import Any

import json
import os
import shutil
from datetime import datetime

from ADSMOD.server.entities.training import TrainingMetadata
from ADSMOD.server.learning.metrics import (
    MaskedMeanSquaredError,
    MaskedRSquared,
)
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
from ADSMOD.server.common.constants import CHECKPOINTS_PATH
from ADSMOD.server.common.utils.logger import logger


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


