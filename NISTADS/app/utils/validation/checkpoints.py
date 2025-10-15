from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model

from NISTADS.app.client.workers import check_thread_status, update_progress_callback
from NISTADS.app.constants import CHECKPOINT_PATH
from NISTADS.app.logger import logger
from NISTADS.app.utils.repository.serializer import DataSerializer, ModelSerializer
from NISTADS.app.utils.learning.callbacks import LearningInterruptCallback


# [LOAD MODEL]
################################################################################
class ModelEvaluationSummary:
    def __init__(
        self, configuration: dict[str, Any], model: Model | None = None
    ) -> None:
        self.serializer = DataSerializer()
        self.modser = ModelSerializer()
        self.model = model
        self.configuration = configuration

    # --------------------------------------------------------------------------
    def scan_checkpoint_folder(self) -> list[str]:
        model_paths = []
        for entry in os.scandir(CHECKPOINT_PATH):
            if entry.is_dir():
                pretrained_model_path = os.path.join(entry.path, "saved_model.keras")
                if os.path.isfile(pretrained_model_path):
                    model_paths.append(entry.path)

        return model_paths

    # --------------------------------------------------------------------------
    def get_checkpoints_summary(self, **kwargs) -> pd.DataFrame:
        # look into checkpoint folder to get pretrained model names
        model_paths = self.scan_checkpoint_folder()
        model_parameters = []
        for i, model_path in enumerate(model_paths):
            _ = self.modser.load_checkpoint(model_path)
            configuration, metadata, history = self.modser.load_training_configuration(
                model_path
            )
            model_name = os.path.basename(model_path)
            precision = 16 if configuration.get("use_mixed_precision", np.nan) else 32
            has_scheduler = configuration.get("use_scheduler", False)
            scores = history.get("history", {})
            chkp_config = {
                "checkpoint": model_name,
                "sample_size": metadata.get("sample_size", np.nan),
                "validation_size": metadata.get("validation_size", np.nan),
                "seed": configuration.get("train_seed", np.nan),
                "precision": precision,
                "epochs": history.get("epochs", np.nan),
                "batch_size": configuration.get("batch_size", np.nan),
                "split_seed": metadata.get("split_seed", np.nan),
                "jit_compile": configuration.get("jit_compile", np.nan),
                "has_tensorboard_logs": configuration.get("use_tensorboard", np.nan),
                "initial_LR": configuration.get("initial_LR", np.nan),
                "constant_steps_LR": configuration.get("constant_steps", np.nan)
                if has_scheduler
                else np.nan,
                "decay_steps_LR": configuration.get("decay_steps", np.nan)
                if has_scheduler
                else np.nan,
                "target_LR": configuration.get("target_LR", np.nan)
                if has_scheduler
                else np.nan,
                "max_measurements": configuration.get("max_measurements", np.nan),
                "SMILE_size": configuration.get("SMILE_size", np.nan),
                "attention_heads": configuration.get("attention_heads", np.nan),
                "n_encoders": configuration.get("num_encoders", np.nan),
                "embedding_dimensions": configuration.get(
                    "embedding_dimensions", np.nan
                ),
                "train_loss": scores.get("loss", [np.nan])[-1],
                "val_loss": scores.get("val_loss", [np.nan])[-1],
                "train_R_square": scores.get("MaskedR2", [np.nan])[-1],
                "val_R_square": scores.get("val_MaskedR2", [np.nan])[-1],
            }

            model_parameters.append(chkp_config)

            # check for thread status and progress bar update
            check_thread_status(kwargs.get("worker", None))
            update_progress_callback(
                i + 1, len(model_paths), kwargs.get("progress_callback", None)
            )

        dataframe = pd.DataFrame(model_parameters)
        self.serializer.save_checkpoints_summary(dataframe)

        return dataframe

    # -------------------------------------------------------------------------
    def get_evaluation_report(
        self, model: Model, validation_dataset: tf.data.Dataset | np.ndarray, **kwargs
    ) -> None:
        callbacks_list = [LearningInterruptCallback(kwargs.get("worker", None))]
        validation = model.evaluate(
            validation_dataset,
            verbose=1,
            callbacks=callbacks_list,  # type: ignore
        )
        logger.info(
            f"Mean Square Error Loss {validation[0]:.3f} - R square metrics {validation[1]:.3f}"
        )
