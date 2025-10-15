from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from keras import Model
from keras.utils import set_random_seed

from NISTADS.app.constants import PAD_VALUE, SCADS_ATOMIC_MODEL
from NISTADS.app.utils.services.loader import SCADSDataLoader
from NISTADS.app.utils.learning.callbacks import LearningInterruptCallback


# [INFERENCE]
###############################################################################
class AdsorptionPredictions:
    def __init__(
        self,
        model: Model,
        configuration: dict[str, Any],
        metadata: dict,
        checkpoint_path: str,
        dataloader_factory: Any,
        model_name: str,
    ) -> None:
        set_random_seed(metadata.get("seed", 42))
        self.checkpoint = os.path.basename(checkpoint_path)
        self.configuration = configuration
        self.metadata = metadata
        self.model = model
        self.model_name = model_name
        self.dataloader_factory = dataloader_factory or SCADSDataLoader

    # -------------------------------------------------------------------------
    def process_inference_output(
        self, inputs: dict, predictions: np.ndarray
    ) -> list[Any]:
        if self.model_name == SCADS_ATOMIC_MODEL:
            single_values = np.asarray(predictions, dtype=np.float32).reshape(-1, 1)
            return [row for row in single_values]

        predictions = np.squeeze(predictions, axis=-1)
        pressure = inputs["pressure_input"]
        flipped = np.flip(pressure, axis=1)
        true_values_mask = flipped != PAD_VALUE

        trailing_counts = np.where(
            true_values_mask.any(axis=1),
            true_values_mask.argmax(axis=1),
            pressure.shape[1],
        )

        unpadded_length = pressure.shape[1] - trailing_counts
        unpadded_predictions = [
            pred_row[:length] for pred_row, length in zip(predictions, unpadded_length)
        ]

        return unpadded_predictions

    # -------------------------------------------------------------------------
    def predict_adsorption_isotherm(self, data: pd.DataFrame, **kwargs) -> list[Any]:
        # preprocess inputs before feeding them to the pretrained model for inference
        # add padding, normalize data, encode categoricals
        dataloader = self.dataloader_factory(
            self.configuration, self.metadata, shuffle=False
        )
        processed_inputs = dataloader.process_inference_inputs(data)
        processed_inputs, _ = dataloader.separate_inputs_and_output(processed_inputs)
        # add interruption callback to stop model predictions if requested
        callbacks_list = [LearningInterruptCallback(kwargs.get("worker", None))]
        # perform prediction of adsorption isotherm sequences
        predictions = self.model.predict(
            processed_inputs,
            verbose=1,  # type: ignore
            callbacks=callbacks_list,
        )
        # postprocess obtained outputs
        # remove padding, rescale, decode categoricals
        predictions = self.process_inference_output(processed_inputs, predictions)

        return predictions

    # -------------------------------------------------------------------------
    def build_predictions_dataset(
        self, data, predictions: list[np.ndarray | Any]
    ) -> Any:
        concat_predictions = np.concatenate(predictions)
        # TO DO ADD COLUMN FOR CHECKPOINT, FIND WAY TO AVOID FETCHING REDUNDANT DATA
        data["predicted_adsorbed_amount"] = concat_predictions

        return data
