from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from keras import Model
from keras.utils import set_random_seed

from NISTADS.app.constants import PAD_VALUE
from NISTADS.app.utils.data.loader import SCADSDataLoader
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
    ) -> None:
        set_random_seed(metadata.get("seed", 42))
        self.checkpoint = os.path.basename(checkpoint_path)
        self.configuration = configuration
        self.metadata = metadata
        self.model = model

    # -------------------------------------------------------------------------
    def process_inference_output(
        self, inputs: dict, predictions: np.ndarray
    ) -> list[Any]:
        # reshape predictions from (samples, measurements, 1) to (samples, measurements)
        predictions = np.squeeze(predictions, axis=-1)
        pressure = inputs["pressure_input"]
        unpadded_predictions = []

        # Reverse each row to get pad values as leading values
        # Create a boolean mask to set pad values to False
        flipped = np.flip(pressure, axis=1)
        true_values_mask = flipped != PAD_VALUE

        # Find the index of the first true value, or set to the full length if all values are padded
        trailing_counts = np.where(
            true_values_mask.any(axis=1),
            true_values_mask.argmax(axis=1),
            pressure.shape[1],
        )

        # trim the predicted sequences based on true values number
        unpadded_length = pressure.shape[1] - trailing_counts
        unpadded_predictions = [
            pred_row[:length] for pred_row, length in zip(predictions, unpadded_length)
        ]

        return unpadded_predictions

    # -------------------------------------------------------------------------
    def predict_adsorption_isotherm(self, data: pd.DataFrame, **kwargs) -> list[Any]:
        # preprocess inputs before feeding them to the pretrained model for inference
        # add padding, normalize data, encode categoricals
        dataloader = SCADSDataLoader(self.configuration, self.metadata, shuffle=False)
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
