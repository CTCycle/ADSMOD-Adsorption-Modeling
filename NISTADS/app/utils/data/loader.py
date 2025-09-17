from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

from NISTADS.app.constants import PAD_VALUE
from NISTADS.app.logger import logger
from NISTADS.app.utils.data.serializer import DataSerializer
from NISTADS.app.utils.processing.sanitizer import AggregateDatasets


# [CUSTOM DATA GENERATOR FOR TRAINING]
###############################################################################
class DataLoaderProcessor:
    def __init__(self, configuration: dict[str, Any], metadata: dict) -> None:
        # load source datasets to obtain the guest and host data references
        # then load the metadata from the processed dataset. At any time,
        # only a single instance of the processed dataset may exist, therefor
        # the user should be careful about loading a model trained on a different dataset
        self.normalization_config = metadata.get("normalization", {})
        self.series_length = metadata.get("max_measurements", 30)
        self.smile_length = metadata.get("SMILE_sequence_size", 30)
        self.SMILE_vocab = metadata.get("SMILE_vocabulary", {})
        self.adsorbent_vocab = metadata.get("adsorbent_vocabulary", {})
        self.serializer = DataSerializer()
        self.configuration = configuration

    # this method is tailored on the inference input dataset, which is provided
    # with pressure already converted to Pascal and fewer columns compared to source data
    # -------------------------------------------------------------------------
    def aggregate_inference_data(self, dataset: pd.DataFrame) -> pd.DataFrame:
        aggregate_dict = {
            "temperature": "first",
            "adsorbent_name": "first",
            "adsorbate_name": "first",
            "pressure": lambda x: [float(v) for v in x],
        }

        grouped_data = dataset.groupby(by="filename").agg(aggregate_dict).reset_index()
        grouped_data.drop(columns=["filename"], inplace=True)

        return grouped_data

    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    # -------------------------------------------------------------------------
    def add_properties_to_inference_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        _, guest_data, host_data = self.serializer.load_adsorption_datasets()
        aggregator = AggregateDatasets(self.configuration)
        processed_data = aggregator.join_materials_properties(
            data, guest_data, host_data
        )

        return processed_data

    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    # -------------------------------------------------------------------------
    def remove_invalid_measurements(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data[data["temperature"] >= 0]
        data = data[
            (data["pressure"] >= 0)
            & (data["pressure"] <= self.normalization_config["pressure"])
        ]

        return data

    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    # -------------------------------------------------------------------------
    def normalize_from_references(self, data: pd.DataFrame) -> pd.DataFrame:
        data["temperature"] = (
            data["temperature"] / self.normalization_config["temperature"]
        )
        data["adsorbate_molecular_weight"] = (
            data["adsorbate_molecular_weight"]
            / self.normalization_config["adsorbate_molecular_weight"]
        )
        data["pressure"] = data["pressure"].apply(
            lambda x: [s / self.normalization_config["pressure"] for s in x]
        )

        return data

    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    # -------------------------------------------------------------------------
    def encode_SMILE_from_vocabulary(self, smile: str) -> list[Any]:
        encoded_tokens = []
        i = 0
        # Sort tokens by descending length to prioritize multi-character tokens
        sorted_tokens = sorted(self.SMILE_vocab.keys(), key=len, reverse=True)
        while i < len(smile):
            matched = False
            for token in sorted_tokens:
                if smile[i : i + len(token)] == token:
                    encoded_tokens.append(self.SMILE_vocab[token])
                    i += len(token)
                    matched = True
                    break
            if not matched:
                logger.warning(
                    f"SMILE Tokenization error: no valid token found in '{smile}' at position {i}"
                )

        return encoded_tokens

    # -------------------------------------------------------------------------
    def encode_from_references(self, data: pd.DataFrame) -> pd.DataFrame:
        data["adsorbate_encoded_SMILE"] = data["adsorbate_SMILE"].apply(
            lambda x: self.encode_SMILE_from_vocabulary(x)
        )
        data["encoded_adsorbent"] = (
            data["adsorbent_name"].str.lower().map(self.adsorbent_vocab)
        )

        return data

    # -------------------------------------------------------------------------
    def apply_padding(self, data: pd.DataFrame) -> pd.DataFrame:
        data["pressure"] = pad_sequences(
            data["pressure"],
            maxlen=self.series_length,
            value=PAD_VALUE,
            dtype="float32",
            padding="post",
        ).tolist()

        data["adsorbate_encoded_SMILE"] = pad_sequences(
            data["adsorbate_encoded_SMILE"],
            maxlen=self.smile_length,
            value=PAD_VALUE,
            dtype="float32",
            padding="post",
        ).tolist()

        return data


# wrapper function to run the data pipeline from raw inputs to tensor dataset
###############################################################################
class SCADSDataLoader:
    def __init__(
        self,
        configuration: dict[str, Any],
        metadata: dict[str, Any],
        shuffle: bool = True,
    ) -> None:
        self.processor = DataLoaderProcessor(configuration, metadata)
        self.batch_size = configuration.get("batch_size", 32)
        self.inference_batch_size = configuration.get("inference_batch_size", 32)
        self.shuffle_samples = configuration.get("shuffle_size", 1024)
        self.buffer_size = tf.data.AUTOTUNE
        self.metadata = metadata
        self.configuration = configuration
        self.shuffle = shuffle
        self.output = "adsorbed_amount"

    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    # -------------------------------------------------------------------------
    def separate_inputs_and_output(
        self, data: pd.DataFrame
    ) -> tuple[dict, np.ndarray | None]:
        state = data["temperature"].to_numpy(dtype=np.float32)
        chemo = data["adsorbate_molecular_weight"].to_numpy(dtype=np.float32)
        adsorbent = data["encoded_adsorbent"].to_numpy(dtype=np.float32)
        # vstack arrays for columns that contain sequences
        adsorbate = np.vstack(
            [
                np.asarray(x, dtype=np.float32)
                for x in data["adsorbate_encoded_SMILE"].to_list()
            ]
        )
        pressure = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in data["pressure"].to_list()]
        )
        inputs_dict = {
            "state_input": state,
            "chemo_input": chemo,
            "adsorbent_input": adsorbent,
            "adsorbate_input": adsorbate,
            "pressure_input": pressure,
        }

        # output is reshaped to match the expected shape of the model
        # (batch size, pressure points, 1)
        output = None
        if self.output in data.columns:
            output = data[self.output]
            output = np.vstack([np.asarray(x, dtype=np.float32) for x in output.values])

        return inputs_dict, output

    # -------------------------------------------------------------------------
    def process_inference_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        processed_data = self.processor.remove_invalid_measurements(data)
        processed_data = self.processor.aggregate_inference_data(processed_data)
        processed_data = self.processor.add_properties_to_inference_inputs(
            processed_data
        )
        processed_data = self.processor.encode_from_references(processed_data)
        processed_data = self.processor.normalize_from_references(processed_data)
        # add padding to pressure and uptake series to match max length
        processed_data = self.processor.apply_padding(processed_data)

        return processed_data

    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    # -------------------------------------------------------------------------
    def build_training_dataloader(
        self, data, batch_size=None, buffer_size: int = tf.data.AUTOTUNE
    ) -> Any:
        batch_size = self.batch_size if batch_size is None else batch_size
        inputs, output = self.separate_inputs_and_output(data)
        dataset = tf.data.Dataset.from_tensor_slices((inputs, output))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)
        dataset = (
            dataset.shuffle(buffer_size=self.shuffle_samples)
            if self.shuffle
            else dataset
        )

        return dataset

    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    # -------------------------------------------------------------------------
    def build_inference_dataloader(
        self,
        data: pd.DataFrame,
        batch_size: int | None = None,
        buffer_size: int = tf.data.AUTOTUNE,
    ) -> Any:
        batch_size = self.inference_batch_size if batch_size is None else batch_size
        processed_data = self.process_inference_inputs(data)
        inputs, output = self.separate_inputs_and_output(processed_data)
        dataset = tf.data.Dataset.from_tensor_slices((inputs, output))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)

        return dataset


################################################################################
class SCADSAtomicDataLoader:
    def __init__(
        self,
        configuration: dict[str, Any],
        metadata: dict[str, Any],
        shuffle: bool = True,
    ) -> None:
        self.processor = DataLoaderProcessor(configuration, metadata)
        self.batch_size = configuration.get("batch_size", 32)
        self.inference_batch_size = configuration.get("inference_batch_size", 32)
        self.shuffle_samples = configuration.get("shuffle_size", 1024)
        self.buffer_size = tf.data.AUTOTUNE
        self.metadata = metadata
        self.configuration = configuration
        self.shuffle = shuffle
        self.output = "adsorbed_amount"
        self.smile_length = metadata.get("SMILE_sequence_size", 20)

    # -------------------------------------------------------------------------
    def expand_to_single_measurements(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            columns = list(data.columns)
            return pd.DataFrame(columns=columns)

        non_sequence_cols = [
            col for col in data.columns if col not in {"pressure", self.output}
        ]
        has_output = self.output in data.columns
        expanded_rows: list[dict[str, Any]] = []

        for _, row in data.iterrows():
            base = {col: row[col] for col in non_sequence_cols}
            pressure_values = row.get("pressure", [])
            if isinstance(pressure_values, np.ndarray):
                pressure_values = pressure_values.tolist()
            elif not isinstance(pressure_values, (list, tuple)):
                pressure_values = [pressure_values]

            uptake_values: list[Any]
            if has_output:
                uptake_values = row.get(self.output, [])
                if isinstance(uptake_values, np.ndarray):
                    uptake_values = uptake_values.tolist()
                elif not isinstance(uptake_values, (list, tuple)):
                    uptake_values = [uptake_values] * len(pressure_values)
            else:
                uptake_values = [None] * len(pressure_values)

            for pressure_value, uptake_value in zip(pressure_values, uptake_values):
                if pd.isna(pressure_value) or pressure_value == PAD_VALUE:
                    continue
                if has_output and uptake_value is not None and pd.isna(uptake_value):
                    continue
                entry = base.copy()
                entry["pressure"] = float(pressure_value)
                if (
                    has_output
                    and uptake_value is not None
                    and uptake_value != PAD_VALUE
                ):
                    entry[self.output] = float(uptake_value)
                expanded_rows.append(entry)

        expanded_df = pd.DataFrame(expanded_rows)
        if expanded_df.empty:
            columns = non_sequence_cols + ["pressure"]
            if has_output:
                columns.append(self.output)
            expanded_df = pd.DataFrame(columns=columns)
            return expanded_df

        required_cols = [
            "temperature",
            "adsorbate_molecular_weight",
            "encoded_adsorbent",
            "pressure",
        ]
        if has_output:
            required_cols.append(self.output)
        available_cols = [col for col in required_cols if col in expanded_df.columns]
        expanded_df = expanded_df.dropna(subset=available_cols).reset_index(drop=True)

        return expanded_df

    # -------------------------------------------------------------------------
    def separate_inputs_and_output(
        self, data: pd.DataFrame
    ) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
        expanded = self.expand_to_single_measurements(data)
        if expanded.empty:
            smile_shape = (0, self.smile_length)
            inputs = {
                "adsorbate_input": np.empty(smile_shape, dtype=np.float32),
                "features_input": np.empty((0, 4), dtype=np.float32),
            }
            return inputs, np.empty((0, 1), dtype=np.float32)

        adsorbate = np.vstack(
            [
                np.asarray(x, dtype=np.float32)
                for x in expanded["adsorbate_encoded_SMILE"].to_list()
            ]
        )
        temperature = expanded["temperature"].to_numpy(dtype=np.float32)
        molecular_weight = expanded["adsorbate_molecular_weight"].to_numpy(
            dtype=np.float32
        )
        adsorbent = expanded["encoded_adsorbent"].to_numpy(dtype=np.float32)
        pressure = expanded["pressure"].to_numpy(dtype=np.float32)
        features = np.column_stack((temperature, molecular_weight, adsorbent, pressure))
        inputs_dict = {"adsorbate_input": adsorbate, "features_input": features}

        output = None
        if self.output in expanded.columns:
            output = expanded[self.output].to_numpy(dtype=np.float32).reshape(-1, 1)

        return inputs_dict, output

    # -------------------------------------------------------------------------
    def build_training_dataloader(
        self,
        data: pd.DataFrame,
        batch_size: int | None = None,
        buffer_size: int = tf.data.AUTOTUNE,
    ) -> Any:
        batch_size = self.batch_size if batch_size is None else batch_size
        inputs, output = self.separate_inputs_and_output(data)
        dataset = tf.data.Dataset.from_tensor_slices((inputs, output))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)
        dataset = (
            dataset.shuffle(buffer_size=self.shuffle_samples)
            if self.shuffle
            else dataset
        )

        return dataset

    # -------------------------------------------------------------------------
    def process_inference_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        processed_data = self.processor.remove_invalid_measurements(data)
        processed_data = self.processor.aggregate_inference_data(processed_data)
        processed_data = self.processor.add_properties_to_inference_inputs(
            processed_data
        )
        processed_data = self.processor.encode_from_references(processed_data)
        processed_data = self.processor.normalize_from_references(processed_data)
        processed_data = self.processor.apply_padding(processed_data)
        processed_data = self.expand_to_single_measurements(processed_data)

        return processed_data

    # -------------------------------------------------------------------------
    def build_inference_dataloader(
        self,
        data: pd.DataFrame,
        batch_size: int | None = None,
        buffer_size: int = tf.data.AUTOTUNE,
    ) -> Any:
        batch_size = self.inference_batch_size if batch_size is None else batch_size
        processed_data = self.process_inference_inputs(data)
        inputs, output = self.separate_inputs_and_output(processed_data)
        dataset = tf.data.Dataset.from_tensor_slices((inputs, output))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)

        return dataset
