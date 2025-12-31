from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pandas as pd

from ADSMOD.server.database.database import database
from ADSMOD.server.utils.logger import logger
from ADSMOD.server.utils.services.sanitizer import (
    AdsorbentEncoder,
    AggregateDatasets,
    DataSanitizer,
    FeatureNormalizer,
    TrainValidationSplit,
)
from ADSMOD.server.utils.services.sequences import (
    PressureUptakeSeriesProcess,
    SMILETokenization,
)


###############################################################################
class DatasetBuilderConfig:
    def __init__(
        self,
        sample_size: float = 1.0,
        validation_size: float = 0.2,
        min_measurements: int = 1,
        max_measurements: int = 30,
        smile_sequence_size: int = 20,
        max_pressure: float = 10000.0,
        max_uptake: float = 20.0,
        seed: int = 42,
        split_seed: int = 76,
    ) -> None:
        self.sample_size = sample_size
        self.validation_size = validation_size
        self.min_measurements = min_measurements
        self.max_measurements = max_measurements
        self.smile_sequence_size = smile_sequence_size
        self.max_pressure = max_pressure
        self.max_uptake = max_uptake
        self.seed = seed
        self.split_seed = split_seed

    # -------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "validation_size": self.validation_size,
            "min_measurements": self.min_measurements,
            "max_measurements": self.max_measurements,
            "smile_sequence_size": self.smile_sequence_size,
            "max_pressure": self.max_pressure,
            "max_uptake": self.max_uptake,
            "seed": self.seed,
            "split_seed": self.split_seed,
        }


###############################################################################
class DatasetBuilder:
    def __init__(self, config: DatasetBuilderConfig) -> None:
        self.config = config
        self.configuration = config.to_dict()

    # -------------------------------------------------------------------------
    def build_training_dataset(
        self,
        adsorption_data: pd.DataFrame,
        guest_data: pd.DataFrame | None = None,
        host_data: pd.DataFrame | None = None,
        dataset_name: str = "default",
    ) -> dict[str, Any]:
        if adsorption_data.empty:
            logger.warning("No adsorption data provided for building dataset")
            return {"success": False, "error": "No adsorption data provided"}

        logger.info(f"{len(adsorption_data)} measurements in the dataset")

        aggregator = AggregateDatasets(self.configuration)
        processed_data = aggregator.aggregate_adsorption_measurements(adsorption_data)

        sample_size = self.config.sample_size
        if sample_size < 1.0:
            processed_data = processed_data.sample(
                frac=sample_size, random_state=self.config.seed
            ).reset_index(drop=True)

        logger.info(f"Aggregated dataset has {len(processed_data)} experiments")

        if guest_data is not None and host_data is not None:
            processed_data = aggregator.join_materials_properties(
                processed_data, guest_data, host_data
            )

        sanitizer = DataSanitizer(self.configuration)
        logger.info("Filtering Out-of-Boundary values")
        processed_data = sanitizer.exclude_OOB_values(processed_data)

        sequencer = PressureUptakeSeriesProcess(self.configuration)
        logger.info("Performing sequence sanitization and filter by size")
        processed_data = sequencer.remove_leading_zeros(processed_data)
        processed_data = sequencer.filter_by_sequence_size(processed_data)

        if processed_data.empty:
            logger.warning("No data remaining after filtering")
            return {"success": False, "error": "No data remaining after filtering"}

        smile_vocab = {}
        if "adsorbate_SMILE" in processed_data.columns:
            tokenization = SMILETokenization(self.configuration)
            logger.info("Tokenizing SMILE sequences for adsorbate species")
            processed_data, smile_vocab = tokenization.process_SMILE_sequences(
                processed_data
            )

        logger.info("Generate train and validation datasets through stratified splitting")
        splitter = TrainValidationSplit(self.configuration)
        training_data = splitter.split_train_and_validation(processed_data)
        train_samples = training_data[training_data["split"] == "train"]
        validation_samples = training_data[training_data["split"] == "validation"]

        normalizer = FeatureNormalizer(self.configuration, train_samples)
        training_data = normalizer.normalize_molecular_features(training_data)
        training_data = normalizer.PQ_series_normalization(training_data)

        training_data = sequencer.PQ_series_padding(training_data)

        adsorbent_vocab = {}
        if "adsorbent_name" in training_data.columns:
            encoding = AdsorbentEncoder(self.configuration, train_samples)
            training_data = encoding.encode_adsorbents_by_name(training_data)
            adsorbent_vocab = encoding.mapping

        training_data["dataset_name"] = dataset_name

        training_data["pressure"] = training_data["pressure"].apply(json.dumps)
        training_data["adsorbed_amount"] = training_data["adsorbed_amount"].apply(json.dumps)
        if "adsorbate_encoded_SMILE" in training_data.columns:
            training_data["adsorbate_encoded_SMILE"] = training_data[
                "adsorbate_encoded_SMILE"
            ].apply(json.dumps)

        self.save_training_dataset(training_data)
        self.save_training_metadata(
            train_samples=len(train_samples),
            validation_samples=len(validation_samples),
            smile_vocab=smile_vocab,
            adsorbent_vocab=adsorbent_vocab,
            statistics=normalizer.statistics,
        )

        logger.info(f"Saved train dataset with {len(train_samples)} records")
        logger.info(f"Saved validation dataset with {len(validation_samples)} records")

        return {
            "success": True,
            "total_samples": len(training_data),
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
        }

    # -------------------------------------------------------------------------
    def save_training_dataset(self, training_data: pd.DataFrame) -> None:
        columns_to_save = [
            "dataset_name",
            "split",
            "temperature",
            "pressure",
            "adsorbed_amount",
        ]

        if "encoded_adsorbent" in training_data.columns:
            columns_to_save.append("encoded_adsorbent")
        if "adsorbate_molecular_weight" in training_data.columns:
            columns_to_save.append("adsorbate_molecular_weight")
        if "adsorbate_encoded_SMILE" in training_data.columns:
            columns_to_save.append("adsorbate_encoded_SMILE")

        available_columns = [c for c in columns_to_save if c in training_data.columns]
        data_to_save = training_data[available_columns].copy()

        database.save_into_database(data_to_save, "TRAINING_DATASET")

    # -------------------------------------------------------------------------
    def save_training_metadata(
        self,
        train_samples: int,
        validation_samples: int,
        smile_vocab: dict,
        adsorbent_vocab: dict,
        statistics: dict | None,
    ) -> None:
        metadata = pd.DataFrame([{
            "created_at": datetime.now().isoformat(),
            "sample_size": self.config.sample_size,
            "validation_size": self.config.validation_size,
            "min_measurements": self.config.min_measurements,
            "max_measurements": self.config.max_measurements,
            "smile_sequence_size": self.config.smile_sequence_size,
            "max_pressure": self.config.max_pressure,
            "max_uptake": self.config.max_uptake,
            "total_samples": train_samples + validation_samples,
            "train_samples": train_samples,
            "validation_samples": validation_samples,
            "smile_vocabulary": json.dumps(smile_vocab),
            "adsorbent_vocabulary": json.dumps(adsorbent_vocab),
            "normalization_stats": json.dumps(statistics) if statistics else "{}",
        }])

        database.save_into_database(metadata, "TRAINING_METADATA")

    # -------------------------------------------------------------------------
    @staticmethod
    def get_training_dataset_info() -> dict[str, Any] | None:
        try:
            metadata = database.load_from_database("TRAINING_METADATA")
            if metadata.empty:
                return None

            row = metadata.iloc[0]
            return {
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
            }
        except Exception as e:
            logger.warning(f"Failed to load training dataset info: {e}")
            return None

    # -------------------------------------------------------------------------
    @staticmethod
    def clear_training_dataset() -> bool:
        try:
            empty_df = pd.DataFrame()
            database.save_into_database(empty_df, "TRAINING_DATASET")
            database.save_into_database(empty_df, "TRAINING_METADATA")
            logger.info("Training dataset cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear training dataset: {e}")
            return False
