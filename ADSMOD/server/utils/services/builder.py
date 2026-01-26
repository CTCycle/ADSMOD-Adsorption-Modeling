from __future__ import annotations

import json
import hashlib
from datetime import datetime
from typing import Any

import pandas as pd

from ADSMOD.server.utils.logger import logger
from ADSMOD.server.schemas.training import TrainingMetadata
from ADSMOD.server.utils.repository.serializer import TrainingDataSerializer
from ADSMOD.server.utils.services.conversion import PQ_units_conversion
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
    def __init__(
        self, config: DatasetBuilderConfig, dataset_label: str = "default"
    ) -> None:
        self.config = config
        self.configuration = config.to_dict()
        self.serializer = TrainingDataSerializer()
        self.dataset_label = dataset_label

    # -------------------------------------------------------------------------
    def build_training_dataset(
        self,
        adsorption_data: pd.DataFrame,
        guest_data: pd.DataFrame | None = None,
        host_data: pd.DataFrame | None = None,
        dataset_name: str = "default",
        reference_metadata: TrainingMetadata | None = None,
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

        logger.info("Converting pressure into Pascal and uptake into mmol/g")
        processed_data = PQ_units_conversion(processed_data)

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

        if "adsorbate_SMILE" not in processed_data.columns:
            logger.warning("Training data missing adsorbate_SMILE column")
            return {
                "success": False,
                "error": "Training data missing adsorbate_SMILE values.",
            }

        if reference_metadata is not None:
            mismatches = []
            for field_name in [
                "sample_size",
                "validation_size",
                "min_measurements",
                "max_measurements",
                "smile_sequence_size",
                "max_pressure",
                "max_uptake",
            ]:
                current_value = getattr(self.config, field_name)
                reference_value = getattr(reference_metadata, field_name)
                if current_value != reference_value:
                    mismatches.append(
                        f"{field_name}={current_value} (expected {reference_value})"
                    )
            if mismatches:
                logger.warning(
                    "Dataset build config does not match reference metadata: %s",
                    ", ".join(mismatches),
                )
                return {
                    "success": False,
                    "error": "Dataset build config does not match reference metadata.",
                }

        smile_vocab = {}
        reference_smile_vocab = None
        if reference_metadata is not None:
            reference_smile_vocab = (
                reference_metadata.smile_vocabulary
                or reference_metadata.SMILE_vocabulary
                or {}
            )
        if "adsorbate_SMILE" in processed_data.columns:
            tokenization = SMILETokenization(self.configuration)
            logger.info("Tokenizing SMILE sequences for adsorbate species")
            processed_data, smile_vocab = tokenization.process_SMILE_sequences(
                processed_data, reference_vocabulary=reference_smile_vocab
            )
            if processed_data.empty:
                logger.warning("No data remaining after SMILE tokenization")
                return {
                    "success": False,
                    "error": "No valid SMILE sequences found in the dataset.",
                }
            if not smile_vocab:
                logger.warning("SMILE vocabulary is empty after tokenization")
                return {
                    "success": False,
                    "error": "SMILE vocabulary is empty. Check adsorbate_SMILE values.",
                }
            if reference_metadata is not None and reference_smile_vocab:
                smile_vocab = reference_smile_vocab

        logger.info(
            "Generate train and validation datasets through stratified splitting"
        )
        splitter = TrainValidationSplit(self.configuration)
        training_data = splitter.split_train_and_validation(processed_data)
        train_samples = training_data[training_data["split"] == "train"]
        validation_samples = training_data[training_data["split"] == "validation"]

        reference_stats = None
        if reference_metadata is not None and reference_metadata.normalization_stats:
            reference_stats = reference_metadata.normalization_stats
        normalizer = FeatureNormalizer(
            self.configuration, train_samples, statistics=reference_stats
        )
        training_data = normalizer.normalize_molecular_features(training_data)
        training_data = normalizer.PQ_series_normalization(training_data)

        training_data = sequencer.PQ_series_padding(training_data)

        adsorbent_vocab = {}
        if "adsorbent_name" in training_data.columns:
            if (
                reference_metadata is not None
                and reference_metadata.adsorbent_vocabulary
            ):
                encoding = AdsorbentEncoder(self.configuration, train_samples)
                training_data, _ = encoding.encode_adsorbents_from_vocabulary(
                    training_data, reference_metadata.adsorbent_vocabulary
                )
                adsorbent_vocab = reference_metadata.adsorbent_vocabulary
            else:
                encoding = AdsorbentEncoder(self.configuration, train_samples)
                training_data = encoding.encode_adsorbents_by_name(training_data)
                adsorbent_vocab = encoding.mapping

        training_data["dataset_name"] = dataset_name
        training_data["dataset_label"] = self.dataset_label

        training_data["pressure"] = training_data["pressure"].apply(json.dumps)
        training_data["adsorbed_amount"] = training_data["adsorbed_amount"].apply(
            json.dumps
        )
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
            "dataset_label",
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

        self.serializer.save_training_dataset(data_to_save, self.dataset_label)

    # -------------------------------------------------------------------------
    def save_training_metadata(
        self,
        train_samples: int,
        validation_samples: int,
        smile_vocab: dict,
        adsorbent_vocab: dict,
        statistics: dict | None,
    ) -> None:
        # Construct the metadata object first
        metadata = TrainingMetadata(
            created_at=datetime.now().isoformat(),
            sample_size=self.config.sample_size,
            validation_size=self.config.validation_size,
            min_measurements=self.config.min_measurements,
            max_measurements=self.config.max_measurements,
            smile_sequence_size=self.config.smile_sequence_size,
            max_pressure=self.config.max_pressure,
            max_uptake=self.config.max_uptake,
            total_samples=train_samples + validation_samples,
            train_samples=train_samples,
            validation_samples=validation_samples,
            smile_vocabulary=smile_vocab,
            adsorbent_vocabulary=adsorbent_vocab,
            normalization_stats=statistics or {},
            normalization=statistics or {},
        )

        # Compute hash using the centralized logic in serializer
        dataset_hash = TrainingDataSerializer.compute_metadata_hash(metadata)
        metadata.dataset_hash = dataset_hash

        # Convert to DataFrame for compatibility with existing save method if needed
        # Or preferably use the serializer's method directly if it accepts the object
        # Looking at serializer.py logic, save_training_metadata(metadata: pd.DataFrame)
        # So we convert to DataFrame as before, but with the correct hash.

        metadata_df = pd.DataFrame(
            [
                {
                    "dataset_label": self.dataset_label,
                    "created_at": metadata.created_at,
                    "dataset_hash": metadata.dataset_hash,
                    "sample_size": metadata.sample_size,
                    "validation_size": metadata.validation_size,
                    "min_measurements": metadata.min_measurements,
                    "max_measurements": metadata.max_measurements,
                    "smile_sequence_size": metadata.smile_sequence_size,
                    "max_pressure": metadata.max_pressure,
                    "max_uptake": metadata.max_uptake,
                    "total_samples": metadata.total_samples,
                    "train_samples": metadata.train_samples,
                    "validation_samples": metadata.validation_samples,
                    "smile_vocabulary": json.dumps(metadata.smile_vocabulary),
                    "adsorbent_vocabulary": json.dumps(metadata.adsorbent_vocabulary),
                    "normalization_stats": json.dumps(metadata.normalization_stats),
                }
            ]
        )

        self.serializer.save_training_metadata(metadata_df, self.dataset_label)

    # -------------------------------------------------------------------------
    @staticmethod
    def get_training_dataset_info(
        dataset_label: str = "default",
    ) -> dict[str, Any] | None:
        try:
            serializer = TrainingDataSerializer()
            # load_training_metadata returns a TrainingMetadata object
            metadata = serializer.load_training_metadata(dataset_label)

            # Check if empty (default object usually has total_samples=0)
            if not metadata or metadata.total_samples == 0:
                return None

            return {
                "dataset_label": dataset_label,
                "created_at": metadata.created_at,
                "sample_size": metadata.sample_size,
                "validation_size": metadata.validation_size,
                "min_measurements": metadata.min_measurements,
                "max_measurements": metadata.max_measurements,
                "smile_sequence_size": metadata.smile_sequence_size,
                "max_pressure": metadata.max_pressure,
                "max_uptake": metadata.max_uptake,
                "total_samples": metadata.total_samples,
                "train_samples": metadata.train_samples,
                "validation_samples": metadata.validation_samples,
            }
        except Exception as e:
            logger.warning(f"Failed to load training dataset info: {e}")
            return None

    # -------------------------------------------------------------------------
    @staticmethod
    def clear_training_dataset(dataset_label: str | None = None) -> bool:
        try:
            serializer = TrainingDataSerializer()
            serializer.clear_training_dataset(dataset_label)
            if dataset_label:
                logger.info(f"Training dataset '{dataset_label}' cleared")
            else:
                logger.info("All training datasets cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear training dataset: {e}")
            return False

    # -------------------------------------------------------------------------
    @staticmethod
    def list_processed_datasets() -> list[dict[str, Any]]:
        """Returns a list of all processed datasets with their metadata."""
        try:
            return TrainingDataSerializer.list_processed_datasets()
        except Exception as e:
            logger.error(f"Failed to list processed datasets: {e}")
            return []
