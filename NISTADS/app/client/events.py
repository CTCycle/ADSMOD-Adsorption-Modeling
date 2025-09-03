from __future__ import annotations

from typing import Any

import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pandas.core.frame import DataFrame
from PySide6.QtGui import QImage, QPixmap

from NISTADS.app.client.workers import check_thread_status, update_progress_callback
from NISTADS.app.logger import logger
from NISTADS.app.utils.api.server import AdsorptionDataFetch, GuestHostDataFetch
from NISTADS.app.utils.data.builder import BuildAdsorptionDataset
from NISTADS.app.utils.data.loader import SCADSDataLoader
from NISTADS.app.utils.data.properties import MolecularProperties
from NISTADS.app.utils.data.serializer import DataSerializer, ModelSerializer
from NISTADS.app.utils.learning.device import DeviceConfig
from NISTADS.app.utils.learning.inference.predictor import AdsorptionPredictions
from NISTADS.app.utils.learning.models.qmodel import SCADSModel
from NISTADS.app.utils.learning.training.fitting import ModelTraining
from NISTADS.app.utils.processing.conversion import PQ_units_conversion
from NISTADS.app.utils.processing.sanitizer import (
    AdsorbentEncoder,
    AggregateDatasets,
    DataSanitizer,
    FeatureNormalizer,
    TrainValidationSplit,
)
from NISTADS.app.utils.processing.sequences import (
    PressureUptakeSeriesProcess,
    SMILETokenization,
)
from NISTADS.app.utils.validation.checkpoints import ModelEvaluationSummary
from NISTADS.app.utils.validation.dataset import AdsorptionPredictionsQuality


###############################################################################
class GraphicsHandler:
    def __init__(self) -> None:
        self.image_encoding = cv2.IMREAD_UNCHANGED
        self.gray_scale_encoding = cv2.IMREAD_GRAYSCALE
        self.BGRA_encoding = cv2.COLOR_BGRA2RGBA
        self.BGR_encoding = cv2.COLOR_BGR2RGB

    # -------------------------------------------------------------------------
    def convert_fig_to_qpixmap(self, fig) -> QPixmap:
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        # get the size in pixels and initialize raw RGBA buffer
        width, height = canvas.get_width_height()
        buf = canvas.buffer_rgba()
        # construct a QImage pointing at that memory (no PNG decoding)
        qimg = QImage(buf, width, height, QImage.Format.Format_RGBA8888)

        return QPixmap.fromImage(qimg)

    # -------------------------------------------------------------------------
    def load_image_as_pixmap(self, path: str) -> None | QPixmap:
        img = cv2.imread(path, self.image_encoding)
        # Handle grayscale, RGB, or RGBA
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, self.gray_scale_encoding)
        elif img.shape[2] == 4:  # BGRA
            img = cv2.cvtColor(img, self.BGRA_encoding)
        else:  # BGR
            img = cv2.cvtColor(img, self.BGR_encoding)

        h, w = img.shape[:2]
        if img.shape[2] == 3:
            qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        else:
            qimg = QImage(img.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)

        return QPixmap.fromImage(qimg)


###############################################################################
class DatasetEvents:
    def __init__(self, configuration: dict[str, Any]) -> None:
        self.serializer = DataSerializer()
        self.modser = ModelSerializer()
        self.seed = configuration.get("seed", 42)
        self.configuration = configuration

    # -------------------------------------------------------------------------
    def run_data_collection_pipeline(
        self, progress_callback: Any | None = None, worker=None
    ) -> None:
        # 1. get isotherm indexes invoking API from NIST-ARPA-E database
        logger.info("Collect adsorption isotherm indices from NIST-ARPA-E database")
        API = AdsorptionDataFetch(self.configuration)
        experiments_index = API.get_experiments_index()
        # 2. collect experiments data based on the fetched index
        logger.info("Extracting adsorption isotherms data from JSON response")
        adsorption_data = API.get_experiments_data(
            experiments_index, worker=worker, progress_callback=progress_callback
        )

        if adsorption_data is None:
            logger.warning("Adsorption data was not collected, dataset is empty")
            return

        # remove excluded columns from the dataframe
        builder = BuildAdsorptionDataset()
        logger.info("Cleaning and processing adsorption dataset")
        adsorption_data = builder.drop_excluded_columns(adsorption_data)
        # split current dataframe by complexity of the mixture (single component or binary mixture)
        logger.info("Experiments will be split based on mixture complexity")
        single_component, binary_mixture = builder.split_by_mixture_complexity(
            adsorption_data
        )
        # extract nested data in dataframe rows and reorganise them into columns
        single_component = builder.extract_nested_data(single_component)
        binary_mixture = builder.extract_nested_data(binary_mixture)

        # expand the dataset to represent each measurement with a single row
        # save the final version of the adsorption dataset
        single_component, binary_mixture = builder.expand_dataset(
            single_component, binary_mixture
        )
        self.serializer.save_adsorption_datasets(single_component, binary_mixture)
        logger.info("Experiments data collection is concluded")

        # get guest and host indexes invoking API
        logger.info("Collect guest and host indices from NIST-ARPA-E database")
        API = GuestHostDataFetch(self.configuration)
        guest_index, host_index = API.get_materials_index()
        # extract adsorbents and sorbates data from relative indices
        logger.info("Extracting adsorbents and sorbates data from relative indices")
        guest_data, host_data = API.get_materials_data(
            guest_index, host_index, worker=worker, progress_callback=progress_callback
        )

        # save the final version of the materials dataset
        self.serializer.save_materials_datasets(guest_data, host_data)
        logger.info("Materials data collection is concluded")

    # -------------------------------------------------------------------------
    def run_chemical_properties_pipeline(
        self, target: str = "guest", progress_callback: Any | None = None, worker=None
    ) -> None:
        experiments, guest_data, host_data = self.serializer.load_adsorption_datasets()
        properties = MolecularProperties(self.configuration)
        # process guest (adsorbed species) data by adding molecular properties
        if target == "guest":
            logger.info(
                "Retrieving molecular properties for sorbate species using PubChem API"
            )
            guest_data = properties.fetch_guest_properties(
                experiments,
                guest_data,
                worker=worker,
                progress_callback=progress_callback,
            )
            # save the final version of the materials dataset
            self.serializer.save_materials_datasets(guest_data=guest_data)
            records = guest_data.shape[0] if guest_data is not None else 0
            logger.info(f"Guest properties updated in the database ({records} records)")
        # process host (adsorbent materials) data by adding molecular properties
        elif target == "host":
            logger.info(
                "Retrieving molecular properties for adsorbent materials using PubChem API"
            )
            host_data = properties.fetch_host_properties(
                experiments,
                host_data,
                worker=worker,
                progress_callback=progress_callback,
            )
            # save the final version of the materials dataset
            self.serializer.save_materials_datasets(host_data=host_data)
            records = host_data.shape[0] if host_data is not None else 0
            logger.info(f"Host properties updated in the database ({records} records)")

    # -------------------------------------------------------------------------
    def run_dataset_builder(
        self, progress_callback: Any | None = None, worker=None
    ) -> None:
        adsorption_data, guest_data, host_data = (
            self.serializer.load_adsorption_datasets()
        )
        if adsorption_data.empty:
            logger.warning(
                "No adsorption data found in the database, please fetch data first"
            )
            return

        logger.info(f"{len(adsorption_data)} measurements in the dataset")
        logger.info(
            f"{len(guest_data)} total guests (adsorbates species) in the dataset"
        )
        logger.info(
            f"{len(host_data)} total hosts (adsorbent materials) in the dataset"
        )

        # check thread for interruption
        check_thread_status(worker)
        update_progress_callback(1, 8, progress_callback)

        # group single component data based on the experiment name
        # merge adsorption data with retrieved materials properties (guest and host)
        aggregator = AggregateDatasets(self.configuration)
        processed_data = aggregator.aggregate_adsorption_measurements(adsorption_data)
        # select only a fraction of adsorption isotherm experiments
        sample_size = self.configuration.get("sample_size", 1.0)
        if sample_size < 1.0:
            processed_data = processed_data.sample(
                frac=sample_size, random_state=self.seed
            ).reset_index(drop=True)
        logger.info(f"Aggregated dataset has {len(processed_data)} experiments")
        # start joining materials properties
        processed_data = aggregator.join_materials_properties(
            processed_data, guest_data, host_data
        )

        # check thread for interruption
        check_thread_status(worker)
        update_progress_callback(2, 8, progress_callback)

        # convert pressure and uptake into standard units:
        # pressure to Pascal, uptake to mol/g
        sequencer = PressureUptakeSeriesProcess(self.configuration)
        logger.info("Converting pressure into Pascal and uptake into mol/g")
        processed_data = PQ_units_conversion(processed_data)

        # check thread for interruption
        check_thread_status(worker)
        update_progress_callback(3, 8, progress_callback)

        # exlude all data outside given boundaries, such as negative temperature values
        # and pressure and uptake values below zero or above upper limits
        sanitizer = DataSanitizer(self.configuration)
        logger.info("Filtering Out-of-Boundary values")
        processed_data = sanitizer.exclude_OOB_values(processed_data)

        # remove repeated zero values at the beginning of pressure and uptake series
        # then filter out experiments with not enough measurements
        logger.info("Performing sequence sanitization and filter by size")
        processed_data = sequencer.remove_leading_zeros(processed_data)
        processed_data = sequencer.filter_by_sequence_size(processed_data)

        # check thread for interruption
        check_thread_status(worker)
        update_progress_callback(4, 8, progress_callback)

        # perform SMILE sequence tokenization
        tokenization = SMILETokenization(self.configuration)
        logger.info("Tokenizing SMILE sequences for adsorbate species")
        processed_data, smile_vocab = tokenization.process_SMILE_sequences(
            processed_data
        )

        # split data into train set and validation set
        logger.info(
            "Generate train and validation datasets through stratified splitting"
        )
        splitter = TrainValidationSplit(self.configuration)
        training_data = splitter.split_train_and_validation(processed_data)
        train_samples = training_data[training_data["split"] == "train"]
        validation_samples = training_data[training_data["split"] == "validation"]

        # check thread for interruption
        check_thread_status(worker)
        update_progress_callback(5, 8, progress_callback)

        # normalize pressure and uptake series using max values computed from
        # the training set, then pad sequences to a fixed length
        normalizer = FeatureNormalizer(self.configuration, train_samples)
        training_data = normalizer.normalize_molecular_features(training_data)
        training_data = normalizer.PQ_series_normalization(training_data)

        # check thread for interruption
        check_thread_status(worker)
        update_progress_callback(6, 8, progress_callback)
        # add padding to pressure and uptake series to match max length
        training_data = sequencer.PQ_series_padding(training_data)

        encoding = AdsorbentEncoder(self.configuration, train_samples)
        training_data = encoding.encode_adsorbents_by_name(training_data)
        adsorbent_vocab = encoding.mapping

        # check thread for interruption
        check_thread_status(worker)
        update_progress_callback(7, 8, progress_callback)

        # save preprocessed data using data serializer
        training_data = sanitizer.isolate_processed_features(training_data)
        self.serializer.save_training_data(
            training_data,
            self.configuration,
            smile_vocab,
            adsorbent_vocab,
            normalizer.statistics,
        )

        # check thread for interruption
        update_progress_callback(8, 8, progress_callback)

        logger.info(f"Saved train dataset with {len(train_samples)} records saved")
        logger.info(f"Saved validation dataset with {len(validation_samples)} records")

    # -------------------------------------------------------------------------
    @staticmethod
    def rebuild_dataset_from_metadata(metadata: dict) -> tuple[DataFrame, DataFrame]:
        serializer = DataSerializer()
        adsorption_data, guest_data, host_data = serializer.load_adsorption_datasets()
        logger.info(f"{len(adsorption_data)} measurements in the dataset")
        logger.info(
            f"{len(guest_data)} total guests (adsorbates species) in the dataset"
        )
        logger.info(
            f"{len(host_data)} total hosts (adsorbent materials) in the dataset"
        )

        # group single component data based on the experiment name
        # merge adsorption data with retrieved materials properties (guest and host)
        aggregator = AggregateDatasets(metadata)
        processed_data = aggregator.aggregate_adsorption_measurements(adsorption_data)
        # select only a fraction of adsorption isotherm experiments
        sample_size = metadata.get("sample_size", 1.0)
        seed = metadata.get("seed", 42)
        processed_data = processed_data.sample(
            frac=sample_size, random_state=seed
        ).reset_index(drop=True)
        logger.info(f"Aggregated dataset has {len(processed_data)} experiments")

        # start joining materials properties
        processed_data = aggregator.join_materials_properties(
            processed_data, guest_data, host_data
        )

        # convert pressure and uptake into standard units:
        # pressure to Pascal, uptake to mol/g
        sequencer = PressureUptakeSeriesProcess(metadata)
        logger.info("Converting pressure into Pascal and uptake into mol/g")
        processed_data = PQ_units_conversion(processed_data)

        # exlude all data outside given boundaries, such as negative temperature values
        # and pressure and uptake values below zero or above upper limits
        sanitizer = DataSanitizer(metadata)
        logger.info("Filtering Out-of-Boundary values")
        processed_data = sanitizer.exclude_OOB_values(processed_data)

        # remove repeated zero values at the beginning of pressure and uptake series
        # then filter out experiments with not enough measurements
        logger.info("Performing sequence sanitization and filter by size")
        processed_data = sequencer.remove_leading_zeros(processed_data)
        processed_data = sequencer.filter_by_sequence_size(processed_data)

        # perform SMILE sequence tokenization
        tokenization = SMILETokenization(metadata)
        logger.info("Tokenizing SMILE sequences for adsorbate species")
        processed_data, smile_vocab = tokenization.process_SMILE_sequences(
            processed_data
        )

        # split data into train set and validation set
        logger.info(
            "Generate train and validation datasets through stratified splitting"
        )
        splitter = TrainValidationSplit(metadata)
        training_data = splitter.split_train_and_validation(processed_data)
        train_samples = training_data[training_data["split"] == "train"]
        validation_samples = training_data[training_data["split"] == "validation"]

        # normalize pressure and uptake series using max values computed from
        # the training set, then pad sequences to a fixed length
        normalizer = FeatureNormalizer(metadata, train_samples)
        training_data = normalizer.normalize_molecular_features(training_data)
        training_data = normalizer.PQ_series_normalization(training_data)
        # add padding to pressure and uptake series to match max length
        training_data = sequencer.PQ_series_padding(training_data)

        encoding = AdsorbentEncoder(metadata, train_samples)
        training_data = encoding.encode_adsorbents_by_name(training_data)

        # save preprocessed data using data serializer
        training_data = sanitizer.isolate_processed_features(training_data)

        return train_samples, validation_samples


###############################################################################
class ValidationEvents:
    def __init__(self, configuration: dict[str, Any]):
        self.serializer = DataSerializer()
        self.modser = ModelSerializer()
        self.configuration = configuration

    # -------------------------------------------------------------------------
    def run_dataset_evaluation_pipeline(
        self, metrics, progress_callback: Any | None = None, worker=None
    ):
        adsorption_data, guest_data, host_data = (
            self.serializer.load_adsorption_datasets()
        )
        logger.info(f"{adsorption_data.shape[0]} measurements in the dataset")
        logger.info(f"{guest_data.shape[0]} adsorbates species in the dataset")
        logger.info(f"{host_data.shape[0]} adsorbent materials in the dataset")

        # check thread for interruption
        check_thread_status(worker)

        metric_map = {
            "experiments_clustering": lambda *args, **kwargs: None,
            "experiments_clustering_2": lambda *args, **kwargs: None,
        }

        images = []
        for metric in metrics:
            if metric in metric_map:
                # check worker status to allow interruption
                check_thread_status(worker)
                metric_name = metric.replace("_", " ").title()
                logger.info(f"Current metric: {metric_name}")
                result = metric_map[metric](
                    adsorption_data, progress_callback=progress_callback, worker=worker
                )
                images.append(result)

    # -------------------------------------------------------------------------
    def get_checkpoints_summary(
        self, progress_callback: Any | None = None, worker=None
    ):
        summarizer = ModelEvaluationSummary(self.configuration)
        checkpoints_summary = summarizer.get_checkpoints_summary(
            progress_callback=progress_callback, worker=worker
        )

        logger.info(
            f"Checkpoints summary has been created for {checkpoints_summary.shape[0]} models"
        )

    # -------------------------------------------------------------------------
    def run_model_evaluation_pipeline(
        self,
        metrics,
        selected_checkpoint: str,
        progress_callback: Any | None = None,
        worker=None,
    ):
        if selected_checkpoint is None:
            logger.warning("No checkpoint selected for resuming training")
            return

        logger.info(f"Loading {selected_checkpoint} checkpoint")
        model, train_config, model_metadata, _, checkpoint_path = (
            self.modser.load_checkpoint(selected_checkpoint)
        )
        model.summary(expand_nested=True)
        # set device for training operations
        logger.info("Setting device for training operations")
        device = DeviceConfig(self.configuration)
        device.set_device()
        # load validation data and current preprocessing metadata. This must
        # be compatible with the currently loaded checkpoint configurations
        current_metadata = self.serializer.load_training_data(only_metadata=True)
        is_validated = self.serializer.validate_metadata(
            current_metadata, model_metadata
        )
        # just load the data if metadata is compatible
        if is_validated:
            logger.info(
                "Loading processed dataset as it is compatible with the selected checkpoint"
            )
            _, validation_data, model_metadata = self.serializer.load_training_data()
        else:
            logger.info(f"Rebuilding dataset from {selected_checkpoint} metadata")
            _, validation_data = DatasetEvents.rebuild_dataset_from_metadata(
                model_metadata
            )

        loader = SCADSDataLoader(train_config, model_metadata)
        validation_dataset = loader.build_training_dataloader(validation_data)

        summarizer = ModelEvaluationSummary(self.configuration, model)
        validator = AdsorptionPredictionsQuality(
            model, train_config, model_metadata, checkpoint_path
        )

        # Mapping metric name to method and arguments
        metric_map = {
            "evaluation_report": summarizer.get_evaluation_report,
            "prediction_quality": validator.visualize_adsorption_isotherms,
        }

        images = []
        for metric in metrics:
            if metric in metric_map:
                # check worker status to allow interruption
                check_thread_status(worker)
                metric_name = metric.replace("_", " ").title()
                logger.info(f"Current metric: {metric_name}")
                result = metric_map[metric](
                    validation_dataset,
                    progress_callback=progress_callback,
                    worker=worker,
                )
                images.append(result)

        return images


###############################################################################
class ModelEvents:
    def __init__(self, configuration: dict[str, Any]) -> None:
        self.serializer = DataSerializer()
        self.modser = ModelSerializer()
        self.configuration = configuration

    # -------------------------------------------------------------------------
    def get_available_checkpoints(self) -> list[str]:
        return self.modser.scan_checkpoints_folder()

    # -------------------------------------------------------------------------
    def run_training_pipeline(
        self, progress_callback: Any | None = None, worker=None
    ) -> None:
        train_data, validation_data, metadata = self.serializer.load_training_data()
        if train_data.empty or validation_data.empty:
            logger.warning("No data found in the database for training")
            return

        logger.info(
            "Building model data loaders with prefetching and parallel processing"
        )
        builder = SCADSDataLoader(self.configuration, metadata)
        train_dataset = builder.build_training_dataloader(train_data)
        validation_dataset = builder.build_training_dataloader(validation_data)

        # check thread for interruption
        check_thread_status(worker)

        # set device for training operations
        logger.info("Setting device for training operations")
        device = DeviceConfig(self.configuration)
        device.set_device()
        # create checkpoint folder
        checkpoint_path = self.modser.create_checkpoint_folder()
        # Setting callbacks and training routine for the machine learning model
        logger.info("Building SCADS model")
        wrapper = SCADSModel(self.configuration, metadata)
        model = wrapper.get_model(model_summary=True)
        # generate graphviz plot fo the model layout
        self.modser.save_model_plot(model, checkpoint_path)
        # perform training and save model at the end
        logger.info("Starting SCADS model training")
        trainer = ModelTraining(self.configuration, metadata)
        model, history = trainer.train_model(
            model,
            train_dataset,
            validation_dataset,
            checkpoint_path,
            progress_callback=progress_callback,
            worker=worker,
        )

        self.modser.save_pretrained_model(model, checkpoint_path)
        self.modser.save_training_configuration(
            checkpoint_path, history, self.configuration, metadata
        )

    # -------------------------------------------------------------------------
    def resume_training_pipeline(
        self,
        selected_checkpoint: str,
        progress_callback: Any | None = None,
        worker=None,
    ) -> None:
        logger.info(f"Loading {selected_checkpoint} checkpoint")
        model, train_config, model_metadata, session, checkpoint_path = (
            self.modser.load_checkpoint(selected_checkpoint)
        )
        model.summary(expand_nested=True)
        # set device for training operations
        logger.info("Setting device for training operations")
        device = DeviceConfig(self.configuration)
        device.set_device()

        # check thread for interruption
        check_thread_status(worker)

        # load metadata and check whether this is compatible with the current checkpoint
        # rebuild dataset if metadata is not compatible and the user has requested this feature
        current_metadata = self.serializer.load_training_data(only_metadata=True)
        is_validated = self.serializer.validate_metadata(
            current_metadata, model_metadata
        )
        # just load the data if metadata is compatible
        if is_validated:
            logger.info(
                "Loading processed dataset as it is compatible with the selected checkpoint"
            )
            train_data, validation_data, model_metadata = (
                self.serializer.load_training_data()
            )
        else:
            logger.info(f"Rebuilding dataset from {selected_checkpoint} metadata")
            train_data, validation_data = DatasetEvents.rebuild_dataset_from_metadata(
                model_metadata
            )

        # create the tf.datasets using the previously initialized generators
        logger.info("Loading preprocessed data and building dataloaders")
        builder = SCADSDataLoader(train_config, model_metadata)
        train_dataset = builder.build_training_dataloader(train_data)
        validation_dataset = builder.build_training_dataloader(validation_data)

        # check worker status to allow interruption
        check_thread_status(worker)

        # Setting callbacks and training routine for the machine learning model
        # resume training from pretrained model
        logger.info(f"Resuming training from checkpoint {selected_checkpoint}")
        additional_epochs = self.configuration.get("additional_epochs", 10)
        trainer = ModelTraining(train_config, model_metadata)
        model, history = trainer.resume_training(
            model,
            train_dataset,
            validation_dataset,
            checkpoint_path,
            session,
            additional_epochs,
            progress_callback=progress_callback,
            worker=worker,
        )

        self.modser.save_pretrained_model(model, checkpoint_path)
        self.modser.save_training_configuration(
            checkpoint_path, history, self.configuration, model_metadata
        )

    # -------------------------------------------------------------------------
    def run_inference_pipeline(
        self,
        selected_checkpoint: str,
        progress_callback: Any | None = None,
        worker=None,
    ) -> None:
        if selected_checkpoint is None:
            logger.warning("No checkpoint selected for resuming training")
            return

        logger.info(f"Loading {selected_checkpoint} checkpoint")
        model, train_config, metadata, _, checkpoint_path = self.modser.load_checkpoint(
            selected_checkpoint
        )
        model.summary(expand_nested=True)

        # setting device for training
        device = DeviceConfig(self.configuration)
        device.set_device()

        # check worker status to allow interruption
        check_thread_status(worker)

        # select images from the inference folder and retrieve current paths
        inference_data = self.serializer.load_inference_dataset()

        # initialize the adsorption prediction framework. This takes raw input and process
        # them based on loaded checkpoint metadata (including vocaularies)
        # output is processed as well and merged with raw input data table
        logger.info(
            "Preprocessing inference input data according to model configuration"
        )
        predictor = AdsorptionPredictions(
            model, train_config, metadata, checkpoint_path
        )
        predictions = predictor.predict_adsorption_isotherm(
            inference_data, progress_callback=progress_callback, worker=worker
        )

        predictions_dataset = predictor.build_predictions_dataset(
            inference_data, predictions
        )
        self.serializer.save_predictions_dataset(predictions_dataset)
        logger.info("Predictions dataset saved successfully in database")
