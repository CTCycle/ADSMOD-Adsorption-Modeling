# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any

from pathlib import Path

from server.configurations.settings import AdsmodConfig
from server.domain.training_data import TrainingDataAccess
from server.models.callbacks import WorkerInterrupted
from server.models.device import DeviceConfig
from server.models.training.loader import (
    SCADSAtomicDataLoader,
    SCADSDataLoader,
)
from server.models.training.qmodel import SCADSAtomicModel, SCADSModel
from server.models.training.serialization.model import ModelSerializer
from server.models.training.serialization.training import TrainingDataSerializer
from server.models.training.fitting import ModelTraining
from server.common.utils.logger import logger
from server.common.constants import SCADS_ATOMIC_MODEL, SCADS_SERIES_MODEL


###############################################################################
MODEL_COMPONENTS = {
    SCADS_SERIES_MODEL: (SCADSModel, SCADSDataLoader),
    SCADS_ATOMIC_MODEL: (SCADSAtomicModel, SCADSAtomicDataLoader),
}


###############################################################################
def put_worker_result(result_queue: Any | None, payload: dict[str, Any]) -> None:
    if result_queue is None:
        return
    try:
        result_queue.put(payload, block=False)
    except Exception:
        try:
            result_queue.put(payload)
        except Exception:
            return


###############################################################################
class TrainingProcessRunner:
    # -------------------------------------------------------------------------
    def __init__(
        self,
        worker: Any | None = None,
        *,
        snapshot_access: TrainingDataAccess,
        artifact_root: Path,
        checkpoints_dir: Path,
    ) -> None:
        self.worker = worker
        self.data_serializer = TrainingDataSerializer(snapshot_access, artifact_root)
        self.model_serializer = ModelSerializer(checkpoints_dir)

    # -------------------------------------------------------------------------
    def should_stop(self) -> bool:
        if self.worker is None:
            return False
        checker = getattr(self.worker, "is_interrupted", None)
        if callable(checker):
            return bool(checker())
        return False

    # -------------------------------------------------------------------------
    def on_epoch_end(self, epoch: int, total_epochs: int, logs: dict[str, Any]) -> None:
        self.send_training_message(
            self.worker,
            {
                "type": "epoch_end",
                "epoch": epoch,
                "total_epochs": total_epochs,
                "logs": logs,
            },
        )

    # -------------------------------------------------------------------------
    def log(self, message: str) -> None:
        self.send_training_message(
            self.worker,
            {
                "type": "log",
                "message": message,
            },
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_model_name(name: str | None) -> str:
        if not name:
            return SCADS_SERIES_MODEL
        lowered = name.strip().lower()
        if "atomic" in lowered:
            return SCADS_ATOMIC_MODEL
        return SCADS_SERIES_MODEL

    # -------------------------------------------------------------------------
    @staticmethod
    def send_training_message(worker: Any | None, payload: dict[str, Any]) -> None:
        if worker is None:
            return
        sender = getattr(worker, "send_message", None)
        if not callable(sender):
            return
        try:
            sender(payload)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to send training message: %s", exc)

    # -------------------------------------------------------------------------
    def ensure_required_columns(self, data: Any, required: list[str]) -> None:
        if data is None or getattr(data, "empty", True):
            raise ValueError("Training dataset is empty.")
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Training dataset missing columns: {', '.join(missing)}")

    # -------------------------------------------------------------------------
    def validate_resume_model(self, model: Any) -> None:
        optimizer = getattr(model, "optimizer", None)
        loss = getattr(model, "loss", None)
        if optimizer is None or loss is None:
            raise ValueError(
                "Checkpoint model is not compiled or missing optimizer state. "
                "Resume requires a compiled model with saved optimizer momentum."
            )
        opt_vars: list[Any] = []
        variables = getattr(optimizer, "variables", None)
        if variables is not None:
            if callable(variables):
                opt_vars = list(variables())
            else:
                opt_vars = list(variables)
        if not opt_vars:
            get_weights = getattr(optimizer, "get_weights", None)
            if callable(get_weights):
                opt_vars = list(get_weights())
        if not opt_vars:
            raise ValueError(
                "Checkpoint optimizer state is empty. Resume requires optimizer "
                "momentum to be saved in the checkpoint."
            )

    # -------------------------------------------------------------------------
    def start_training(self, configuration: dict[str, Any]) -> None:
        dataset_label = self.data_serializer.normalize_dataset_label(
            configuration.get("dataset_label")
        )
        train_data, validation_data, metadata = self.data_serializer.load_training_data(
            dataset_label
        )
        if train_data.empty or validation_data.empty:
            raise ValueError("No training data available. Build the dataset first.")

        selected_model = self.normalize_model_name(configuration.get("selected_model"))
        model_builder, dataloader_builder = MODEL_COMPONENTS.get(
            selected_model, MODEL_COMPONENTS[SCADS_SERIES_MODEL]
        )

        required_columns = [
            "temperature",
            "pressure",
            "adsorbed_amount",
            "adsorbate_encoded_SMILE",
            "adsorbate_molecular_weight",
            "encoded_adsorbent",
        ]
        self.ensure_required_columns(train_data, required_columns)
        self.ensure_required_columns(validation_data, required_columns)

        train_loader = dataloader_builder(
            configuration,
            metadata.model_dump(),
            shuffle=configuration.get("shuffle_dataset", True),
        )
        val_loader = dataloader_builder(
            configuration, metadata.model_dump(), shuffle=False
        )
        train_dataset = train_loader.build_training_dataloader(train_data)
        validation_dataset = val_loader.build_training_dataloader(validation_data)

        DeviceConfig(configuration).set_device()
        custom_name = configuration.get("custom_name")
        if custom_name and isinstance(custom_name, str) and custom_name.strip():
            safe_name = "".join(
                c for c in custom_name.strip() if c.isalnum() or c in ("-", "_")
            )
            model_name = safe_name if safe_name else selected_model.replace(" ", "_")
        else:
            model_name = selected_model.replace(" ", "_")

        self.model_serializer = ModelSerializer(
            self.model_serializer.checkpoints_dir,
            model_name=model_name,
        )
        checkpoint_path = self.model_serializer.create_checkpoint_folder()

        wrapper = model_builder(configuration, metadata.model_dump())
        model = wrapper.get_model(model_summary=True)

        trainer = ModelTraining(configuration, metadata.model_dump())
        model, history = trainer.train_model(
            model,
            train_dataset,
            validation_dataset,
            checkpoint_path,
            should_stop=self.should_stop,
            on_epoch_end=self.on_epoch_end,
            worker=self.worker,
        )

        self.model_serializer.save_pretrained_model(model, checkpoint_path)
        self.model_serializer.save_training_configuration(
            checkpoint_path, history, configuration, metadata
        )

    # -------------------------------------------------------------------------
    def resume_training(self, checkpoint: str, additional_epochs: int) -> None:
        (
            model,
            train_config,
            model_metadata,
            session,
            checkpoint_path,
        ) = self.model_serializer.load_checkpoint(checkpoint)
        self.validate_resume_model(model)

        dataset_label = self.data_serializer.normalize_dataset_label(
            train_config.get("dataset_label")
        )
        current_metadata = self.data_serializer.load_training_metadata(dataset_label)
        if not self.data_serializer.validate_metadata(current_metadata, model_metadata):
            raise ValueError(
                "Training dataset metadata does not match the checkpoint. "
                "Rebuild the dataset using the checkpoint configuration before resuming."
            )

        train_data, validation_data, _ = self.data_serializer.load_training_data(
            dataset_label
        )
        if train_data.empty or validation_data.empty:
            raise ValueError("No training data available. Build the dataset first.")

        selected_model = self.normalize_model_name(train_config.get("selected_model"))
        _, dataloader_builder = MODEL_COMPONENTS.get(
            selected_model, MODEL_COMPONENTS[SCADS_SERIES_MODEL]
        )

        required_columns = [
            "temperature",
            "pressure",
            "adsorbed_amount",
            "adsorbate_encoded_SMILE",
            "adsorbate_molecular_weight",
            "encoded_adsorbent",
        ]
        self.ensure_required_columns(train_data, required_columns)
        self.ensure_required_columns(validation_data, required_columns)

        train_loader = dataloader_builder(
            train_config,
            model_metadata.model_dump(),
            shuffle=train_config.get("shuffle_dataset", True),
        )
        val_loader = dataloader_builder(
            train_config, model_metadata.model_dump(), shuffle=False
        )
        train_dataset = train_loader.build_training_dataloader(train_data)
        validation_dataset = val_loader.build_training_dataloader(validation_data)

        DeviceConfig(train_config).set_device()

        from_epoch = session.get("epochs", 0)
        total_epochs = from_epoch + additional_epochs
        self.send_training_message(
            self.worker,
            {
                "type": "state_update",
                "current_epoch": from_epoch,
                "total_epochs": total_epochs,
            },
        )

        trainer = ModelTraining(train_config, model_metadata.model_dump())
        model, history = trainer.resume_training(
            model,
            train_dataset,
            validation_dataset,
            checkpoint_path,
            session,
            additional_epochs,
            should_stop=self.should_stop,
            on_epoch_end=self.on_epoch_end,
            worker=self.worker,
        )

        self.model_serializer.save_pretrained_model(model, checkpoint_path)
        self.model_serializer.save_training_configuration(
            checkpoint_path, history, train_config, model_metadata
        )


###############################################################################
def run_training_process(
    configuration: dict[str, Any] | None,
    checkpoint: str | None = None,
    additional_epochs: int = 0,
    worker: Any | None = None,
    config_payload: dict[str, Any] | None = None,
    artifact_root: str = "",
    checkpoints_dir: str = "",
) -> None:
    result_queue = getattr(worker, "result_queue", None)
    stop_event = getattr(worker, "stop_event", None)
    snapshot_access = None
    try:
        if stop_event is not None and stop_event.is_set():
            put_worker_result(result_queue, {"result": {}})
            return
        if config_payload is None or not artifact_root or not checkpoints_dir:
            raise ValueError("ML process runtime configuration and paths are required.")
        # Deferred on purpose: server.models must not statically import the
        # repository-backed services (enforced by the dependency-boundary tests).
        from server.services.training_data import open_training_data_service

        runtime_config = AdsmodConfig.model_validate(config_payload)
        snapshot_access = open_training_data_service(runtime_config)
        runner = TrainingProcessRunner(
            worker=worker,
            snapshot_access=snapshot_access,
            artifact_root=Path(artifact_root),
            checkpoints_dir=Path(checkpoints_dir),
        )
        if checkpoint:
            runner.log(
                f"Resuming training from checkpoint {checkpoint} for {additional_epochs} additional epochs."
            )
            runner.resume_training(checkpoint, additional_epochs)
            put_worker_result(
                result_queue, {"result": {"success": True, "checkpoint": checkpoint}}
            )
            return
        if configuration is None:
            raise ValueError("Training configuration is required.")
        runner.log("Starting training session.")
        runner.start_training(configuration)
        put_worker_result(result_queue, {"result": {"success": True}})
    except WorkerInterrupted:
        put_worker_result(result_queue, {"result": {}})
    except Exception as exc:  # noqa: BLE001
        put_worker_result(result_queue, {"error": str(exc)})
    finally:
        if snapshot_access is not None:
            snapshot_access.close()

