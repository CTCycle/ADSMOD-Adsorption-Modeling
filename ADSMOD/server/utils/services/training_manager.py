from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from ADSMOD.server.utils.constants import SCADS_ATOMIC_MODEL, SCADS_SERIES_MODEL
from ADSMOD.server.utils.learning.device import DeviceConfig
from ADSMOD.server.utils.learning.models.qmodel import SCADSAtomicModel, SCADSModel
from ADSMOD.server.utils.learning.training.fitting import ModelTraining
from ADSMOD.server.utils.logger import logger
from ADSMOD.server.utils.repository.serializer import ModelSerializer, TrainingDataSerializer
from ADSMOD.server.utils.services.loader import (
    SCADSAtomicDataLoader,
    SCADSDataLoader,
)


MODEL_COMPONENTS = {
    SCADS_SERIES_MODEL: (SCADSModel, SCADSDataLoader),
    SCADS_ATOMIC_MODEL: (SCADSAtomicModel, SCADSAtomicDataLoader),
}


def normalize_model_name(name: str | None) -> str:
    if not name:
        return SCADS_SERIES_MODEL
    lowered = name.strip().lower()
    if "atomic" in lowered:
        return SCADS_ATOMIC_MODEL
    return SCADS_SERIES_MODEL


@dataclass
class TrainingState:
    is_training: bool = False
    current_epoch: int = 0
    total_epochs: int = 0
    session_id: str | None = None
    stop_requested: bool = False
    last_error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    # ---------------------------------------------------------------------
    def update(self, **kwargs: Any) -> None:
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    # ---------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "is_training": self.is_training,
                "current_epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "session_id": self.session_id,
                "stop_requested": self.stop_requested,
                "last_error": self.last_error,
            }


class TrainingManager:
    def __init__(self) -> None:
        self.state = TrainingState()
        self._thread: threading.Thread | None = None
        self._thread_lock = threading.Lock()
        self.data_serializer = TrainingDataSerializer()
        self.model_serializer = ModelSerializer()

    # ---------------------------------------------------------------------
    def start_training(self, configuration: dict[str, Any]) -> str:
        with self._thread_lock:
            if self.state.is_training:
                raise RuntimeError("Training is already in progress.")
            session_id = str(uuid.uuid4())[:8]
            total_epochs = int(configuration.get("epochs", 0))
            self.state.update(
                is_training=True,
                current_epoch=0,
                total_epochs=total_epochs,
                session_id=session_id,
                stop_requested=False,
                last_error=None,
            )
            self._thread = threading.Thread(
                target=self._run_training,
                args=(configuration, None, 0),
                daemon=True,
            )
            self._thread.start()
            return session_id

    # ---------------------------------------------------------------------
    def resume_training(self, checkpoint: str, additional_epochs: int) -> str:
        with self._thread_lock:
            if self.state.is_training:
                raise RuntimeError("Training is already in progress.")
            session_id = str(uuid.uuid4())[:8]
            self.state.update(
                is_training=True,
                current_epoch=0,
                total_epochs=additional_epochs,
                session_id=session_id,
                stop_requested=False,
                last_error=None,
            )
            self._thread = threading.Thread(
                target=self._run_training,
                args=({}, checkpoint, additional_epochs),
                daemon=True,
            )
            self._thread.start()
            return session_id

    # ---------------------------------------------------------------------
    def stop_training(self) -> None:
        if not self.state.is_training:
            return
        self.state.update(stop_requested=True)

    # ---------------------------------------------------------------------
    def _on_epoch_end(self, epoch: int, total_epochs: int, logs: dict[str, Any]) -> None:
        self.state.update(current_epoch=epoch, total_epochs=total_epochs)

    # ---------------------------------------------------------------------
    def _should_stop(self) -> bool:
        return self.state.stop_requested

    # ---------------------------------------------------------------------
    @staticmethod
    def _ensure_required_columns(data: Any, required: list[str]) -> None:
        if data is None or getattr(data, "empty", True):
            raise ValueError("Training dataset is empty.")
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Training dataset missing columns: {', '.join(missing)}")

    # ---------------------------------------------------------------------
    def _run_training(
        self, configuration: dict[str, Any], checkpoint: str | None, additional_epochs: int
    ) -> None:
        try:
            if checkpoint:
                self._resume_training_internal(checkpoint, additional_epochs)
            else:
                self._start_training_internal(configuration)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Training pipeline failed")
            self.state.update(last_error=str(exc))
        finally:
            self.state.update(is_training=False, stop_requested=False)

    # ---------------------------------------------------------------------
    def _start_training_internal(self, configuration: dict[str, Any]) -> None:
        train_data, validation_data, metadata = self.data_serializer.load_training_data()
        if train_data.empty or validation_data.empty:
            raise ValueError("No training data available. Build the dataset first.")

        selected_model = normalize_model_name(configuration.get("selected_model"))
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
        self._ensure_required_columns(train_data, required_columns)
        self._ensure_required_columns(validation_data, required_columns)

        train_loader = dataloader_builder(
            configuration, metadata, shuffle=configuration.get("shuffle_dataset", True)
        )
        val_loader = dataloader_builder(configuration, metadata, shuffle=False)
        train_dataset = train_loader.build_training_dataloader(train_data)
        validation_dataset = val_loader.build_training_dataloader(validation_data)

        DeviceConfig(configuration).set_device()
        self.model_serializer = ModelSerializer(model_name=selected_model.replace(" ", "_"))
        checkpoint_path = self.model_serializer.create_checkpoint_folder()

        wrapper = model_builder(configuration, metadata)
        model = wrapper.get_model(model_summary=True)

        trainer = ModelTraining(configuration, metadata)
        model, history = trainer.train_model(
            model,
            train_dataset,
            validation_dataset,
            checkpoint_path,
            should_stop=self._should_stop,
            on_epoch_end=self._on_epoch_end,
        )

        self.model_serializer.save_pretrained_model(model, checkpoint_path)
        self.model_serializer.save_training_configuration(
            checkpoint_path, history, configuration, metadata
        )

    # ---------------------------------------------------------------------
    def _resume_training_internal(self, checkpoint: str, additional_epochs: int) -> None:
        (
            model,
            train_config,
            model_metadata,
            session,
            checkpoint_path,
        ) = self.model_serializer.load_checkpoint(checkpoint)

        current_metadata = self.data_serializer.load_training_metadata()
        if not self.data_serializer.validate_metadata(current_metadata, model_metadata):
            raise ValueError(
                "Training dataset metadata does not match the checkpoint. "
                "Rebuild the dataset using the checkpoint configuration before resuming."
            )

        train_data, validation_data, _ = self.data_serializer.load_training_data()
        if train_data.empty or validation_data.empty:
            raise ValueError("No training data available. Build the dataset first.")

        selected_model = normalize_model_name(train_config.get("selected_model"))
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
        self._ensure_required_columns(train_data, required_columns)
        self._ensure_required_columns(validation_data, required_columns)

        train_loader = dataloader_builder(
            train_config, model_metadata, shuffle=train_config.get("shuffle_dataset", True)
        )
        val_loader = dataloader_builder(train_config, model_metadata, shuffle=False)
        train_dataset = train_loader.build_training_dataloader(train_data)
        validation_dataset = val_loader.build_training_dataloader(validation_data)

        DeviceConfig(train_config).set_device()

        from_epoch = session.get("epochs", 0)
        total_epochs = from_epoch + additional_epochs
        self.state.update(current_epoch=from_epoch, total_epochs=total_epochs)

        trainer = ModelTraining(train_config, model_metadata)
        model, history = trainer.resume_training(
            model,
            train_dataset,
            validation_dataset,
            checkpoint_path,
            session,
            additional_epochs,
            should_stop=self._should_stop,
            on_epoch_end=self._on_epoch_end,
        )

        self.model_serializer.save_pretrained_model(model, checkpoint_path)
        self.model_serializer.save_training_configuration(
            checkpoint_path, history, train_config, model_metadata
        )


training_manager = TrainingManager()
