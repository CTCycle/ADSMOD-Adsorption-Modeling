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
    metrics: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    # ---------------------------------------------------------------------
    def update(self, **kwargs: Any) -> None:
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    # ---------------------------------------------------------------------
    def add_log(self, message: str) -> None:
        with self.lock:
            self.log.append(message)
            # Keep only last 1000 logs
            if len(self.log) > 1000:
                self.log = self.log[-1000:]

    # ---------------------------------------------------------------------
    def add_history(self, epoch_data: dict[str, Any]) -> None:
        with self.lock:
            self.history.append(epoch_data)

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
                "metrics": self.metrics.copy(),
                "history": list(self.history),
                "log": list(self.log),
            }

###############################################################################
class TrainingManager:
    def __init__(self) -> None:
        self.state = TrainingState()
        self._thread: threading.Thread | None = None
        self._thread_lock = threading.Lock()
        self.data_serializer = TrainingDataSerializer()
        self.model_serializer = ModelSerializer()

    # -------------------------------------------------------------------------
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
                metrics={},
                history=[],
                log=[],
            )
            self.state.add_log(f"Starting training session: {session_id}")
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
                metrics={},
                history=[],
                log=[],
            )
            self.state.add_log(f"Resuming training session: {session_id}")
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
        self.state.add_log("Stop requested by user...")

    # ---------------------------------------------------------------------
    def _on_epoch_end(self, epoch: int, total_epochs: int, logs: dict[str, Any]) -> None:
        self.state.update(current_epoch=epoch, total_epochs=total_epochs)

        # Extract metrics
        loss_value = logs.get("loss")
        loss = float(loss_value) if isinstance(loss_value, (int, float)) else 0.0
        val_loss_value = logs.get("val_loss")
        val_loss = (
            float(val_loss_value) if isinstance(val_loss_value, (int, float)) else 0.0
        )

        accuracy_value = None
        for key in ["accuracy", "MaskedAccuracy", "masked_accuracy"]:
            candidate = logs.get(key)
            if isinstance(candidate, (int, float)):
                accuracy_value = candidate
                break
        accuracy = (
            float(accuracy_value) if isinstance(accuracy_value, (int, float)) else 0.0
        )

        val_accuracy_value = None
        for key in ["val_accuracy", "val_MaskedAccuracy", "val_masked_accuracy"]:
            candidate = logs.get(key)
            if isinstance(candidate, (int, float)):
                val_accuracy_value = candidate
                break
        val_accuracy = (
            float(val_accuracy_value)
            if isinstance(val_accuracy_value, (int, float))
            else 0.0
        )

        masked_r2_value = None
        for key in ["MaskedR2", "masked_r2", "masked_r_squared"]:
            candidate = logs.get(key)
            if isinstance(candidate, (int, float)):
                masked_r2_value = candidate
                break
        masked_r2 = (
            float(masked_r2_value) if isinstance(masked_r2_value, (int, float)) else 0.0
        )

        val_masked_r2_value = None
        for key in ["val_MaskedR2", "val_masked_r2", "val_masked_r_squared"]:
            candidate = logs.get(key)
            if isinstance(candidate, (int, float)):
                val_masked_r2_value = candidate
                break
        val_masked_r2 = (
            float(val_masked_r2_value)
            if isinstance(val_masked_r2_value, (int, float))
            else 0.0
        )

        metrics = {
            "loss": loss,
            "val_loss": val_loss,
        }
        if isinstance(accuracy_value, (int, float)):
            metrics["accuracy"] = accuracy
        if isinstance(val_accuracy_value, (int, float)):
            metrics["val_accuracy"] = val_accuracy
        if isinstance(masked_r2_value, (int, float)):
            metrics["masked_r2"] = masked_r2
        if isinstance(val_masked_r2_value, (int, float)):
            metrics["val_masked_r2"] = val_masked_r2
        self.state.update(metrics=metrics)

        metric_label = "acc" if isinstance(accuracy_value, (int, float)) else "r2"
        metric_value = accuracy if isinstance(accuracy_value, (int, float)) else masked_r2
        val_metric_value = (
            val_accuracy if isinstance(val_accuracy_value, (int, float)) else val_masked_r2
        )

        # Add generic log entry
        log_message = (
            f"Epoch {epoch}/{total_epochs} - loss: {loss:.4f} - "
            f"{metric_label}: {metric_value:.4f} - val_loss: {val_loss:.4f} - "
            f"val_{metric_label}: {val_metric_value:.4f}"
        )
        self.state.add_log(log_message)
        
        # Add to history for plotting
        history_entry = {
            "epoch": epoch,
            **metrics
        }
        self.state.add_history(history_entry)

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
            configuration, metadata.model_dump(), shuffle=configuration.get("shuffle_dataset", True)
        )
        val_loader = dataloader_builder(configuration, metadata.model_dump(), shuffle=False)
        train_dataset = train_loader.build_training_dataloader(train_data)
        validation_dataset = val_loader.build_training_dataloader(validation_data)

        DeviceConfig(configuration).set_device()
        custom_name = configuration.get("custom_name")
        if custom_name and isinstance(custom_name, str) and custom_name.strip():
            # Sanitize custom name to be safe for file system
            safe_name = "".join(c for c in custom_name.strip() if c.isalnum() or c in ("-", "_"))
            model_name = safe_name if safe_name else selected_model.replace(" ", "_")
        else:
            model_name = selected_model.replace(" ", "_")
            
        self.model_serializer = ModelSerializer(model_name=model_name)
        checkpoint_path = self.model_serializer.create_checkpoint_folder()

        wrapper = model_builder(configuration, metadata.model_dump())
        model = wrapper.get_model(model_summary=True)

        trainer = ModelTraining(configuration, metadata.model_dump())
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
            train_config, model_metadata.model_dump(), shuffle=train_config.get("shuffle_dataset", True)
        )
        val_loader = dataloader_builder(train_config, model_metadata.model_dump(), shuffle=False)
        train_dataset = train_loader.build_training_dataloader(train_data)
        validation_dataset = val_loader.build_training_dataloader(validation_data)

        DeviceConfig(train_config).set_device()

        from_epoch = session.get("epochs", 0)
        total_epochs = from_epoch + additional_epochs
        self.state.update(current_epoch=from_epoch, total_epochs=total_epochs)

        trainer = ModelTraining(train_config, model_metadata.model_dump())
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
