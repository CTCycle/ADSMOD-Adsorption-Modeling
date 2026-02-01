from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from ADSMOD.server.utils.constants import SCADS_ATOMIC_MODEL, SCADS_SERIES_MODEL
from ADSMOD.server.services.learning.callbacks import WorkerInterrupted
from ADSMOD.server.services.learning.device import DeviceConfig
from ADSMOD.server.services.learning.models.qmodel import SCADSAtomicModel, SCADSModel
from ADSMOD.server.services.learning.training.fitting import ModelTraining
from ADSMOD.server.utils.logger import logger
from ADSMOD.server.repositories.serializer import (
    ModelSerializer,
    TrainingDataSerializer,
)
from ADSMOD.server.services.learning.loader import (
    SCADSAtomicDataLoader,
    SCADSDataLoader,
)


MODEL_COMPONENTS = {
    SCADS_SERIES_MODEL: (SCADSModel, SCADSDataLoader),
    SCADS_ATOMIC_MODEL: (SCADSAtomicModel, SCADSAtomicDataLoader),
}


class TrainingProcessRunner:
    def __init__(
        self,
        worker: Any | None = None,
    ) -> None:
        self.worker = worker
        self.data_serializer = TrainingDataSerializer()
        self.model_serializer = ModelSerializer()

    # ---------------------------------------------------------------------
    def should_stop(self) -> bool:
        if self.worker is None:
            return False
        checker = getattr(self.worker, "is_interrupted", None)
        if callable(checker):
            return bool(checker())
        return False

    # ---------------------------------------------------------------------
    def on_epoch_end(
        self, epoch: int, total_epochs: int, logs: dict[str, Any]
    ) -> None:
        self.send_training_message(
            self.worker,
            {
                "type": "epoch_end",
                "epoch": epoch,
                "total_epochs": total_epochs,
                "logs": logs,
            },
        )

    # ---------------------------------------------------------------------
    def log(self, message: str) -> None:
        self.send_training_message(
            self.worker,
            {
                "type": "log",
                "message": message,
            },
        )

    # ---------------------------------------------------------------------
    @staticmethod
    def normalize_model_name(name: str | None) -> str:
        if not name:
            return SCADS_SERIES_MODEL
        lowered = name.strip().lower()
        if "atomic" in lowered:
            return SCADS_ATOMIC_MODEL
        return SCADS_SERIES_MODEL

    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    def ensure_required_columns(self, data: Any, required: list[str]) -> None:
        if data is None or getattr(data, "empty", True):
            raise ValueError("Training dataset is empty.")
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Training dataset missing columns: {', '.join(missing)}")

    # ---------------------------------------------------------------------
    def start_training(self, configuration: dict[str, Any]) -> None:
        dataset_label = self.data_serializer.normalize_dataset_label(
            configuration.get("dataset_label")
        )
        train_data, validation_data, metadata = (
            self.data_serializer.load_training_data(dataset_label)
        )
        if train_data.empty or validation_data.empty:
            raise ValueError("No training data available. Build the dataset first.")

        selected_model = self.normalize_model_name(
            configuration.get("selected_model")
        )
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
            should_stop=self.should_stop,
            on_epoch_end=self.on_epoch_end,
            worker=self.worker,
        )

        self.model_serializer.save_pretrained_model(model, checkpoint_path)
        self.model_serializer.save_training_configuration(
            checkpoint_path, history, configuration, metadata
        )

    # ---------------------------------------------------------------------
    def resume_training(self, checkpoint: str, additional_epochs: int) -> None:
        (
            model,
            train_config,
            model_metadata,
            session,
            checkpoint_path,
        ) = self.model_serializer.load_checkpoint(checkpoint)

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
        send_training_message(
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


def run_training_process(
    configuration: dict[str, Any] | None,
    checkpoint: str | None = None,
    additional_epochs: int = 0,
    worker: Any | None = None,
) -> None:
    result_queue = getattr(worker, "result_queue", None)
    stop_event = getattr(worker, "stop_event", None)

    def _safe_put(payload: dict[str, Any]) -> None:
        if result_queue is None:
            return
        try:
            result_queue.put(payload, block=False)
        except Exception:
            try:
                result_queue.put(payload)
            except Exception:
                return

    try:
        if stop_event is not None and stop_event.is_set():
            _safe_put({"result": {}})
            return

        runner = TrainingProcessRunner(worker=worker)
        if checkpoint:
            runner.log(
                f"Resuming training from checkpoint {checkpoint} "
                f"for {additional_epochs} additional epochs."
            )
            runner.resume_training(checkpoint, additional_epochs)
            _safe_put({"result": {"success": True, "checkpoint": checkpoint}})
            return

        if configuration is None:
            raise ValueError("Training configuration is required.")

        runner.log("Starting training session.")
        runner.start_training(configuration)
        _safe_put({"result": {"success": True}})
    except WorkerInterrupted:
        _safe_put({"result": {}})
    except Exception as exc:  # noqa: BLE001
        _safe_put({"error": str(exc)})


@dataclass
class TrainingState:
    is_training: bool = False
    current_epoch: int = 0
    total_epochs: int = 0
    progress: float = 0.0
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
                "progress": self.progress,
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
        self.data_serializer = TrainingDataSerializer()
        self.model_serializer = ModelSerializer()

    # ---------------------------------------------------------------------
    def handle_process_message(self, job_id: str, message: dict[str, Any]) -> None:
        if job_id != self.state.session_id:
            return

        message_type = message.get("type")
        if message_type == "epoch_end":
            epoch = message.get("epoch")
            total_epochs = message.get("total_epochs")
            logs = message.get("logs")
            if (
                isinstance(epoch, int)
                and isinstance(total_epochs, int)
                and isinstance(logs, dict)
            ):
                self._on_epoch_end(epoch, total_epochs, logs)
            return

        if message_type == "state_update":
            current_epoch = message.get("current_epoch")
            total_epochs = message.get("total_epochs")
            update_payload: dict[str, Any] = {}
            if isinstance(current_epoch, int):
                update_payload["current_epoch"] = current_epoch
            if isinstance(total_epochs, int):
                update_payload["total_epochs"] = total_epochs
            if update_payload:
                self.state.update(**update_payload)
            return

        if message_type == "training_update":
            current_epoch = message.get("epoch")
            total_epochs = message.get("total_epochs")
            progress_percent = message.get("progress_percent")
            update_payload: dict[str, Any] = {}
            if isinstance(current_epoch, int):
                update_payload["current_epoch"] = current_epoch
            if isinstance(total_epochs, int):
                update_payload["total_epochs"] = total_epochs
            if isinstance(progress_percent, (int, float)):
                update_payload["progress"] = float(progress_percent)
            metrics_update: dict[str, float] = {}
            for key in [
                "loss",
                "val_loss",
                "accuracy",
                "val_accuracy",
                "masked_r2",
                "val_masked_r2",
            ]:
                value = message.get(key)
                if isinstance(value, (int, float)):
                    metrics_update[key] = float(value)
            if metrics_update:
                current_metrics = self.state.snapshot().get("metrics", {})
                if isinstance(current_metrics, dict):
                    merged = dict(current_metrics)
                    merged.update(metrics_update)
                    update_payload["metrics"] = merged
                else:
                    update_payload["metrics"] = metrics_update
            if update_payload:
                self.state.update(**update_payload)
            return

        if message_type == "log":
            message_text = message.get("message")
            if message_text:
                self.state.add_log(str(message_text))
            return

        if message_type == "error":
            error_text = message.get("error")
            if error_text:
                self.state.update(last_error=str(error_text))
                self.state.add_log(f"Training error: {error_text}")

    # ---------------------------------------------------------------------
    def handle_job_completion(
        self,
        job_id: str,
        status: str,
        result: dict[str, Any] | None,
        error: str | None,
    ) -> None:
        if job_id != self.state.session_id:
            return

        if status == "failed":
            self.state.update(last_error=error)
            message = error or "Training failed."
            self.state.add_log(f"Training failed: {message}")
        elif status == "cancelled":
            self.state.add_log("Training cancelled.")
        elif status == "completed":
            self.state.add_log("Training completed.")
        elif status:
            self.state.add_log(f"Training finished with status: {status}")

        completion_payload = {"is_training": False, "stop_requested": False}
        if status == "completed":
            completion_payload["progress"] = 100.0
        self.state.update(**completion_payload)

    # ---------------------------------------------------------------------
    def _on_epoch_end(
        self, epoch: int, total_epochs: int, logs: dict[str, Any]
    ) -> None:
        progress = 0.0
        if total_epochs > 0:
            progress = (epoch / total_epochs) * 100
        self.state.update(
            current_epoch=epoch, total_epochs=total_epochs, progress=progress
        )

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
        metric_value = (
            accuracy if isinstance(accuracy_value, (int, float)) else masked_r2
        )
        val_metric_value = (
            val_accuracy
            if isinstance(val_accuracy_value, (int, float))
            else val_masked_r2
        )

        # Add generic log entry
        log_message = (
            f"Epoch {epoch}/{total_epochs} - loss: {loss:.4f} - "
            f"{metric_label}: {metric_value:.4f} - val_loss: {val_loss:.4f} - "
            f"val_{metric_label}: {val_metric_value:.4f}"
        )
        self.state.add_log(log_message)

        # Add to history for plotting
        history_entry = {"epoch": epoch, **metrics}
        self.state.add_history(history_entry)


training_manager = TrainingManager()
