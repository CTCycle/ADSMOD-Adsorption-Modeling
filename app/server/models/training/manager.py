# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import math
from typing import Any

from pathlib import Path

from server.configurations.settings import AdsmodConfig
from server.common.path import resolve_checkpoint_root, resolve_storage_root
from server.domain.training_data import TrainingDataAccess
from server.models.training.serialization.model import ModelSerializer
from server.models.training.serialization.training import TrainingDataSerializer
from server.models.training.state import TrainingState


TRAINING_UPDATE_METRIC_KEYS = (
    "loss",
    "val_loss",
    "accuracy",
    "val_accuracy",
    "masked_r2",
    "val_masked_r2",
)
EPOCH_METRIC_ALIASES = {
    "accuracy": ("accuracy", "MaskedAccuracy", "masked_accuracy"),
    "val_accuracy": ("val_accuracy", "val_MaskedAccuracy", "val_masked_accuracy"),
    "masked_r2": ("MaskedR2", "masked_r2", "masked_r_squared"),
    "val_masked_r2": ("val_MaskedR2", "val_masked_r2", "val_masked_r_squared"),
}


###############################################################################
def is_finite_number(value: Any) -> bool:
    """NaN/inf metrics (a diverged run) are treated as missing: JSON cannot carry them."""
    return isinstance(value, (int, float)) and math.isfinite(value)


###############################################################################
def first_numeric(logs: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        candidate = logs.get(key)
        if is_finite_number(candidate):
            return float(candidate)
    return None


HISTORY_KEY_ALIASES = {
    "MaskedAccuracy": "accuracy",
    "val_MaskedAccuracy": "val_accuracy",
    "masked_accuracy": "accuracy",
    "val_masked_accuracy": "val_accuracy",
    "MaskedR2": "masked_r2",
    "val_MaskedR2": "val_masked_r2",
    "masked_r_squared": "masked_r2",
    "val_masked_r_squared": "val_masked_r2",
}


###############################################################################
def run_training_process(*args: Any, **kwargs: Any) -> None:
    """Spawn target for the training worker.

    The torch/keras training code lives in `process`, imported only inside the
    child process so the API process never loads the machine learning stack.
    """
    from server.models.training.process import run_training_process as run

    run(*args, **kwargs)


###############################################################################
class TrainingManager:
    # -------------------------------------------------------------------------
    def __init__(
        self,
        config: AdsmodConfig,
        *,
        snapshot_access: TrainingDataAccess,
        artifact_root: Path | None = None,
        checkpoints_dir: Path | None = None,
    ) -> None:
        resolved_artifact_root = (
            artifact_root or resolve_storage_root(config) / "training"
        )
        resolved_checkpoints_dir = checkpoints_dir or resolve_checkpoint_root(config)
        self.state = TrainingState()
        self.data_serializer = TrainingDataSerializer(
            snapshot_access, resolved_artifact_root
        )
        self.model_serializer = ModelSerializer(resolved_checkpoints_dir)

    # -------------------------------------------------------------------------
    def build_history_entries(self, session: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(session, dict):
            return []
        session_history = session.get("history")
        if not isinstance(session_history, dict):
            return []
        lengths = [
            len(values)
            for values in session_history.values()
            if isinstance(values, list)
        ]
        if not lengths:
            return []
        max_len = max(lengths)
        entries: list[dict[str, Any]] = []
        for index in range(max_len):
            entry: dict[str, Any] = {"epoch": index + 1}
            for key, values in session_history.items():
                if not isinstance(values, list) or index >= len(values):
                    continue
                value = values[index]
                if is_finite_number(value):
                    entry[HISTORY_KEY_ALIASES.get(key, key)] = float(value)
            entries.append(entry)
        return entries

    # -------------------------------------------------------------------------
    def extract_last_metrics(
        self, history_entries: list[dict[str, Any]]
    ) -> dict[str, float]:
        if not history_entries:
            return {}
        last_entry = history_entries[-1]
        metrics: dict[str, float] = {}
        for key, value in last_entry.items():
            if key == "epoch":
                continue
            if is_finite_number(value):
                metrics[key] = float(value)
        return metrics

    # -------------------------------------------------------------------------
    def handle_process_message(self, job_id: str, message: dict[str, Any]) -> None:
        if job_id != self.state.session_id:
            return

        message_type = message.get("type")
        if message_type == "epoch_end":
            self._apply_epoch_end(message)
        elif message_type == "state_update":
            self._apply_state_update(message)
        elif message_type == "training_update":
            self._apply_training_update(message)
        elif message_type == "log":
            message_text = message.get("message")
            if message_text:
                self.state.add_log(str(message_text))
        elif message_type == "error":
            error_text = message.get("error")
            if error_text:
                self.state.update(last_error=str(error_text))
                self.state.add_log(f"Training error: {error_text}")

    # -------------------------------------------------------------------------
    def _apply_epoch_end(self, message: dict[str, Any]) -> None:
        epoch = message.get("epoch")
        total_epochs = message.get("total_epochs")
        logs = message.get("logs")
        if (
            isinstance(epoch, int)
            and isinstance(total_epochs, int)
            and isinstance(logs, dict)
        ):
            self._on_epoch_end(epoch, total_epochs, logs)

    # -------------------------------------------------------------------------
    def _apply_state_update(self, message: dict[str, Any]) -> None:
        update_payload = self._extract_epoch_fields(message, epoch_key="current_epoch")
        if update_payload:
            self.state.update(**update_payload)

    # -------------------------------------------------------------------------
    def _extract_epoch_fields(
        self, message: dict[str, Any], epoch_key: str
    ) -> dict[str, Any]:
        update_payload: dict[str, Any] = {}
        current_epoch = message.get(epoch_key)
        total_epochs = message.get("total_epochs")
        if isinstance(current_epoch, int):
            update_payload["current_epoch"] = current_epoch
        if isinstance(total_epochs, int):
            update_payload["total_epochs"] = total_epochs
        return update_payload

    # -------------------------------------------------------------------------
    def _apply_training_update(self, message: dict[str, Any]) -> None:
        update_payload = self._extract_epoch_fields(message, epoch_key="epoch")
        progress_percent = message.get("progress_percent")
        if is_finite_number(progress_percent):
            update_payload["progress"] = float(progress_percent)
        metrics_update = {
            key: float(message[key])
            for key in TRAINING_UPDATE_METRIC_KEYS
            if is_finite_number(message.get(key))
        }
        if metrics_update:
            current_metrics = self.state.snapshot().get("metrics", {})
            if isinstance(current_metrics, dict):
                update_payload["metrics"] = {**current_metrics, **metrics_update}
            else:
                update_payload["metrics"] = metrics_update
        if update_payload:
            self.state.update(**update_payload)

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    def _on_epoch_end(
        self, epoch: int, total_epochs: int, logs: dict[str, Any]
    ) -> None:
        progress = 0.0
        if total_epochs > 0:
            progress = (epoch / total_epochs) * 100
        self.state.update(
            current_epoch=epoch, total_epochs=total_epochs, progress=progress
        )

        loss = first_numeric(logs, ("loss",))
        val_loss = first_numeric(logs, ("val_loss",))
        optional_metrics = {
            name: first_numeric(logs, aliases)
            for name, aliases in EPOCH_METRIC_ALIASES.items()
        }
        metrics = {
            name: value
            for name, value in {
                "loss": loss,
                "val_loss": val_loss,
                **optional_metrics,
            }.items()
            if value is not None
        }
        self.state.update(metrics=metrics)

        if optional_metrics["accuracy"] is not None:
            metric_label, metric_value = "acc", optional_metrics["accuracy"]
        else:
            metric_label, metric_value = "r2", optional_metrics["masked_r2"]
        if optional_metrics["val_accuracy"] is not None:
            val_metric_value = optional_metrics["val_accuracy"]
        else:
            val_metric_value = optional_metrics["val_masked_r2"]

        def shown(value: float | None) -> str:
            return "nan" if value is None else f"{value:.4f}"

        # Add generic log entry
        log_message = (
            f"Epoch {epoch}/{total_epochs} - loss: {shown(loss)} - "
            f"{metric_label}: {shown(metric_value)} - val_loss: {shown(val_loss)} - "
            f"val_{metric_label}: {shown(val_metric_value)}"
        )
        self.state.add_log(log_message)

        # Add to history for plotting
        history_entry = {"epoch": epoch, **metrics}
        self.state.add_history(history_entry)
