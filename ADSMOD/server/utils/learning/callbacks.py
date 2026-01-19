from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import keras
from keras.callbacks import Callback

from ADSMOD.server.utils.logger import logger


# [CALLBACK FOR TRAINING PROGRESS]
###############################################################################
class TrainingProgressCallback(Callback):
    def __init__(
        self,
        total_epochs: int,
        on_epoch_end: Callable[[int, int, dict[str, Any]], None] | None = None,
        start_epoch: int = 0,
    ) -> None:
        super().__init__()
        self.total_epochs = total_epochs
        self.on_epoch_end_callback = on_epoch_end
        self.start_epoch = start_epoch

    # -------------------------------------------------------------------------
    def on_epoch_end(self, epoch, logs: dict | None = None) -> None:
        current_epoch = epoch + 1
        logs = logs or {}
        if self.on_epoch_end_callback is not None:
            self.on_epoch_end_callback(current_epoch, self.total_epochs, logs)


# [CALLBACK FOR TRAIN INTERRUPTION]
###############################################################################
class StopTrainingCallback(Callback):
    def __init__(self, should_stop: Callable[[], bool] | None = None) -> None:
        super().__init__()
        self.should_stop = should_stop

    # -------------------------------------------------------------------------
    def on_batch_end(self, batch, logs: dict | None = None) -> None:
        if self.should_stop is not None and self.should_stop():
            logger.info("Stop requested; halting training after batch %s", batch)
            self.model.stop_training = True

    # -------------------------------------------------------------------------
    def on_epoch_end(self, epoch, logs: dict | None = None) -> None:
        if self.should_stop is not None and self.should_stop():
            logger.info("Stop requested; halting training after epoch %s", epoch + 1)
            self.model.stop_training = True


# [CALLBACK FOR PERIODIC CHECKPOINTS]
###############################################################################
class PeriodicCheckpointCallback(Callback):
    def __init__(self, checkpoint_dir: str, frequency: int = 1) -> None:
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.frequency = max(1, int(frequency))

    # -------------------------------------------------------------------------
    def on_epoch_end(self, epoch, logs: dict | None = None) -> None:
        if (epoch + 1) % self.frequency != 0:
            return
        filename = f"model_checkpoint_E{epoch + 1:02d}.keras"
        target_path = os.path.join(self.checkpoint_dir, filename)
        try:
            self.model.save(target_path)
            logger.info("Saved checkpoint %s", target_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to save checkpoint: %s", exc)


###############################################################################
def build_training_callbacks(
    configuration: dict[str, Any],
    checkpoint_path: str | None,
    total_epochs: int,
    start_epoch: int = 0,
    should_stop: Callable[[], bool] | None = None,
    on_epoch_end: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> list[Callback]:
    callbacks_list: list[Callback] = [
        TrainingProgressCallback(
            total_epochs=total_epochs,
            on_epoch_end=on_epoch_end,
            start_epoch=start_epoch,
        ),
        StopTrainingCallback(should_stop=should_stop),
        keras.callbacks.TerminateOnNaN(),
    ]

    if configuration.get("save_checkpoints", False) and checkpoint_path:
        frequency = int(configuration.get("checkpoints_frequency", 1))
        callbacks_list.append(
            PeriodicCheckpointCallback(
                checkpoint_dir=checkpoint_path,
                frequency=frequency,
            )
        )

    return callbacks_list
