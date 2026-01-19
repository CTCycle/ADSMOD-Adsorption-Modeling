from __future__ import annotations

from typing import Any

from keras import Model
from keras.utils import set_random_seed

from ADSMOD.server.utils.learning.callbacks import build_training_callbacks


# [TOOLS FOR TRAINING MACHINE LEARNING MODELS]
###############################################################################
class ModelTraining:
    def __init__(
        self, configuration: dict[str, Any], metadata: dict[str, Any] | None = None
    ) -> None:
        set_random_seed(configuration.get("training_seed", 42))
        self.configuration = configuration
        self.metadata = metadata

    # -------------------------------------------------------------------------
    def train_model(
        self,
        model: Model,
        train_data: Any,
        validation_data: Any,
        checkpoint_path: str | None,
        **kwargs,
    ) -> tuple[Model, dict[str, Any]]:
        total_epochs = int(self.configuration.get("epochs", 10))
        callbacks_list = build_training_callbacks(
            self.configuration,
            checkpoint_path,
            total_epochs=total_epochs,
            start_epoch=0,
            should_stop=kwargs.get("should_stop"),
            on_epoch_end=kwargs.get("on_epoch_end"),
        )

        session = model.fit(
            train_data,
            epochs=total_epochs,
            validation_data=validation_data,
            callbacks=callbacks_list,
        )

        history = {"history": session.history, "epochs": session.epoch[-1] + 1}

        return model, history

    # -------------------------------------------------------------------------
    def resume_training(
        self,
        model: Model,
        train_data: Any,
        validation_data: Any,
        checkpoint_path: str | None,
        session: dict | None = None,
        additional_epochs: int = 10,
        **kwargs,
    ) -> tuple[Model, dict[str, Any]]:
        session = session or {}
        from_epoch = session.get("epochs", 0)
        total_epochs = from_epoch + additional_epochs
        callbacks_list = build_training_callbacks(
            self.configuration,
            checkpoint_path,
            total_epochs=total_epochs,
            start_epoch=from_epoch,
            should_stop=kwargs.get("should_stop"),
            on_epoch_end=kwargs.get("on_epoch_end"),
        )

        new_session = model.fit(
            train_data,
            epochs=total_epochs,
            validation_data=validation_data,
            callbacks=callbacks_list,
            initial_epoch=from_epoch,
        )

        history = {"history": new_session.history, "epochs": new_session.epoch[-1] + 1}
        if session and "history" in session:
            merged_history: dict[str, list[Any]] = {}
            for key, values in session["history"].items():
                merged_history[key] = list(values) + list(new_session.history.get(key, []))
            history["history"] = merged_history

        return model, history
