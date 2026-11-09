# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import threading
from importlib import import_module
from typing import TYPE_CHECKING, Any

from server.configurations.settings import AdsmodConfig
from server.domain.training_data import TrainingDataAccess

if TYPE_CHECKING:
    from server.services.jobs import JobManager
    from server.services.ml_container import MlServiceContainer

###############################################################################
class _LazyTrainingService:
    """Forwards attribute access to the real training service, loading it on demand."""

    # -------------------------------------------------------------------------
    def __init__(self, owner: LazyMlContainer) -> None:
        self._owner = owner

    # -------------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        return getattr(self._owner.load().training_service, name)

###############################################################################
class LazyMlContainer:
    """Defers importing torch/keras and building the ML services until first use.

    The API process only needs the machine learning stack when a training
    endpoint is called; training itself runs in a spawned child process.
    """

    # -------------------------------------------------------------------------
    def __init__(self, config: AdsmodConfig, *, snapshot_access: TrainingDataAccess) -> None:
        self.config = config
        self.snapshot_access = snapshot_access
        self.training_service = _LazyTrainingService(self)
        self._container: MlServiceContainer | None = None
        self._lock = threading.Lock()

    # -------------------------------------------------------------------------
    @property
    def job_manager(self) -> JobManager | None:
        """Return the training job manager, or None while the ML stack is unloaded."""
        container = self._container
        return container.job_manager if container is not None else None

    # -------------------------------------------------------------------------
    def load(self) -> MlServiceContainer:
        with self._lock:
            if self._container is None:
                module = import_module("server.services.ml_container")
                self._container = module.MlServiceContainer(
                    self.config, snapshot_access=self.snapshot_access
                )
            return self._container

    # -------------------------------------------------------------------------
    def apply_settings(self, config: AdsmodConfig) -> None:
        with self._lock:
            self.config = config
            if self._container is not None:
                self._container.apply_settings(config)

    # -------------------------------------------------------------------------
    def shutdown(self) -> None:
        with self._lock:
            container = self._container
        if container is not None:
            container.shutdown()


__all__ = ["LazyMlContainer"]
