from __future__ import annotations

from server.configurations.settings import AdsmodConfig
from server.common.path import resolve_checkpoint_root, resolve_storage_root
from server.domain.training_data import TrainingDataAccess
from server.common.utils.logger import logger
from server.models.training.manager import TrainingManager
from server.services.jobs import JobManager
from server.services.training import TrainingJobRunner, TrainingService, TrainingSession

###############################################################################
class MlServiceContainer:

    # -------------------------------------------------------------------------
    def __init__(self, config: AdsmodConfig, *, snapshot_access: TrainingDataAccess) -> None:
        self.config = config
        self.snapshot_access = snapshot_access
        self.artifact_root = resolve_storage_root(config) / "training"
        self.checkpoints_dir = resolve_checkpoint_root(config)
        self.job_manager = JobManager(logger=logger)
        self.training_manager = TrainingManager(config, snapshot_access=self.snapshot_access, artifact_root=self.artifact_root, checkpoints_dir=self.checkpoints_dir)
        self.training_session = TrainingSession(training_manager=self.training_manager)
        self.training_job_runner = TrainingJobRunner(session=self.training_session, job_manager=self.job_manager, training_manager=self.training_manager, config=config, process_context={"config_payload": config.model_dump(mode="json"), "artifact_root": str(self.artifact_root), "checkpoints_dir": str(self.checkpoints_dir)})
        self.training_service = TrainingService(config=config, snapshot_access=self.snapshot_access, artifact_root=self.artifact_root, checkpoints_dir=self.checkpoints_dir, job_manager=self.job_manager, training_manager=self.training_manager, training_session=self.training_session, training_job_runner=self.training_job_runner)

    # -------------------------------------------------------------------------
    def shutdown(self) -> None:
        self.job_manager.shutdown()


__all__ = ["MlServiceContainer"]
