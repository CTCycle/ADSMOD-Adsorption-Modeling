from __future__ import annotations

from adsmod_common.config import AdsmodConfig
from adsmod_common.paths import resolve_checkpoint_root, resolve_storage_root
from adsmod_ml.clients.core_client import CoreSnapshotClient
from adsmod_ml.common.utils.logger import logger
from adsmod_ml.learning.training.manager import TrainingManager
from adsmod_ml.services.training import (
    TrainingJobRunner,
    TrainingService,
    TrainingSession,
)
from adsmod_ml.services.jobs import JobManager

###############################################################################
class MlServiceContainer:

    # -------------------------------------------------------------------------
    def __init__(self, config: AdsmodConfig, *, internal_token: str | None = None) -> None:
        self.config = config
        self.snapshot_client = CoreSnapshotClient.from_config(
            config,
            internal_token=internal_token,
        )
        self.artifact_root = resolve_storage_root(config) / "training"
        self.checkpoints_dir = resolve_checkpoint_root(config)
        self.job_manager = JobManager(logger=logger)
        self.training_manager = TrainingManager(
            config,
            snapshot_client=self.snapshot_client,
            artifact_root=self.artifact_root,
            checkpoints_dir=self.checkpoints_dir,
        )
        self.training_session = TrainingSession(training_manager=self.training_manager)
        self.training_job_runner = TrainingJobRunner(
            session=self.training_session,
            job_manager=self.job_manager,
            training_manager=self.training_manager,
            config=config,
            process_context={
                "core_base_url": self.snapshot_client.base_url,
                "internal_token": self.snapshot_client.internal_token,
                "artifact_root": str(self.artifact_root),
                "checkpoints_dir": str(self.checkpoints_dir),
            },
        )
        self.training_service = TrainingService(
            config=config,
            snapshot_client=self.snapshot_client,
            artifact_root=self.artifact_root,
            checkpoints_dir=self.checkpoints_dir,
            internal_token=self.snapshot_client.internal_token,
            job_manager=self.job_manager,
            training_manager=self.training_manager,
            training_session=self.training_session,
            training_job_runner=self.training_job_runner,
        )
