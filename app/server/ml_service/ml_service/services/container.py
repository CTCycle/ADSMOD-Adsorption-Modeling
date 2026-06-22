from __future__ import annotations

from ml_service.common.utils.logger import logger
from ml_service.learning.training.manager import TrainingManager
from ml_service.services.training import (
    TrainingJobRunner,
    TrainingService,
    TrainingSession,
)
from shared.services.jobs import JobManager


###############################################################################
class MlServiceContainer:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.job_manager = JobManager(logger=logger)
        self.training_manager = TrainingManager()
        self.training_session = TrainingSession(training_manager=self.training_manager)
        self.training_job_runner = TrainingJobRunner(
            session=self.training_session,
            job_manager=self.job_manager,
            training_manager=self.training_manager,
        )
        self.training_service = TrainingService(
            job_manager=self.job_manager,
            training_manager=self.training_manager,
            training_session=self.training_session,
            training_job_runner=self.training_job_runner,
        )
