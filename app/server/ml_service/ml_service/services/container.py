from __future__ import annotations

from ml_service.common.utils.logger import logger
from ml_service.services.jobs import JobManager
from ml_service.services.training import (
    TrainingJobRunner,
    TrainingService,
    TrainingSession,
)


class MlServiceContainer:
    def __init__(self) -> None:
        self.job_manager = JobManager(logger=logger)
        self.training_session = TrainingSession()
        self.training_job_runner = TrainingJobRunner(
            session=self.training_session,
            job_manager=self.job_manager,
        )
        self.training_service = TrainingService(
            job_manager=self.job_manager,
            training_session=self.training_session,
            training_job_runner=self.training_job_runner,
        )
