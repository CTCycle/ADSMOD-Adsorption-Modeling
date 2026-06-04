from __future__ import annotations

from core_service.common.utils.logger import logger
from core_service.services.data.datasets import DatasetService
from core_service.services.data.nist_service import NISTDataService
from core_service.services.fitting import FittingService
from core_service.services.jobs import JobManager


class CoreServiceContainer:
    def __init__(self) -> None:
        self.job_manager = JobManager(logger=logger)
        self.dataset_service = DatasetService()
        self.nist_service = NISTDataService(job_manager=self.job_manager)
        self.fitting_service = FittingService(job_manager=self.job_manager)
