from __future__ import annotations

from core_service.services.data.datasets import DatasetService
from core_service.services.data.nist_mapper import NISTCanonicalMapper
from core_service.services.data.nist_service import NISTDataService
from core_service.services.fitting import FittingService
from shared.common.settings import get_server_settings
from shared.common.utils.logger import logger
from shared.repositories.database.manager import DatabaseManager
from shared.repositories.datasets import DatasetRepository
from shared.repositories.fitting import FittingRepository
from shared.repositories.materials import MaterialRepository
from shared.repositories.nist import NISTRepository
from shared.services.jobs import JobManager

###############################################################################
class CoreServiceContainer:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.job_manager = JobManager(logger=logger)
        self.database = DatabaseManager(get_server_settings().database)
        self.datasets = DatasetRepository(self.database)
        self.materials = MaterialRepository(self.database)
        self.fitting = FittingRepository(self.database)
        self.dataset_service = DatasetService(repository=self.datasets)
        self.nist_repository = NISTRepository(
            database=self.database,
            datasets=self.datasets,
            materials=self.materials,
        )
        self.nist_service = NISTDataService(
            job_manager=self.job_manager,
            repository=self.nist_repository,
            mapper=NISTCanonicalMapper(),
        )
        self.fitting_service = FittingService(
            datasets=self.datasets,
            results=self.fitting,
            job_manager=self.job_manager,
        )
