from __future__ import annotations

from adsmod_common.config import AdsmodConfig
from adsmod_core.services.data.datasets import DatasetService
from adsmod_core.services.data.nist_mapper import NISTCanonicalMapper
from adsmod_core.services.data.nist_service import NISTDataService
from adsmod_core.services.fitting import FittingService
from adsmod_core.common.utils.logger import logger
from adsmod_core.repositories.database.manager import DatabaseManager
from adsmod_core.repositories.datasets import DatasetRepository
from adsmod_core.repositories.fitting import FittingRepository
from adsmod_core.repositories.materials import MaterialRepository
from adsmod_core.repositories.nist import NISTRepository
from adsmod_core.services.jobs import JobManager
from adsmod_core.persistence.paths import resolve_storage_root


###############################################################################
class CoreServiceContainer:
    # -------------------------------------------------------------------------
    def __init__(self, config: AdsmodConfig) -> None:
        self.config = config
        self.job_manager = JobManager(logger=logger)
        self.database = DatabaseManager(
            config.application.database,
            storage_root=resolve_storage_root(config),
        )
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
            config=config,
            job_manager=self.job_manager,
            repository=self.nist_repository,
            mapper=NISTCanonicalMapper(),
        )
        self.fitting_service = FittingService(
            config=config,
            datasets=self.datasets,
            results=self.fitting,
            job_manager=self.job_manager,
        )
