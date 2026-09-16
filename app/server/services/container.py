from __future__ import annotations

from server.configurations.settings import AdsmodConfig
from server.common.utils.logger import logger
from server.common.path import resolve_storage_root
from server.services.providers import CODProvider, NISTPublicDataProvider, PubChemProvider
from server.repositories.database.manager import DatabaseManager
from server.repositories.datasets import DatasetRepository
from server.repositories.fitting import FittingRepository
from server.repositories.materials import MaterialRepository
from server.repositories.nist import NISTRepository
from server.repositories.public_data import PublicDataRepository
from server.services.data.datasets import DatasetService
from server.services.data.nist_mapper import NISTCanonicalMapper
from server.services.data.nist_service import NISTDataService
from server.services.data.public_data import PublicDataService
from server.services.fitting import FittingService
from server.services.jobs import JobManager

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
        self.public_data_repository = PublicDataRepository(self.database)
        self.dataset_service = DatasetService(
            repository=self.datasets,
            allowed_extensions=config.application.datasets.allowed_extensions,
        )
        self.nist_repository = NISTRepository(
            database=self.database,
            datasets=self.datasets,
            materials=self.materials,
            public_data=self.public_data_repository,
        )
        self.nist_service = NISTDataService(
            config=config,
            job_manager=self.job_manager,
            repository=self.nist_repository,
            mapper=NISTCanonicalMapper(),
        )
        public_data_config = config.application.public_data
        self.public_data_service = PublicDataService(
            repository=self.public_data_repository,
            providers=[
                NISTPublicDataProvider(self.nist_service),
                PubChemProvider(
                    parallel_requests=public_data_config.pubchem_parallel_requests,
                    request_timeout_seconds=public_data_config.request_timeout_seconds,
                    retry_attempts=public_data_config.retry_attempts,
                ),
                CODProvider(
                    request_timeout_seconds=public_data_config.request_timeout_seconds,
                    retry_attempts=public_data_config.retry_attempts,
                    max_interactive_results=public_data_config.cod_max_interactive_results,
                ),
            ],
        )
        self.fitting_service = FittingService(
            config=config,
            datasets=self.datasets,
            results=self.fitting,
            job_manager=self.job_manager,
        )

    # -------------------------------------------------------------------------
    def shutdown(self) -> None:
        self.job_manager.shutdown()

    # -------------------------------------------------------------------------
    async def shutdown_async(self) -> None:
        await self.public_data_service.close()
        self.shutdown()
