# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import asyncio

from server.configurations.settings import AdsmodConfig, PublicDataConfig
from server.common.utils.logger import logger
from server.common.path import resolve_storage_root
from server.services.providers import CODProvider, NISTPublicDataProvider, PubChemProvider
from server.services.providers.public_data import RequestBudget
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
from server.services.data.public_data_sync import PublicDataSyncCoordinator
from server.services.fitting import FittingService
from server.services.jobs import JobManager


class CoreServiceContainer:
    def __init__(
        self,
        config: AdsmodConfig,
        *,
        database_manager: DatabaseManager | None = None,
    ) -> None:
        self.config = config
        self.job_manager = JobManager(logger=logger)
        self.database = database_manager or DatabaseManager(
            config.application.database, storage_root=resolve_storage_root(config)
        )
        self.datasets = DatasetRepository(self.database)
        self.materials = MaterialRepository(self.database)
        self.fitting = FittingRepository(self.database)
        self.public_data_repository = PublicDataRepository(self.database)
        self.pubchem_request_budget = RequestBudget()
        self.dataset_service = DatasetService(
            repository=self.datasets,
            allowed_extensions=config.application.datasets.allowed_extensions,
            column_detection_cutoff=config.application.datasets.column_detection_cutoff,
            preview_row_limit=config.application.fitting.preview_row_limit,
        )
        self.nist_repository = NISTRepository(
            database=self.database,
            datasets=self.datasets,
            materials=self.materials,
            public_data=self.public_data_repository,
        )
        self.nist_service = self._build_nist_service(config)
        self.public_data_service = self._build_public_data_service(
            config.application.public_data,
            self.nist_service,
        )
        self.fitting_service = FittingService(
            config=config,
            datasets=self.datasets,
            results=self.fitting,
            job_manager=self.job_manager,
        )
        self.public_data_sync = PublicDataSyncCoordinator(
            repository=self.public_data_repository,
            public_data=self.public_data_service,
            nist=self.nist_service,
            config=config.application.public_data,
        )
        self._retired_public_data_services: list[PublicDataService] = []

    def _build_nist_service(self, config: AdsmodConfig) -> NISTDataService:
        return NISTDataService(
            config=config,
            repository=self.nist_repository,
            mapper=NISTCanonicalMapper(),
        )

    def _build_public_data_service(
        self, config: PublicDataConfig, nist_service: NISTDataService
    ) -> PublicDataService:
        return PublicDataService(
            repository=self.public_data_repository,
            health_cache_ttl_seconds=config.health_cache_ttl_seconds,
            providers=[
                NISTPublicDataProvider(nist_service),
                PubChemProvider(
                    parallel_requests=config.pubchem_parallel_requests,
                    request_timeout_seconds=config.request_timeout_seconds,
                    retry_attempts=config.retry_attempts,
                    retry_backoff_max_seconds=config.retry_backoff_max_seconds,
                    max_response_bytes=config.sync_response_size_limit_bytes,
                    request_budget=self.pubchem_request_budget,
                ),
                CODProvider(
                    request_timeout_seconds=config.request_timeout_seconds,
                    retry_attempts=config.retry_attempts,
                    max_interactive_results=config.cod_max_interactive_results,
                    retry_backoff_max_seconds=config.retry_backoff_max_seconds,
                    max_response_bytes=config.sync_response_size_limit_bytes,
                ),
            ],
        )

    def apply_settings(self, config: AdsmodConfig) -> None:
        """Refresh configuration used by future API requests and jobs."""

        previous = self.config
        if (
            previous.user_settings() == config.user_settings()
            and previous.application.database == config.application.database
        ):
            return
        self.config = config
        self.dataset_service.apply_settings(
            allowed_extensions=config.application.datasets.allowed_extensions,
            column_detection_cutoff=config.application.datasets.column_detection_cutoff,
            preview_row_limit=config.application.fitting.preview_row_limit,
        )
        self.fitting_service.config = config
        self.nist_service = self._build_nist_service(config)
        old_service = self.public_data_service
        self.public_data_service = self._build_public_data_service(
            config.application.public_data,
            self.nist_service,
        )
        self._retired_public_data_services.append(old_service)
        self.public_data_sync.apply_settings(
            config=config.application.public_data,
            public_data=self.public_data_service,
            nist=self.nist_service,
        )

    def shutdown(self) -> None:
        self.job_manager.shutdown()

    async def shutdown_async(self) -> None:
        await self.public_data_sync.close()
        await self.public_data_service.close()
        await asyncio.gather(
            *(service.close() for service in self._retired_public_data_services),
            return_exceptions=True,
        )
        self.shutdown()


__all__ = ["CoreServiceContainer"]
