# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from server.domain.public_data import (
    AdsorptionDetailResponse,
    AdsorptionPageResponse,
    CODSearchResponse,
    CODStructureImportRequest,
    ChemicalPageResponse,
    ChemicalRecordView,
    MaterialPageResponse,
    PublicDataSyncCreateRequest,
    PublicDataSyncFailureListResponse,
    PublicDataSyncRunListResponse,
    PublicDataSyncRunView,
    PublicDataOverviewResponse,
    PublicSourceListResponse,
    PubChemResolveRequest,
    StructurePageResponse,
    StructureRecordView,
)
from server.services.providers import (
    ProviderNotFoundError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from server.services.container import CoreServiceContainer
from server.services.data.public_data import PublicDataService
from server.services.data.public_data_sync import PublicDataSyncCoordinator

###############################################################################
class PublicDataEndpoint:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        router: APIRouter,
        service: PublicDataService,
        sync: PublicDataSyncCoordinator,
    ) -> None:
        self.router = router
        self.service = service
        self.sync = sync

    # -------------------------------------------------------------------------
    @staticmethod
    def _lookup_error(exc: LookupError) -> HTTPException:
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    # -------------------------------------------------------------------------
    @staticmethod
    def _provider_error(exc: Exception) -> HTTPException:
        if isinstance(exc, ProviderNotFoundError):
            code = status.HTTP_404_NOT_FOUND
        elif isinstance(exc, ProviderRateLimitError):
            code = status.HTTP_429_TOO_MANY_REQUESTS
        else:
            code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HTTPException(status_code=code, detail=str(exc))

    # -------------------------------------------------------------------------
    def list_sources(self) -> PublicSourceListResponse:
        return self.service.list_sources()

    # -------------------------------------------------------------------------
    async def start_source_health_check(self, source_key: str) -> PublicDataSyncRunView:
        try:
            run = await self.sync.enqueue(
                source_key=source_key,
                operation="health_check",
                parameters={},
            )
        except LookupError as exc:
            raise self._lookup_error(exc) from exc
        return PublicDataSyncRunView.model_validate(run)

    # -------------------------------------------------------------------------
    async def create_sync_run(
        self, request: PublicDataSyncCreateRequest
    ) -> PublicDataSyncRunView:
        parameters = request.model_dump(
            exclude={"source_key", "operation"}, exclude_none=True
        )
        try:
            run = await self.sync.enqueue(
                source_key=request.source_key,
                operation=request.operation,
                parameters=parameters,
            )
        except LookupError as exc:
            raise self._lookup_error(exc) from exc
        return PublicDataSyncRunView.model_validate(run)

    # -------------------------------------------------------------------------
    async def list_sync_runs(
        self,
        limit: int = Query(default=50, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
        source_key: str | None = Query(default=None, max_length=64),
        run_status: str | None = Query(
            default=None,
            alias="status",
            pattern="^(queued|running|completed|partially_completed|failed|cancelled|interrupted)$",
        ),
    ) -> PublicDataSyncRunListResponse:
        runs, total = await self.sync.list(
            limit=limit,
            offset=offset,
            source_key=source_key,
            status_filter=run_status,
        )
        return PublicDataSyncRunListResponse(
            runs=[PublicDataSyncRunView.model_validate(run) for run in runs],
            total=total,
        )

    # -------------------------------------------------------------------------
    async def get_sync_run(self, run_id: str) -> PublicDataSyncRunView:
        try:
            run = await self.sync.get(run_id)
        except LookupError as exc:
            raise self._lookup_error(exc) from exc
        return PublicDataSyncRunView.model_validate(run)

    # -------------------------------------------------------------------------
    async def cancel_sync_run(self, run_id: str) -> PublicDataSyncRunView:
        try:
            run = await self.sync.cancel(run_id)
        except LookupError as exc:
            raise self._lookup_error(exc) from exc
        return PublicDataSyncRunView.model_validate(run)

    # -------------------------------------------------------------------------
    async def list_sync_failures(
        self,
        run_id: str,
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> PublicDataSyncFailureListResponse:
        try:
            failures, total = await self.sync.failures(
                run_id, limit=limit, offset=offset
            )
        except LookupError as exc:
            raise self._lookup_error(exc) from exc
        return PublicDataSyncFailureListResponse(failures=failures, total=total)

    # -------------------------------------------------------------------------
    def get_overview(self) -> PublicDataOverviewResponse:
        return self.service.get_overview()

    # -------------------------------------------------------------------------
    def list_adsorption(
        self,
        page: int = Query(1, ge=1),
        page_size: int = Query(25, ge=1, le=100),
        source: str | None = None,
        material: str | None = None,
        adsorbate: str | None = None,
        temperature_min_k: float | None = Query(default=None, gt=0),
        temperature_max_k: float | None = Query(default=None, gt=0),
    ) -> AdsorptionPageResponse:
        if (
            temperature_min_k is not None
            and temperature_max_k is not None
            and temperature_min_k > temperature_max_k
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="temperature_min_k cannot exceed temperature_max_k.",
            )
        return self.service.list_adsorption(
            page=page,
            page_size=page_size,
            source=source,
            material=material,
            adsorbate=adsorbate,
            temperature_min_k=temperature_min_k,
            temperature_max_k=temperature_max_k,
        )

    # -------------------------------------------------------------------------
    def get_adsorption(self, isotherm_id: int) -> AdsorptionDetailResponse:
        try:
            return self.service.get_adsorption(isotherm_id)
        except LookupError as exc:
            raise self._lookup_error(exc) from exc

    # -------------------------------------------------------------------------
    def list_materials(
        self,
        page: int = Query(1, ge=1),
        page_size: int = Query(25, ge=1, le=100),
        q: str | None = Query(default=None, max_length=512),
        formula: str | None = Query(default=None, max_length=255),
        source: str | None = Query(default=None, max_length=64),
        has_structure: bool | None = None,
    ) -> MaterialPageResponse:
        return self.service.list_materials(
            page=page,
            page_size=page_size,
            query_text=q,
            formula=formula,
            source=source,
            has_structure=has_structure,
        )

    # -------------------------------------------------------------------------
    def list_chemicals(
        self,
        page: int = Query(1, ge=1),
        page_size: int = Query(25, ge=1, le=100),
        q: str | None = Query(default=None, max_length=512),
        formula: str | None = Query(default=None, max_length=255),
        source: str | None = Query(default=None, max_length=64),
        molecular_weight_min: float | None = Query(default=None, ge=0),
        molecular_weight_max: float | None = Query(default=None, ge=0),
    ) -> ChemicalPageResponse:
        if (
            molecular_weight_min is not None
            and molecular_weight_max is not None
            and molecular_weight_min > molecular_weight_max
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="molecular_weight_min cannot exceed molecular_weight_max.",
            )
        return self.service.list_chemicals(
            page=page,
            page_size=page_size,
            query_text=q,
            formula=formula,
            source=source,
            molecular_weight_min=molecular_weight_min,
            molecular_weight_max=molecular_weight_max,
        )

    # -------------------------------------------------------------------------
    def get_chemical(self, adsorbate_id: int) -> ChemicalRecordView:
        try:
            return self.service.get_chemical(adsorbate_id)
        except LookupError as exc:
            raise self._lookup_error(exc) from exc

    # -------------------------------------------------------------------------
    async def resolve_pubchem(self, request: PubChemResolveRequest) -> ChemicalRecordView:
        try:
            return await self.service.resolve_pubchem(request.query.strip())
        except (ProviderNotFoundError, ProviderRateLimitError, ProviderUnavailableError) as exc:
            raise self._provider_error(exc) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc

    # -------------------------------------------------------------------------
    async def search_cod(
        self,
        q: str | None = Query(default=None, max_length=512),
        formula: str | None = Query(default=None, max_length=255),
        cod_id: str | None = Query(default=None, pattern=r"^\d{4,12}$"),
    ) -> CODSearchResponse:
        try:
            return await self.service.search_cod(text=q, formula=formula, cod_id=cod_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except (ProviderNotFoundError, ProviderRateLimitError, ProviderUnavailableError) as exc:
            raise self._provider_error(exc) from exc

    # -------------------------------------------------------------------------
    async def import_cod(self, request: CODStructureImportRequest) -> StructureRecordView:
        try:
            return await self.service.import_cod(request)
        except LookupError as exc:
            raise self._lookup_error(exc) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except (ProviderNotFoundError, ProviderRateLimitError, ProviderUnavailableError) as exc:
            raise self._provider_error(exc) from exc

    # -------------------------------------------------------------------------
    def list_structures(
        self,
        page: int = Query(1, ge=1),
        page_size: int = Query(25, ge=1, le=100),
        q: str | None = Query(default=None, max_length=512),
        source: str | None = Query(default=None, max_length=64),
        linked_only: bool | None = None,
    ) -> StructurePageResponse:
        return self.service.list_structures(
            page=page,
            page_size=page_size,
            query_text=q,
            source=source,
            linked_only=linked_only,
        )

    # -------------------------------------------------------------------------
    def get_structure(self, structure_id: int) -> StructureRecordView:
        try:
            return self.service.get_structure(structure_id)
        except LookupError as exc:
            raise self._lookup_error(exc) from exc

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "/overview",
            self.get_overview,
            methods=["GET"],
            response_model=PublicDataOverviewResponse,
        )
        self.router.add_api_route(
            "/sources",
            self.list_sources,
            methods=["GET"],
            response_model=PublicSourceListResponse,
        )
        self.router.add_api_route(
            "/sources/{source_key}/health-check",
            self.start_source_health_check,
            methods=["POST"],
            response_model=PublicDataSyncRunView,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/sync-runs",
            self.list_sync_runs,
            methods=["GET"],
            response_model=PublicDataSyncRunListResponse,
        )
        self.router.add_api_route(
            "/sync-runs",
            self.create_sync_run,
            methods=["POST"],
            response_model=PublicDataSyncRunView,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/sync-runs/{run_id}",
            self.get_sync_run,
            methods=["GET"],
            response_model=PublicDataSyncRunView,
        )
        self.router.add_api_route(
            "/sync-runs/{run_id}/cancel",
            self.cancel_sync_run,
            methods=["POST"],
            response_model=PublicDataSyncRunView,
        )
        self.router.add_api_route(
            "/sync-runs/{run_id}",
            self.cancel_sync_run,
            methods=["DELETE"],
            response_model=PublicDataSyncRunView,
        )
        self.router.add_api_route(
            "/sync-runs/{run_id}/failures",
            self.list_sync_failures,
            methods=["GET"],
            response_model=PublicDataSyncFailureListResponse,
        )
        self.router.add_api_route(
            "/adsorption",
            self.list_adsorption,
            methods=["GET"],
            response_model=AdsorptionPageResponse,
        )
        self.router.add_api_route(
            "/adsorption/{isotherm_id}",
            self.get_adsorption,
            methods=["GET"],
            response_model=AdsorptionDetailResponse,
        )
        self.router.add_api_route(
            "/materials",
            self.list_materials,
            methods=["GET"],
            response_model=MaterialPageResponse,
        )
        self.router.add_api_route(
            "/chemicals",
            self.list_chemicals,
            methods=["GET"],
            response_model=ChemicalPageResponse,
        )
        self.router.add_api_route(
            "/chemicals/resolve",
            self.resolve_pubchem,
            methods=["POST"],
            response_model=ChemicalRecordView,
        )
        self.router.add_api_route(
            "/chemicals/{adsorbate_id}",
            self.get_chemical,
            methods=["GET"],
            response_model=ChemicalRecordView,
        )
        self.router.add_api_route(
            "/structures/search",
            self.search_cod,
            methods=["GET"],
            response_model=CODSearchResponse,
        )
        self.router.add_api_route(
            "/structures/import",
            self.import_cod,
            methods=["POST"],
            response_model=StructureRecordView,
        )
        self.router.add_api_route(
            "/structures",
            self.list_structures,
            methods=["GET"],
            response_model=StructurePageResponse,
        )
        self.router.add_api_route(
            "/structures/{structure_id}",
            self.get_structure,
            methods=["GET"],
            response_model=StructureRecordView,
        )

###############################################################################
def create_public_data_router(container: CoreServiceContainer) -> APIRouter:
    router = APIRouter(prefix="/public-data", tags=["public-data"])
    PublicDataEndpoint(
        router=router,
        service=container.public_data_service,
        sync=container.public_data_sync,
    ).add_routes()
    return router


__all__ = ["create_public_data_router"]
