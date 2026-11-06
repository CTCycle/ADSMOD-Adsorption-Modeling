# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response, status

from server.domain.public_data import (
    ChemicalLookupResponse,
    CODSearchResponse,
    NISTCollectionListResponse,
    NISTEditedExperimentsResponse,
    NISTSearchResponse,
    PublicDataSyncCreateRequest,
    PublicDataSyncFailureListResponse,
    PublicDataSyncRunListResponse,
    PublicDataSyncRunView,
    PublicDataOverviewResponse,
    PublicSourceListResponse,
    StructureMaterialLinkRequest,
    StructurePageResponse,
    StructureRecordView,
)
from server.services.providers import (
    ProviderNotFoundError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from server.repositories.nist import NISTRepository
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
        nist: NISTRepository,
    ) -> None:
        self.router = router
        self.service = service
        self.sync = sync
        self.nist = nist

    # -------------------------------------------------------------------------
    def nist_edited_experiments(self) -> NISTEditedExperimentsResponse:
        return NISTEditedExperimentsResponse(count=self.nist.edited_experiment_count())

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
        try:
            run = await self.sync.enqueue(
                source_key=request.source_key,
                operation=request.operation,
                parameters=request.parameters(),
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
        scope: str | None = Query(
            default=None, pattern="^(active|history)$"
        ),
    ) -> PublicDataSyncRunListResponse:
        runs, total = await self.sync.list(
            limit=limit,
            offset=offset,
            source_key=source_key,
            status_filter=run_status,
            status_scope=scope,
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
    async def resume_sync_run(self, run_id: str) -> PublicDataSyncRunView:
        try:
            run = await self.sync.resume(run_id)
        except LookupError as exc:
            raise self._lookup_error(exc) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(exc)
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(exc)
            ) from exc
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
    async def list_nist_collections(self) -> NISTCollectionListResponse:
        return NISTCollectionListResponse(
            collections=await self.sync.nist_collections()
        )

    # -------------------------------------------------------------------------
    async def search_nist(
        self,
        adsorbate: str | None = Query(default=None, max_length=255),
        adsorbent: str | None = Query(default=None, max_length=255),
        experiment: str | None = Query(default=None, max_length=255),
        temperature_min: float | None = Query(default=None, ge=0),
        temperature_max: float | None = Query(default=None, ge=0),
        limit: int = Query(default=25, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> NISTSearchResponse:
        try:
            result = await self.sync.search_nist(
                adsorbate=adsorbate,
                adsorbent=adsorbent,
                experiment=experiment,
                temperature_min=temperature_min,
                temperature_max=temperature_max,
                limit=limit,
                offset=offset,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        return NISTSearchResponse.model_validate(result)

    # -------------------------------------------------------------------------
    async def lookup_chemicals(
        self, q: str = Query(min_length=1, max_length=512)
    ) -> ChemicalLookupResponse:
        try:
            return await self.service.lookup_pubchem(q)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        except (ProviderNotFoundError, ProviderRateLimitError, ProviderUnavailableError) as exc:
            raise self._provider_error(exc) from exc

    # -------------------------------------------------------------------------
    async def ensure_conformer(self, adsorbate_id: int) -> Response:
        try:
            sdf, _ = await self.service.ensure_conformer(adsorbate_id)
        except LookupError as exc:
            raise self._lookup_error(exc) from exc
        except (ProviderNotFoundError, ProviderRateLimitError, ProviderUnavailableError) as exc:
            raise self._provider_error(exc) from exc
        return Response(content=sdf, media_type="chemical/x-mdl-sdfile")

    # -------------------------------------------------------------------------
    def get_overview(self) -> PublicDataOverviewResponse:
        return self.service.get_overview()

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
    def list_structures(
        self,
        page: int = Query(1, ge=1),
        page_size: int = Query(25, ge=1, le=100),
        q: str | None = Query(default=None, max_length=512),
        source: str | None = Query(default=None, max_length=64),
        linked_only: bool | None = None,
        material_id: int | None = Query(default=None, ge=1),
    ) -> StructurePageResponse:
        return self.service.list_structures(
            page=page,
            page_size=page_size,
            query_text=q,
            source=source,
            linked_only=linked_only,
            material_id=material_id,
        )

    # -------------------------------------------------------------------------
    def get_structure(self, structure_id: int) -> StructureRecordView:
        try:
            return self.service.get_structure(structure_id)
        except LookupError as exc:
            raise self._lookup_error(exc) from exc

    # -------------------------------------------------------------------------
    def set_structure_material(
        self, structure_id: int, request: StructureMaterialLinkRequest
    ) -> StructureRecordView:
        try:
            return self.service.set_structure_material(
                structure_id, request.adsorbent_id
            )
        except LookupError as exc:
            raise self._lookup_error(exc) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(exc)
            ) from exc

    # -------------------------------------------------------------------------
    def download_structure_file(self, structure_id: int) -> Response:
        try:
            file_format, content = self.service.get_structure_content(structure_id)
        except LookupError as exc:
            raise self._lookup_error(exc) from exc
        return Response(
            content=content,
            media_type="chemical/x-cif" if file_format == "cif" else "text/plain",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="structure-{structure_id}.{file_format}"'
                )
            },
        )

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
            "/sources/nist/edited-experiments",
            self.nist_edited_experiments,
            methods=["GET"],
            response_model=NISTEditedExperimentsResponse,
        )
        self.router.add_api_route(
            "/sources/nist/collections",
            self.list_nist_collections,
            methods=["GET"],
            response_model=NISTCollectionListResponse,
        )
        self.router.add_api_route(
            "/sources/nist/search",
            self.search_nist,
            methods=["GET"],
            response_model=NISTSearchResponse,
        )
        self.router.add_api_route(
            "/chemicals/lookup",
            self.lookup_chemicals,
            methods=["GET"],
            response_model=ChemicalLookupResponse,
        )
        self.router.add_api_route(
            "/chemicals/{adsorbate_id}/conformer",
            self.ensure_conformer,
            methods=["POST"],
            response_class=Response,
            responses={200: {"content": {"chemical/x-mdl-sdfile": {}}}},
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
            "/sync-runs/{run_id}/resume",
            self.resume_sync_run,
            methods=["POST"],
            response_model=PublicDataSyncRunView,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/sync-runs/{run_id}/failures",
            self.list_sync_failures,
            methods=["GET"],
            response_model=PublicDataSyncFailureListResponse,
        )
        self.router.add_api_route(
            "/structures/search",
            self.search_cod,
            methods=["GET"],
            response_model=CODSearchResponse,
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
        self.router.add_api_route(
            "/structures/{structure_id}/material",
            self.set_structure_material,
            methods=["PUT"],
            response_model=StructureRecordView,
        )
        self.router.add_api_route(
            "/structures/{structure_id}/file",
            self.download_structure_file,
            methods=["GET"],
            response_class=Response,
        )

###############################################################################
def create_public_data_router(container: CoreServiceContainer) -> APIRouter:
    router = APIRouter(prefix="/public-data", tags=["public-data"])
    PublicDataEndpoint(
        router=router,
        service=container.public_data_service,
        sync=container.public_data_sync,
        nist=container.nist_repository,
    ).add_routes()
    return router


__all__ = ["create_public_data_router"]
