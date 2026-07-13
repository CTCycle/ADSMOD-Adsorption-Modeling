from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Path, Query, UploadFile, status

from core_service.domain.datasets import (
    DatasetListResponse,
    DatasetMetadata,
    DatasetMutationResponse,
    DatasetRenameRequest,
    DatasetRowsMutationRequest,
    DatasetRowsPage,
    DatasetUploadResponse,
)
from core_service.services.container import CoreServiceContainer
from core_service.services.data.datasets import DatasetService
from shared.common.constants import DATASETS_ROUTER_PREFIX


###############################################################################
class DatasetEndpoint:

    # -------------------------------------------------------------------------
    def __init__(self, router: APIRouter, service: DatasetService) -> None:
        self.router = router
        self.service = service

    # -------------------------------------------------------------------------
    @staticmethod
    def _bad_request(exc: ValueError) -> HTTPException:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    # -------------------------------------------------------------------------
    async def upload_dataset(self, file: UploadFile = File(...)) -> DatasetUploadResponse:
        try:
            payload = await self.service.read_upload_bytes(file)
            _, dataset, summary = self.service.upload_dataset(payload, file.filename)
            return DatasetUploadResponse(dataset=dataset, summary=summary)
        except ValueError as exc:
            raise self._bad_request(exc) from exc

    # -------------------------------------------------------------------------
    def list_datasets(self) -> DatasetListResponse:
        return DatasetListResponse(datasets=self.service.list_uploaded_datasets())

    # -------------------------------------------------------------------------
    def delete_dataset(self, dataset_name: str = Path(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_. -]+$")) -> None:
        try:
            self.service.delete_dataset(dataset_name)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    # -------------------------------------------------------------------------
    def rename_dataset(self, request: DatasetRenameRequest, dataset_name: str = Path(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_. -]+$")) -> DatasetMutationResponse:
        try:
            return DatasetMutationResponse(dataset=self.service.rename_dataset(dataset_name, request.new_name))
        except ValueError as exc:
            raise self._bad_request(exc) from exc

    # -------------------------------------------------------------------------
    def get_metadata(self, dataset_name: str = Path(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_. -]+$")) -> DatasetMetadata:
        try:
            return DatasetMetadata(**self.service.get_dataset_metadata(dataset_name))
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    # -------------------------------------------------------------------------
    def update_metadata(self, request: DatasetMetadata, dataset_name: str = Path(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_. -]+$")) -> DatasetMutationResponse:
        try:
            return DatasetMutationResponse(dataset=self.service.update_dataset_metadata(dataset_name, request.tags, request.description))
        except ValueError as exc:
            raise self._bad_request(exc) from exc

    # -------------------------------------------------------------------------
    def get_rows(self, dataset_name: str = Path(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_. -]+$"), offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=500)) -> DatasetRowsPage:
        try:
            columns, rows, total = self.service.get_dataset_rows(dataset_name, offset, limit)
            return DatasetRowsPage(dataset_name=dataset_name, columns=columns, rows=rows, offset=offset, limit=limit, total_rows=total)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    # -------------------------------------------------------------------------
    def mutate_rows(self, request: DatasetRowsMutationRequest, dataset_name: str = Path(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_. -]+$")) -> DatasetMutationResponse:
        try:
            return DatasetMutationResponse(dataset=self.service.mutate_dataset_rows(dataset_name, [item.model_dump() for item in request.operations]))
        except ValueError as exc:
            raise self._bad_request(exc) from exc

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route("", self.upload_dataset, methods=["POST"], response_model=DatasetUploadResponse, status_code=status.HTTP_201_CREATED)
        self.router.add_api_route("", self.list_datasets, methods=["GET"], response_model=DatasetListResponse)
        self.router.add_api_route("/by-name/{dataset_name}", self.delete_dataset, methods=["DELETE"], status_code=status.HTTP_204_NO_CONTENT)
        self.router.add_api_route("/by-name/{dataset_name}/rename", self.rename_dataset, methods=["PATCH"], response_model=DatasetMutationResponse)
        self.router.add_api_route("/by-name/{dataset_name}/metadata", self.get_metadata, methods=["GET"], response_model=DatasetMetadata)
        self.router.add_api_route("/by-name/{dataset_name}/metadata", self.update_metadata, methods=["PATCH"], response_model=DatasetMutationResponse)
        self.router.add_api_route("/by-name/{dataset_name}/rows", self.get_rows, methods=["GET"], response_model=DatasetRowsPage)
        self.router.add_api_route("/by-name/{dataset_name}/rows", self.mutate_rows, methods=["PATCH"], response_model=DatasetMutationResponse)


###############################################################################
def create_dataset_router(container: CoreServiceContainer) -> APIRouter:
    router = APIRouter(prefix=DATASETS_ROUTER_PREFIX, tags=["datasets"])
    DatasetEndpoint(router=router, service=container.dataset_service).add_routes()
    return router