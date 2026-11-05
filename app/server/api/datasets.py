# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from typing import Literal

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    UploadFile,
    status,
)

from server.domain.datasets import (
    CustomUnitRequest,
    CustomUnitResponse,
    DatasetImportResponse,
    DatasetImportInspectionResponse,
    DatasetDeleteImpact,
    DatasetDetailResponse,
    DatasetEditErrorDetail,
    DatasetListResponse,
    DatasetMutationResponse,
    DatasetUpdateRequest,
    ExperimentDetailResponse,
    ExperimentPage,
    ImportPreviewResponse,
    ImportValidationResponse,
    ObservationPage,
    SupportedUnitsResponse,
)
from server.services.container import CoreServiceContainer
from server.services.data.dataset_editing import DatasetEditValidationError
from server.services.data.datasets import DatasetService
from server.common.constants import DATASETS_ROUTER_PREFIX, MAX_UPLOAD_SIZE_BYTES

###############################################################################
class DatasetEndpoint:

    # -------------------------------------------------------------------------
    def __init__(self, router: APIRouter, service: DatasetService) -> None:
        self.router = router
        self.service = service

    # -------------------------------------------------------------------------
    @staticmethod
    def bad_request(exc: ValueError) -> HTTPException:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    # -------------------------------------------------------------------------
    @staticmethod
    def not_found(exc: LookupError) -> HTTPException:
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    # -------------------------------------------------------------------------
    @staticmethod
    async def read_upload_bytes(file: UploadFile) -> bytes:
        payload = bytearray()
        try:
            while chunk := await file.read(1024 * 1024):
                payload.extend(chunk)
                if len(payload) > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="Uploaded dataset exceeds the 25 MB limit.",
                    )
        finally:
            await file.close()
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded dataset is empty.",
            )
        return bytes(payload)

    # -------------------------------------------------------------------------
    async def preview_import(
        self, file: UploadFile = File(...)
    ) -> ImportPreviewResponse:
        try:
            payload = await self.read_upload_bytes(file)
            return self.service.preview(payload, file.filename)
        except ValueError as exc:
            raise self.bad_request(exc) from exc

    # -------------------------------------------------------------------------
    async def validate_import(
        self,
        mapping: str = Form(...),
        file: UploadFile = File(...),
    ) -> ImportValidationResponse:
        try:
            payload = await self.read_upload_bytes(file)
            return self.service.validate(
                payload, file.filename, self.service.parse_mapping(mapping)
            )
        except ValueError as exc:
            raise self.bad_request(exc) from exc

    # -------------------------------------------------------------------------
    async def commit_import(
        self,
        mapping: str = Form(...),
        file: UploadFile = File(...),
    ) -> DatasetImportResponse:
        try:
            payload = await self.read_upload_bytes(file)
            return self.service.commit(
                payload, file.filename, self.service.parse_mapping(mapping)
            )
        except ValueError as exc:
            raise self.bad_request(exc) from exc

    # -------------------------------------------------------------------------
    def list_datasets(self) -> DatasetListResponse:
        return self.service.list_datasets()

    # -------------------------------------------------------------------------
    def import_inspection(
        self, dataset_id: int = Path(..., ge=1)
    ) -> DatasetImportInspectionResponse:
        try:
            return self.service.import_inspection(dataset_id)
        except LookupError as exc:
            raise self.not_found(exc) from exc

    # -------------------------------------------------------------------------
    def supported_units(self) -> SupportedUnitsResponse:
        return self.service.supported_units()

    # -------------------------------------------------------------------------
    def save_custom_unit(self, request: CustomUnitRequest) -> CustomUnitResponse:
        try:
            return self.service.save_custom_unit(request)
        except ValueError as exc:
            raise self.bad_request(exc) from exc

    # -------------------------------------------------------------------------
    def delete_custom_unit(self, unit_id: int = Path(..., ge=1)) -> None:
        try:
            self.service.delete_custom_unit(unit_id)
        except LookupError as exc:
            raise self.not_found(exc) from exc

    # -------------------------------------------------------------------------
    def get_dataset(self, dataset_id: int = Path(..., ge=1)) -> DatasetDetailResponse:
        try:
            return self.service.get_dataset(dataset_id)
        except LookupError as exc:
            raise self.not_found(exc) from exc

    # -------------------------------------------------------------------------
    def delete_impact(self, dataset_id: int = Path(..., ge=1)) -> DatasetDeleteImpact:
        try:
            return self.service.delete_impact(dataset_id)
        except LookupError as exc:
            raise self.not_found(exc) from exc

    # -------------------------------------------------------------------------
    def list_experiments(
        self,
        dataset_id: int = Path(..., ge=1),
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=500),
        q: str | None = Query(default=None, max_length=255),
        adsorbent_id: int | None = Query(default=None, ge=1),
        adsorbate_id: int | None = Query(default=None, ge=1),
        temperature_min_k: float | None = Query(default=None, gt=0),
        temperature_max_k: float | None = Query(default=None, gt=0),
        edited_only: bool = False,
    ) -> ExperimentPage:
        if (
            temperature_min_k is not None
            and temperature_max_k is not None
            and temperature_min_k > temperature_max_k
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="temperature_min_k cannot exceed temperature_max_k.",
            )
        try:
            return self.service.list_experiments(
                dataset_id,
                offset=offset,
                limit=limit,
                query_text=q,
                adsorbent_id=adsorbent_id,
                adsorbate_id=adsorbate_id,
                temperature_min_k=temperature_min_k,
                temperature_max_k=temperature_max_k,
                edited_only=edited_only,
            )
        except LookupError as exc:
            raise self.not_found(exc) from exc

    # -------------------------------------------------------------------------
    def experiment_detail(
        self,
        dataset_id: int = Path(..., ge=1),
        experiment_id: int = Path(..., ge=1),
    ) -> ExperimentDetailResponse:
        try:
            return self.service.experiment_detail(dataset_id, experiment_id)
        except LookupError as exc:
            raise self.not_found(exc) from exc

    # -------------------------------------------------------------------------
    def get_observations(
        self,
        dataset_id: int = Path(..., ge=1),
        isotherm_id: int = Path(..., ge=1),
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=500),
        order: Literal["pressure", "sequence"] = "pressure",
    ) -> ObservationPage:
        try:
            return self.service.get_observations(
                dataset_id, isotherm_id, offset, limit, order
            )
        except LookupError as exc:
            raise self.not_found(exc) from exc

    # -------------------------------------------------------------------------
    def update_dataset(
        self,
        request: DatasetUpdateRequest,
        dataset_id: int = Path(..., ge=1),
    ) -> DatasetMutationResponse:
        try:
            return self.service.update(dataset_id, request)
        except DatasetEditValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=DatasetEditErrorDetail(
                    message=str(exc), issues=exc.issues
                ).model_dump(mode="json"),
            ) from exc
        except LookupError as exc:
            raise self.not_found(exc) from exc
        except ValueError as exc:
            raise self.bad_request(exc) from exc

    # -------------------------------------------------------------------------
    def delete_dataset(self, dataset_id: int = Path(..., ge=1)) -> None:
        try:
            self.service.delete(dataset_id)
        except LookupError as exc:
            raise self.not_found(exc) from exc

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "/supported-units",
            self.supported_units,
            methods=["GET"],
            response_model=SupportedUnitsResponse,
        )
        self.router.add_api_route(
            "/units",
            self.save_custom_unit,
            methods=["POST"],
            response_model=CustomUnitResponse,
            status_code=status.HTTP_201_CREATED,
        )
        self.router.add_api_route(
            "/units/{unit_id}",
            self.delete_custom_unit,
            methods=["DELETE"],
            status_code=status.HTTP_204_NO_CONTENT,
        )
        self.router.add_api_route(
            "/import/preview",
            self.preview_import,
            methods=["POST"],
            response_model=ImportPreviewResponse,
        )
        self.router.add_api_route(
            "/import/validate",
            self.validate_import,
            methods=["POST"],
            response_model=ImportValidationResponse,
        )
        self.router.add_api_route(
            "/import/commit",
            self.commit_import,
            methods=["POST"],
            response_model=DatasetImportResponse,
            status_code=status.HTTP_201_CREATED,
        )
        self.router.add_api_route(
            "",
            self.list_datasets,
            methods=["GET"],
            response_model=DatasetListResponse,
        )
        self.router.add_api_route(
            "/{dataset_id}/import",
            self.import_inspection,
            methods=["GET"],
            response_model=DatasetImportInspectionResponse,
        )
        self.router.add_api_route(
            "/{dataset_id}/experiments",
            self.list_experiments,
            methods=["GET"],
            response_model=ExperimentPage,
        )
        self.router.add_api_route(
            "/{dataset_id}/experiments/{experiment_id}",
            self.experiment_detail,
            methods=["GET"],
            response_model=ExperimentDetailResponse,
        )
        self.router.add_api_route(
            "/{dataset_id}/experiments/{isotherm_id}/observations",
            self.get_observations,
            methods=["GET"],
            response_model=ObservationPage,
        )
        self.router.add_api_route(
            "/{dataset_id}/delete-impact",
            self.delete_impact,
            methods=["GET"],
            response_model=DatasetDeleteImpact,
        )
        self.router.add_api_route(
            "/{dataset_id}",
            self.get_dataset,
            methods=["GET"],
            response_model=DatasetDetailResponse,
        )
        self.router.add_api_route(
            "/{dataset_id}",
            self.update_dataset,
            methods=["PATCH"],
            response_model=DatasetMutationResponse,
            responses={422: {"model": DatasetEditErrorDetail}},
        )
        self.router.add_api_route(
            "/{dataset_id}",
            self.delete_dataset,
            methods=["DELETE"],
            status_code=status.HTTP_204_NO_CONTENT,
        )

###############################################################################
def create_dataset_router(container: CoreServiceContainer) -> APIRouter:
    router = APIRouter(prefix=DATASETS_ROUTER_PREFIX, tags=["datasets"])
    DatasetEndpoint(router, container.dataset_service).add_routes()
    return router
