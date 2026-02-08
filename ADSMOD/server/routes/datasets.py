from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from ADSMOD.server.schemas.datasets import DatasetLoadResponse, DatasetNamesResponse
from ADSMOD.server.common.constants import (
    DATASETS_FETCH_ENDPOINT,
    DATASETS_LOAD_ENDPOINT,
    DATASETS_NAMES_ENDPOINT,
    DATASETS_ROUTER_PREFIX,
)
from ADSMOD.server.common.utils.logger import logger
from ADSMOD.server.services.data.datasets import DatasetService

router = APIRouter(prefix=DATASETS_ROUTER_PREFIX, tags=["load"])


###############################################################################
class DatasetEndpoint:
    def __init__(self, router: APIRouter, service: DatasetService) -> None:
        self.router = router
        self.service = service

    # -------------------------------------------------------------------------
    async def load_dataset(self, file: UploadFile = File(...)) -> DatasetLoadResponse:
        try:
            payload = await file.read()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read uploaded dataset: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to read uploaded dataset.",
            ) from exc

        try:
            dataset_payload, summary = self.service.load_from_bytes(
                payload, file.filename
            )
        except ValueError as exc:
            logger.warning("Invalid dataset upload: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Dataset processing failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process uploaded dataset.",
            ) from exc

        self.service.save_to_database(dataset_payload)
        return DatasetLoadResponse(summary=summary, dataset=dataset_payload)

    # -------------------------------------------------------------------------
    def get_dataset_names(self) -> DatasetNamesResponse:
        try:
            names = self.service.get_dataset_names()
            return DatasetNamesResponse(names=names)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch dataset names: %s", exc)
            return DatasetNamesResponse(names=[])

    # -------------------------------------------------------------------------
    def get_dataset_by_name(self, dataset_name: str) -> DatasetLoadResponse:
        try:
            dataset_payload, summary = self.service.load_from_database(dataset_name)
            return DatasetLoadResponse(summary=summary, dataset=dataset_payload)
        except ValueError as exc:
            logger.warning("Failed to load dataset '%s': %s", dataset_name, exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Dataset lookup failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to load dataset.",
            ) from exc

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            DATASETS_LOAD_ENDPOINT,
            self.load_dataset,
            methods=["POST"],
            response_model=DatasetLoadResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            DATASETS_NAMES_ENDPOINT,
            self.get_dataset_names,
            methods=["GET"],
            response_model=DatasetNamesResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            DATASETS_FETCH_ENDPOINT,
            self.get_dataset_by_name,
            methods=["GET"],
            response_model=DatasetLoadResponse,
            status_code=status.HTTP_200_OK,
        )


###############################################################################
dataset_service = DatasetService()
dataset_endpoint = DatasetEndpoint(router=router, service=dataset_service)
dataset_endpoint.add_routes()
