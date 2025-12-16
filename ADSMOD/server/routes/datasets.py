from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from ADSMOD.server.schemas.datasets import DatasetLoadResponse
from ADSMOD.server.utils.constants import (
    DATASETS_LOAD_ENDPOINT,
    DATASETS_ROUTER_PREFIX,
)
from ADSMOD.server.utils.logger import logger
from ADSMOD.server.utils.services.datasets import DatasetService

router = APIRouter(prefix=DATASETS_ROUTER_PREFIX, tags=["load"])
dataset_service = DatasetService()


###############################################################################
# -------------------------------------------------------------------------
@router.post(
    DATASETS_LOAD_ENDPOINT,
    response_model=DatasetLoadResponse,
    status_code=status.HTTP_200_OK,
)
async def load_dataset(file: UploadFile = File(...)) -> DatasetLoadResponse:
    try:
        payload = await file.read()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read uploaded dataset: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to read uploaded dataset.",
        ) from exc

    try:
        dataset_payload, summary = dataset_service.load_from_bytes(
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

    return DatasetLoadResponse(summary=summary, dataset=dataset_payload)
