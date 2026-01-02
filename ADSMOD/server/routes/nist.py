from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ADSMOD.server.schemas.nist import (
    NISTFetchRequest,
    NISTFetchResponse,
    NISTPropertiesRequest,
    NISTPropertiesResponse,
    NISTStatusResponse,
)
from ADSMOD.server.utils.constants import (
    NIST_FETCH_ENDPOINT,
    NIST_PROPERTIES_ENDPOINT,
    NIST_ROUTER_PREFIX,
    NIST_STATUS_ENDPOINT,
)
from ADSMOD.server.utils.logger import logger
from ADSMOD.server.utils.services.nistads import NISTDataService

router = APIRouter(prefix=NIST_ROUTER_PREFIX, tags=["nist"])
service = NISTDataService()


###############################################################################
@router.post(
    NIST_FETCH_ENDPOINT,
    response_model=NISTFetchResponse,
    status_code=status.HTTP_200_OK,
)
async def fetch_nist_data(request: NISTFetchRequest) -> NISTFetchResponse:
    try:
        result = await service.fetch_and_store(
            experiments_fraction=request.experiments_fraction,
            guest_fraction=request.guest_fraction,
            host_fraction=request.host_fraction,
        )
    except ValueError as exc:
        logger.warning("NIST fetch validation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("NIST fetch failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch NIST data.",
        ) from exc

    return NISTFetchResponse(**result)


###############################################################################
@router.post(
    NIST_PROPERTIES_ENDPOINT,
    response_model=NISTPropertiesResponse,
    status_code=status.HTTP_200_OK,
)
async def fetch_nist_properties(
    request: NISTPropertiesRequest,
) -> NISTPropertiesResponse:
    try:
        result = await service.enrich_properties(target=request.target)
    except ValueError as exc:
        logger.warning("NIST properties request failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("NIST properties enrichment failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to enrich NIST materials.",
        ) from exc

    return NISTPropertiesResponse(target=request.target, **result)


###############################################################################
@router.get(
    NIST_STATUS_ENDPOINT,
    response_model=NISTStatusResponse,
    status_code=status.HTTP_200_OK,
)
async def fetch_nist_status() -> NISTStatusResponse:
    try:
        result = await service.get_status()
    except Exception as exc:  # noqa: BLE001
        logger.exception("NIST status check failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load NIST status.",
        ) from exc

    return NISTStatusResponse(**result)
