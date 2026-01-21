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


###############################################################################
class NistEndpoint:
    """Endpoint for NIST data collection and enrichment operations."""

    def __init__(self, router: APIRouter, service: NISTDataService) -> None:
        self.router = router
        self.service = service

    # -------------------------------------------------------------------------
    async def fetch_nist_data(self, request: NISTFetchRequest) -> NISTFetchResponse:
        """Fetch NIST adsorption data and store in database."""
        try:
            result = await self.service.fetch_and_store(
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

    # -------------------------------------------------------------------------
    async def fetch_nist_properties(
        self, request: NISTPropertiesRequest
    ) -> NISTPropertiesResponse:
        """Enrich stored materials with PubChem properties."""
        try:
            result = await self.service.enrich_properties(target=request.target)
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

    # -------------------------------------------------------------------------
    async def fetch_nist_status(self) -> NISTStatusResponse:
        """Get availability and row counts for NIST data tables."""
        try:
            result = await self.service.get_status()
        except Exception as exc:  # noqa: BLE001
            logger.exception("NIST status check failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to load NIST status.",
            ) from exc

        return NISTStatusResponse(**result)

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        """Register all NIST-related routes with the router."""
        self.router.add_api_route(
            NIST_FETCH_ENDPOINT,
            self.fetch_nist_data,
            methods=["POST"],
            response_model=NISTFetchResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            NIST_PROPERTIES_ENDPOINT,
            self.fetch_nist_properties,
            methods=["POST"],
            response_model=NISTPropertiesResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            NIST_STATUS_ENDPOINT,
            self.fetch_nist_status,
            methods=["GET"],
            response_model=NISTStatusResponse,
            status_code=status.HTTP_200_OK,
        )


###############################################################################
nist_service = NISTDataService()
nist_endpoint = NistEndpoint(router=router, service=nist_service)
nist_endpoint.add_routes()
