from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ADSMOD.server.domain.fitting import FittingRequest, NISTFittingDatasetResponse
from ADSMOD.server.domain.jobs import (
    JobCancelResponse,
    JobListResponse,
    JobStartResponse,
    JobStatusResponse,
)
from ADSMOD.server.common.constants import (
    FITTING_JOBS_ENDPOINT,
    FITTING_JOB_STATUS_ENDPOINT,
    FITTING_NIST_DATASET_ENDPOINT,
    FITTING_ROUTER_PREFIX,
    FITTING_RUN_ENDPOINT,
)
from ADSMOD.server.common.utils.logger import logger
from ADSMOD.server.services.fitting import FittingService

router = APIRouter(prefix=FITTING_ROUTER_PREFIX, tags=["fitting"])


###############################################################################
class FittingEndpoint:
    def __init__(self, router: APIRouter, service: FittingService) -> None:
        self.router = router
        self.service = service

    # -------------------------------------------------------------------------
    def start_fitting_job(self, payload: FittingRequest) -> JobStartResponse:
        try:
            return self.service.start_fitting_job(payload)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

    # -------------------------------------------------------------------------
    def get_job_status(self, job_id: str) -> JobStatusResponse:
        try:
            return self.service.get_job_status(job_id)
        except LookupError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc

    # -------------------------------------------------------------------------
    def list_jobs(self) -> JobListResponse:
        return self.service.list_jobs()

    # -------------------------------------------------------------------------
    def cancel_job(self, job_id: str) -> JobCancelResponse:
        try:
            return self.service.cancel_job(job_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

    # -------------------------------------------------------------------------
    def get_nist_dataset_for_fitting(self) -> NISTFittingDatasetResponse:
        try:
            return self.service.get_nist_dataset_for_fitting()
        except ValueError as exc:
            logger.warning("Invalid NIST dataset: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        except Exception as exc:  # noqa: BLE001
            error_type = type(exc).__name__
            error_msg = str(exc).split("\n")[0][:120]
            logger.error("NIST dataset load failed: %s - %s", error_type, error_msg)
            logger.debug("NIST dataset load error details", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to load NIST data.",
            ) from exc

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            FITTING_RUN_ENDPOINT,
            self.start_fitting_job,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            FITTING_NIST_DATASET_ENDPOINT,
            self.get_nist_dataset_for_fitting,
            methods=["GET"],
            response_model=NISTFittingDatasetResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            FITTING_JOBS_ENDPOINT,
            self.list_jobs,
            methods=["GET"],
            response_model=JobListResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            FITTING_JOB_STATUS_ENDPOINT,
            self.get_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            FITTING_JOB_STATUS_ENDPOINT,
            self.cancel_job,
            methods=["DELETE"],
            response_model=JobCancelResponse,
            status_code=status.HTTP_200_OK,
        )


###############################################################################
fitting_endpoint = FittingEndpoint(
    router=router,
    service=FittingService(),
)
fitting_endpoint.add_routes()

