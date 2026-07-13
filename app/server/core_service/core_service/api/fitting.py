from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from core_service.domain.fitting import FittingRequest
from core_service.services.container import CoreServiceContainer
from core_service.services.fitting import FittingService
from shared.common.constants import FITTING_JOBS_ENDPOINT, FITTING_JOB_STATUS_ENDPOINT, FITTING_ROUTER_PREFIX, FITTING_RUN_ENDPOINT
from shared.models.jobs import JobCancelResponse, JobListResponse, JobStartResponse, JobStatusResponse

###############################################################################
class FittingEndpoint:

    # -------------------------------------------------------------------------
    def __init__(self, router: APIRouter, service: FittingService) -> None:
        self.router = router
        self.service = service

    # -------------------------------------------------------------------------
    def start_fitting_job(self, payload: FittingRequest) -> JobStartResponse:
        try:
            return self.service.start_fitting_job(payload)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    # -------------------------------------------------------------------------
    def get_job_status(self, job_id: str) -> JobStatusResponse:
        try:
            return self.service.get_job_status(job_id)
        except LookupError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    # -------------------------------------------------------------------------
    def list_jobs(self) -> JobListResponse:
        return self.service.list_jobs()

    # -------------------------------------------------------------------------
    def cancel_job(self, job_id: str) -> JobCancelResponse:
        try:
            return self.service.cancel_job(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(FITTING_RUN_ENDPOINT, self.start_fitting_job, methods=["POST"], response_model=JobStartResponse)
        self.router.add_api_route(FITTING_JOBS_ENDPOINT, self.list_jobs, methods=["GET"], response_model=JobListResponse)
        self.router.add_api_route(FITTING_JOB_STATUS_ENDPOINT, self.get_job_status, methods=["GET"], response_model=JobStatusResponse)
        self.router.add_api_route(FITTING_JOB_STATUS_ENDPOINT, self.cancel_job, methods=["DELETE"], response_model=JobCancelResponse)

###############################################################################
def create_fitting_router(container: CoreServiceContainer) -> APIRouter:
    router = APIRouter(prefix=FITTING_ROUTER_PREFIX, tags=["fitting"])
    FittingEndpoint(router=router, service=container.fitting_service).add_routes()
    return router