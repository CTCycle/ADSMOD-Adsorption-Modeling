from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, HTTPException, status

from ADSMOD.server.schemas.jobs import (
    JobListResponse,
    JobStartResponse,
    JobStatusResponse,
)
from ADSMOD.server.schemas.nist import (
    NISTFetchRequest,
    NISTPropertiesRequest,
    NISTStatusResponse,
)
from ADSMOD.server.utils.constants import (
    NIST_FETCH_ENDPOINT,
    NIST_JOBS_ENDPOINT,
    NIST_JOB_STATUS_ENDPOINT,
    NIST_PROPERTIES_ENDPOINT,
    NIST_ROUTER_PREFIX,
    NIST_STATUS_ENDPOINT,
)
from ADSMOD.server.utils.logger import logger
from ADSMOD.server.services.jobs import job_manager
from ADSMOD.server.services.data.nistads import NISTDataService
from ADSMOD.server.configurations import server_settings

router = APIRouter(prefix=NIST_ROUTER_PREFIX, tags=["nist"])


###############################################################################
class NistEndpoint:
    JOB_TYPE_FETCH = "nist_fetch"
    JOB_TYPE_PROPERTIES = "nist_properties"

    def __init__(self, router: APIRouter, service: NISTDataService) -> None:
        self.router = router
        self.service = service

    # -------------------------------------------------------------------------
    def _run_fetch_sync(
        self,
        experiments_fraction: float,
        guest_fraction: float,
        host_fraction: float,
        job_id: str | None = None,
    ) -> dict:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.service.fetch_and_store(
                    experiments_fraction=experiments_fraction,
                    guest_fraction=guest_fraction,
                    host_fraction=host_fraction,
                    job_id=job_id,
                )
            )
        finally:
            loop.close()

    # -------------------------------------------------------------------------
    def _run_properties_sync(self, target: str, job_id: str | None = None) -> dict:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.service.enrich_properties(target=target, job_id=job_id)
            )
        finally:
            loop.close()

    # -------------------------------------------------------------------------
    def start_fetch_job(self, request: NISTFetchRequest) -> JobStartResponse:
        if job_manager.is_job_running(self.JOB_TYPE_FETCH):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A NIST fetch job is already running.",
            )
        if job_manager.is_job_running(self.JOB_TYPE_PROPERTIES):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A NIST properties job is running. Wait for it to complete.",
            )

        job_id = str(uuid.uuid4())[:8]
        job_manager.start_job(
            job_type=self.JOB_TYPE_FETCH,
            runner=self._run_fetch_sync,
            args=(
                request.experiments_fraction,
                request.guest_fraction,
                request.host_fraction,
            ),
            job_id=job_id,
        )
        logger.info("Started NIST fetch job %s", job_id)
        return JobStartResponse(
            job_id=job_id,
            job_type=self.JOB_TYPE_FETCH,
            status="running",
            message="NIST fetch job started.",
            poll_interval=server_settings.jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def start_properties_job(self, request: NISTPropertiesRequest) -> JobStartResponse:
        if job_manager.is_job_running(self.JOB_TYPE_PROPERTIES):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A NIST properties job is already running.",
            )
        if job_manager.is_job_running(self.JOB_TYPE_FETCH):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A NIST fetch job is running. Wait for it to complete.",
            )

        job_id = str(uuid.uuid4())[:8]
        job_manager.start_job(
            job_type=self.JOB_TYPE_PROPERTIES,
            runner=self._run_properties_sync,
            args=(request.target,),
            job_id=job_id,
        )
        logger.info(
            "Started NIST properties job %s (target=%s)", job_id, request.target
        )
        return JobStartResponse(
            job_id=job_id,
            job_type=self.JOB_TYPE_PROPERTIES,
            status="running",
            message=f"NIST properties enrichment job started for {request.target}.",
            poll_interval=server_settings.jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def get_job_status(self, job_id: str) -> JobStatusResponse:
        job_status = job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found.",
            )
        return JobStatusResponse(
            job_id=job_status["job_id"],
            job_type=job_status["job_type"],
            status=job_status["status"],
            progress=job_status["progress"],
            result=job_status["result"],
            error=job_status["error"],
            poll_interval=server_settings.jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def list_jobs(self) -> JobListResponse:
        fetch_jobs = job_manager.list_jobs(self.JOB_TYPE_FETCH)
        properties_jobs = job_manager.list_jobs(self.JOB_TYPE_PROPERTIES)
        all_jobs = fetch_jobs + properties_jobs
        return JobListResponse(
            jobs=[
                JobStatusResponse(
                    job_id=j["job_id"],
                    job_type=j["job_type"],
                    status=j["status"],
                    progress=j["progress"],
                    result=j["result"],
                    error=j["error"],
                    poll_interval=server_settings.jobs.polling_interval,
                )
                for j in all_jobs
            ]
        )

    # -------------------------------------------------------------------------
    def cancel_job(self, job_id: str) -> dict:
        success = job_manager.cancel_job(job_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job {job_id} cannot be cancelled (not found or already completed).",
            )
        return {"status": "cancelled", "job_id": job_id}

    # -------------------------------------------------------------------------
    async def fetch_nist_status(self) -> NISTStatusResponse:
        try:
            result = await self.service.get_status()
        except Exception as exc:  # noqa: BLE001
            logger.exception("NIST status check failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to load NIST status.",
            ) from exc

        return NISTStatusResponse(
            status="success",
            data_available=bool(result.get("data_available", False)),
            single_component_rows=int(result.get("single_component_rows", 0)),
            binary_mixture_rows=int(result.get("binary_mixture_rows", 0)),
            guest_rows=int(result.get("guest_rows", 0)),
            host_rows=int(result.get("host_rows", 0)),
        )

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            NIST_FETCH_ENDPOINT,
            self.start_fetch_job,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            NIST_PROPERTIES_ENDPOINT,
            self.start_properties_job,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            NIST_STATUS_ENDPOINT,
            self.fetch_nist_status,
            methods=["GET"],
            response_model=NISTStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            NIST_JOBS_ENDPOINT,
            self.list_jobs,
            methods=["GET"],
            response_model=JobListResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            NIST_JOB_STATUS_ENDPOINT,
            self.get_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            NIST_JOB_STATUS_ENDPOINT,
            self.cancel_job,
            methods=["DELETE"],
            status_code=status.HTTP_200_OK,
        )


###############################################################################
nist_service = NISTDataService()
nist_endpoint = NistEndpoint(router=router, service=nist_service)
nist_endpoint.add_routes()
