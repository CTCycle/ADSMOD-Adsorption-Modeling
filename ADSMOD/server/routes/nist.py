from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, status

from ADSMOD.server.entities.jobs import (
    JobListResponse,
    JobStartResponse,
    JobStatusResponse,
)
from ADSMOD.server.entities.nist import (
    NISTCategory,
    NISTCategoryFetchRequest,
    NISTCategoryPingResponse,
    NISTCategoryStatusResponse,
    NISTFetchRequest,
    NISTPropertiesRequest,
    NISTStatusResponse,
)
from ADSMOD.server.common.constants import (
    NIST_CATEGORY_ENRICH_ENDPOINT,
    NIST_CATEGORY_FETCH_ENDPOINT,
    NIST_CATEGORY_INDEX_ENDPOINT,
    NIST_CATEGORY_PING_ENDPOINT,
    NIST_CATEGORY_STATUS_ENDPOINT,
    NIST_FETCH_ENDPOINT,
    NIST_JOBS_ENDPOINT,
    NIST_JOB_STATUS_ENDPOINT,
    NIST_PROPERTIES_ENDPOINT,
    NIST_ROUTER_PREFIX,
    NIST_STATUS_ENDPOINT,
)
from ADSMOD.server.common.utils.logger import logger
from ADSMOD.server.services.jobs import job_manager
from ADSMOD.server.services.data.nistads import NISTDataService
from ADSMOD.server.configurations import server_settings

router = APIRouter(prefix=NIST_ROUTER_PREFIX, tags=["nist"])


###############################################################################
class NistEndpoint:
    JOB_TYPE_FETCH = "nist_fetch"
    JOB_TYPE_PROPERTIES = "nist_properties"
    CATEGORY_INDEX_SUFFIX = "index"
    CATEGORY_FETCH_SUFFIX = "fetch"
    CATEGORY_ENRICH_SUFFIX = "enrich"

    def __init__(self, router: APIRouter, service: NISTDataService) -> None:
        self.router = router
        self.service = service

    # -------------------------------------------------------------------------
    @staticmethod
    def build_category_job_type(category: NISTCategory, suffix: str) -> str:
        return f"nist_{category}_{suffix}"

    # -------------------------------------------------------------------------
    def start_background_job(
        self,
        job_type: str,
        runner: Callable[..., dict[str, Any]],
        args: tuple[Any, ...] = (),
        message: str = "Job started.",
    ) -> JobStartResponse:
        if job_manager.is_job_running(job_type):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"A {job_type} job is already running.",
            )

        job_id = str(uuid.uuid4())[:8]
        job_manager.start_job(
            job_type=job_type,
            runner=runner,
            args=args,
            job_id=job_id,
        )
        return JobStartResponse(
            job_id=job_id,
            job_type=job_type,
            status="running",
            message=message,
            poll_interval=server_settings.jobs.polling_interval,
        )

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
    def _run_category_index_sync(
        self, category: NISTCategory, job_id: str | None = None
    ) -> dict:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            if category == "experiments":
                return loop.run_until_complete(self.service.fetch_experiments_index(job_id=job_id))
            if category == "guest":
                return loop.run_until_complete(self.service.fetch_guest_index(job_id=job_id))
            return loop.run_until_complete(self.service.fetch_host_index(job_id=job_id))
        finally:
            loop.close()

    # -------------------------------------------------------------------------
    def _run_category_fetch_sync(
        self, category: NISTCategory, fraction: float, job_id: str | None = None
    ) -> dict:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            if category == "experiments":
                return loop.run_until_complete(
                    self.service.fetch_experiments_records(fraction=fraction, job_id=job_id)
                )
            if category == "guest":
                return loop.run_until_complete(
                    self.service.fetch_guest_records(fraction=fraction, job_id=job_id)
                )
            return loop.run_until_complete(
                self.service.fetch_host_records(fraction=fraction, job_id=job_id)
            )
        finally:
            loop.close()

    # -------------------------------------------------------------------------
    def _run_category_enrich_sync(
        self, category: NISTCategory, job_id: str | None = None
    ) -> dict:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            if category == "guest":
                return loop.run_until_complete(
                    self.service.enrich_guest_properties(job_id=job_id)
                )
            if category == "host":
                return loop.run_until_complete(
                    self.service.enrich_host_properties(job_id=job_id)
                )
            raise ValueError("Enrichment is not supported for experiments.")
        finally:
            loop.close()

    # -------------------------------------------------------------------------
    def start_fetch_job(self, request: NISTFetchRequest) -> JobStartResponse:
        response = self.start_background_job(
            job_type=self.JOB_TYPE_FETCH,
            runner=self._run_fetch_sync,
            args=(
                request.experiments_fraction,
                request.guest_fraction,
                request.host_fraction,
            ),
            message="NIST fetch job started.",
        )
        logger.info("Started NIST fetch job %s", response.job_id)
        return response

    # -------------------------------------------------------------------------
    def start_properties_job(self, request: NISTPropertiesRequest) -> JobStartResponse:
        response = self.start_background_job(
            job_type=self.JOB_TYPE_PROPERTIES,
            runner=self._run_properties_sync,
            args=(request.target,),
            message=f"NIST properties enrichment job started for {request.target}.",
        )
        logger.info(
            "Started NIST properties job %s (target=%s)",
            response.job_id,
            request.target,
        )
        return response

    # -------------------------------------------------------------------------
    async def ping_category_server(self, category: NISTCategory) -> NISTCategoryPingResponse:
        try:
            if category == "experiments":
                result = await self.service.ping_experiments_server()
            elif category == "guest":
                result = await self.service.ping_guest_server()
            else:
                result = await self.service.ping_host_server()
        except Exception as exc:  # noqa: BLE001
            logger.exception("NIST category ping failed (category=%s)", category)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to ping NIST category server.",
            ) from exc

        return NISTCategoryPingResponse(
            status="success",
            category=category,
            server_ok=bool(result.get("server_ok", False)),
            checked_at=str(result.get("checked_at", "")),
        )

    # -------------------------------------------------------------------------
    def start_category_index_job(self, category: NISTCategory) -> JobStartResponse:
        job_type = self.build_category_job_type(category, self.CATEGORY_INDEX_SUFFIX)
        response = self.start_background_job(
            job_type=job_type,
            runner=self._run_category_index_sync,
            args=(category,),
            message=f"NIST {category} index job started.",
        )
        logger.info("Started NIST %s index job %s", category, response.job_id)
        return response

    # -------------------------------------------------------------------------
    def start_category_fetch_job(
        self, category: NISTCategory, request: NISTCategoryFetchRequest
    ) -> JobStartResponse:
        job_type = self.build_category_job_type(category, self.CATEGORY_FETCH_SUFFIX)
        response = self.start_background_job(
            job_type=job_type,
            runner=self._run_category_fetch_sync,
            args=(category, request.fraction),
            message=f"NIST {category} fetch job started.",
        )
        logger.info("Started NIST %s fetch job %s", category, response.job_id)
        return response

    # -------------------------------------------------------------------------
    def start_category_enrich_job(self, category: NISTCategory) -> JobStartResponse:
        if category == "experiments":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Enrichment is not supported for experiments.",
            )

        job_type = self.build_category_job_type(category, self.CATEGORY_ENRICH_SUFFIX)
        response = self.start_background_job(
            job_type=job_type,
            runner=self._run_category_enrich_sync,
            args=(category,),
            message=f"NIST {category} enrichment job started.",
        )
        logger.info("Started NIST %s enrichment job %s", category, response.job_id)
        return response

    # -------------------------------------------------------------------------
    async def fetch_nist_category_status(self) -> NISTCategoryStatusResponse:
        try:
            categories = await self.service.get_category_status()
        except Exception as exc:  # noqa: BLE001
            logger.exception("NIST category status check failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to load NIST category status.",
            ) from exc

        return NISTCategoryStatusResponse(status="success", categories=categories)

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
        all_jobs = [
            job
            for job in job_manager.list_jobs()
            if str(job.get("job_type", "")).startswith("nist_")
        ]
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
            NIST_CATEGORY_STATUS_ENDPOINT,
            self.fetch_nist_category_status,
            methods=["GET"],
            response_model=NISTCategoryStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            NIST_CATEGORY_PING_ENDPOINT,
            self.ping_category_server,
            methods=["POST"],
            response_model=NISTCategoryPingResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            NIST_CATEGORY_INDEX_ENDPOINT,
            self.start_category_index_job,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            NIST_CATEGORY_FETCH_ENDPOINT,
            self.start_category_fetch_job,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            NIST_CATEGORY_ENRICH_ENDPOINT,
            self.start_category_enrich_job,
            methods=["POST"],
            response_model=JobStartResponse,
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
