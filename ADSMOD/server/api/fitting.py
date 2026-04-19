from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ADSMOD.server.domain.fitting import FittingRequest, NISTFittingDatasetResponse
from ADSMOD.server.domain.jobs import (
    JobCancelResponse,
    JobListResponse,
    JobStartResponse,
    JobStatusResponse,
)
from ADSMOD.server.configurations import get_server_settings
from ADSMOD.server.common.constants import (
    FITTING_JOBS_ENDPOINT,
    FITTING_JOB_STATUS_ENDPOINT,
    FITTING_NIST_DATASET_ENDPOINT,
    FITTING_ROUTER_PREFIX,
    FITTING_RUN_ENDPOINT,
)
from ADSMOD.server.common.utils.logger import logger
from ADSMOD.server.services.job_responses import JobResponseFactory
from ADSMOD.server.services.modeling.fitting import FittingPipeline
from ADSMOD.server.services.modeling.nist_dataset import FittingNISTDatasetService
from ADSMOD.server.services.jobs import job_manager

router = APIRouter(prefix=FITTING_ROUTER_PREFIX, tags=["fitting"])


###############################################################################
class FittingEndpoint:
    JOB_TYPE = "fitting"

    def __init__(
        self,
        router: APIRouter,
        pipeline: FittingPipeline,
        nist_dataset_service: FittingNISTDatasetService,
    ) -> None:
        self.router = router
        self.pipeline = pipeline
        self.nist_dataset_service = nist_dataset_service

    # -------------------------------------------------------------------------
    def _run_fitting_sync(
        self,
        dataset_dict: dict,
        parameter_bounds_dict: dict,
        max_iterations: int,
        optimization_method: str,
    ) -> dict:
        return self.pipeline.run(
            dataset_dict,
            parameter_bounds_dict,
            max_iterations,
            optimization_method,
        )

    # -------------------------------------------------------------------------
    def start_fitting_job(self, payload: FittingRequest) -> JobStartResponse:
        if job_manager.is_job_running(self.JOB_TYPE):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A fitting job is already running.",
            )

        logger.info(
            "Received fitting request: iterations=%s, method=%s",
            payload.max_iterations,
            payload.optimization_method,
        )

        dataset_dict = payload.dataset.model_dump()
        parameter_bounds_dict = {
            name: config.model_dump()
            for name, config in payload.parameter_bounds.items()
        }

        job_id = job_manager.start_job(
            job_type=self.JOB_TYPE,
            runner=self._run_fitting_sync,
            args=(
                dataset_dict,
                parameter_bounds_dict,
                payload.max_iterations,
                payload.optimization_method,
            ),
        )
        logger.info("Started fitting job %s", job_id)
        return JobResponseFactory.start(
            job_id=job_id,
            job_type=self.JOB_TYPE,
            message="Fitting job started.",
            poll_interval=get_server_settings().jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def get_job_status(self, job_id: str) -> JobStatusResponse:
        job_status = job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found.",
            )
        return JobResponseFactory.status(
            job_status=job_status,
            poll_interval=get_server_settings().jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def list_jobs(self) -> JobListResponse:
        all_jobs = job_manager.list_jobs(self.JOB_TYPE)
        return JobResponseFactory.list(
            job_statuses=all_jobs,
            poll_interval=get_server_settings().jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def cancel_job(self, job_id: str) -> JobCancelResponse:
        success = job_manager.cancel_job(job_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job {job_id} cannot be cancelled (not found or already completed).",
            )
        return JobResponseFactory.cancelled(job_id)

    # -------------------------------------------------------------------------
    def get_nist_dataset_for_fitting(self) -> NISTFittingDatasetResponse:
        try:
            return self.nist_dataset_service.load_for_fitting()
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
pipeline = FittingPipeline()
nist_dataset_service = FittingNISTDatasetService()
fitting_endpoint = FittingEndpoint(
    router=router,
    pipeline=pipeline,
    nist_dataset_service=nist_dataset_service,
)
fitting_endpoint.add_routes()

