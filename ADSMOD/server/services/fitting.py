from __future__ import annotations

from ADSMOD.server.common.utils.logger import logger
from ADSMOD.server.configurations import get_server_settings
from ADSMOD.server.domain.fitting import FittingRequest, NISTFittingDatasetResponse
from ADSMOD.server.domain.jobs import (
    JobCancelResponse,
    JobListResponse,
    JobStartResponse,
    JobStatusResponse,
)
from ADSMOD.server.services.job_responses import JobResponseFactory
from ADSMOD.server.services.jobs import job_manager
from ADSMOD.server.services.modeling.fitting import FittingPipeline
from ADSMOD.server.services.modeling.nist_dataset import FittingNISTDatasetService


###############################################################################
class FittingService:
    JOB_TYPE = "fitting"

    def __init__(
        self,
        pipeline: FittingPipeline | None = None,
        nist_dataset_service: FittingNISTDatasetService | None = None,
    ) -> None:
        self.pipeline = pipeline or FittingPipeline()
        self.nist_dataset_service = nist_dataset_service or FittingNISTDatasetService()

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
            raise ValueError("A fitting job is already running.")

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
            raise LookupError(f"Job {job_id} not found.")
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
            raise ValueError(
                f"Job {job_id} cannot be cancelled (not found or already completed)."
            )
        return JobResponseFactory.cancelled(job_id)

    # -------------------------------------------------------------------------
    def get_nist_dataset_for_fitting(self) -> NISTFittingDatasetResponse:
        return self.nist_dataset_service.load_for_fitting()
