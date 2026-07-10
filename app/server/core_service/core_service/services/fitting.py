from __future__ import annotations

from typing import Any

from shared.common.utils.logger import logger
from core_service.configurations import get_server_settings
from core_service.domain.fitting import FittingRequest
from core_service.services.modeling.fitting import FittingPipeline
from core_service.services.modeling.nist_dataset import FittingNISTDatasetService
from shared.models.jobs import (
    JobCancelResponse,
    JobListResponse,
    JobStartResponse,
    JobStatusResponse,
)
from shared.services.job_responses import JobResponseFactory
from shared.services.jobs import JobManager
from shared.repositories.serialization.data import DataSerializer

###############################################################################
class FittingService:
    JOB_TYPE = "fitting"

    # -------------------------------------------------------------------------
    def __init__(
        self,
        job_manager: JobManager | None = None,
        pipeline: FittingPipeline | None = None,
        nist_dataset_service: FittingNISTDatasetService | None = None,
    ) -> None:
        self.job_manager = job_manager or JobManager(logger=logger)
        self.pipeline = pipeline or FittingPipeline()
        self.nist_dataset_service = nist_dataset_service or FittingNISTDatasetService()

    # -------------------------------------------------------------------------
    def _run_fitting_sync(
        self,
        dataset_dict: dict[str, Any],
        parameter_bounds_dict: dict[str, Any],
        max_iterations: int,
        optimization_method: str,
    ) -> dict[str, Any]:
        return self.pipeline.run(
            dataset_dict,
            parameter_bounds_dict,
            max_iterations,
            optimization_method,
        )

    # -------------------------------------------------------------------------
    def resolve_dataset(self, source: str, dataset_name: str | None) -> dict[str, Any]:
        if source == "nist":
            return self.nist_dataset_service.load_for_fitting().dataset.model_dump()
        if not dataset_name:
            raise ValueError("An uploaded dataset name is required.")
        frame, _ = DataSerializer().get_uploaded_dataset_rows(dataset_name, 0, 1_000_000)
        frame = frame.drop(columns=["row_id", "name"], errors="ignore")
        return {"dataset_name": dataset_name, "columns": list(frame.columns), "records": frame.where(frame.notna(), None).to_dict(orient="records")}
    # -------------------------------------------------------------------------
    def start_fitting_job(self, payload: FittingRequest) -> JobStartResponse:
        if self.job_manager.is_job_running(self.JOB_TYPE):
            raise ValueError("A fitting job is already running.")

        logger.info(
            "Received fitting request: iterations=%s, method=%s",
            payload.max_iterations,
            payload.optimization_method,
        )

        dataset_dict = self.resolve_dataset(payload.dataset.source, payload.dataset.dataset_name)
        parameter_bounds_dict = {
            name: config.model_dump()
            for name, config in payload.parameter_bounds.items()
        }

        job_id = self.job_manager.start_job(
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
        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise LookupError(f"Job {job_id} not found.")
        return JobResponseFactory.status(
            job_status=job_status,
            poll_interval=get_server_settings().jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def list_jobs(self) -> JobListResponse:
        all_jobs = self.job_manager.list_jobs(self.JOB_TYPE)
        return JobResponseFactory.list(
            job_statuses=all_jobs,
            poll_interval=get_server_settings().jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def cancel_job(self, job_id: str) -> JobCancelResponse:
        success = self.job_manager.cancel_job(job_id)
        if not success:
            raise ValueError(
                f"Job {job_id} cannot be cancelled (not found or already completed)."
            )
        return JobResponseFactory.cancelled(job_id)
