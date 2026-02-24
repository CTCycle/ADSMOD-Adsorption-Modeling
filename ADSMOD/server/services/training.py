from __future__ import annotations

import time
from typing import Any

from ADSMOD.server.common.utils.logger import logger
from ADSMOD.server.configurations.server import server_settings
from ADSMOD.server.entities.training import TrainingMetadata
from ADSMOD.server.learning.training.manager import (
    run_training_process,
    training_manager,
)
from ADSMOD.server.learning.training.worker import ProcessWorker
from ADSMOD.server.services.jobs import job_manager

TRAINING_PROCESS_STOP_TIMEOUT_SECONDS = max(
    5.0,
    float(server_settings.jobs.polling_interval),
)


###############################################################################
class TrainingSession:
    def __init__(self) -> None:
        self.worker: ProcessWorker | None = None
        self.current_job_id: str | None = None

    # -------------------------------------------------------------------------
    def reset_for_new_session(
        self,
        total_epochs: int,
        job_id: str,
        current_epoch: int = 0,
        message: str | None = None,
        history: list[dict[str, Any]] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        if history is None:
            history = []
        if metrics is None:
            metrics = {}
        training_manager.state.update(
            is_training=True,
            current_epoch=current_epoch,
            total_epochs=total_epochs,
            progress=0.0,
            session_id=job_id,
            stop_requested=False,
            last_error=None,
            metrics=metrics,
            history=history,
            log=[],
        )
        log_message = message or "Starting training session"
        training_manager.state.add_log(f"{log_message}: {job_id}")
        self.current_job_id = job_id

    # -------------------------------------------------------------------------
    def finish_session(self) -> None:
        self.worker = None
        self.current_job_id = None


###############################################################################
def determine_checkpoint_compatibility(
    checkpoint_name: str,
    metadata: TrainingMetadata | None,
    dataset_hashes: set[str],
    log_missing_metadata: bool = True,
) -> bool:
    if metadata is None:
        if log_missing_metadata:
            logger.warning(
                "Checkpoint %s metadata missing or invalid; marking incompatible.",
                checkpoint_name,
            )
        return False

    checkpoint_hash = metadata.dataset_hash
    if not checkpoint_hash:
        logger.warning(
            "Checkpoint %s metadata missing dataset_hash; marking incompatible.",
            checkpoint_name,
        )
        return False

    if not dataset_hashes:
        return False

    return checkpoint_hash in dataset_hashes


###############################################################################
class TrainingJobRunner:
    def __init__(self, session: TrainingSession) -> None:
        self.session = session

    # -------------------------------------------------------------------------
    def handle_training_progress(self, job_id: str, message: dict[str, Any]) -> None:
        training_manager.handle_process_message(job_id, message)

        if not job_id:
            return

        state = training_manager.state.snapshot()
        progress = state.get("progress", 0.0)
        if not isinstance(progress, (int, float)):
            progress = 0.0
        if progress <= 0.0 and state["total_epochs"] > 0:
            progress = (state["current_epoch"] / state["total_epochs"]) * 100

        job_manager.update_progress(job_id, progress)
        job_manager.update_result(
            job_id,
            {
                "current_epoch": state["current_epoch"],
                "total_epochs": state["total_epochs"],
                "progress": progress,
                "metrics": state["metrics"],
            },
        )

    # -------------------------------------------------------------------------
    def drain_worker_progress(self, job_id: str, worker: ProcessWorker) -> None:
        while True:
            message = worker.poll(timeout=0.0)
            if message is None:
                return
            self.handle_training_progress(job_id, message)

    # -------------------------------------------------------------------------
    def monitor_training_process(
        self,
        job_id: str,
        worker: ProcessWorker,
        stop_timeout_seconds: float,
    ) -> dict[str, Any]:
        stop_requested_at: float | None = None

        while worker.is_alive():
            if job_manager.should_stop(job_id):
                if not worker.is_interrupted():
                    worker.stop()
                    stop_requested_at = time.monotonic()
            if stop_requested_at is not None:
                elapsed = time.monotonic() - stop_requested_at
                if elapsed >= stop_timeout_seconds:
                    worker.terminate()
                    break
            message = worker.poll(timeout=0.25)
            if message is not None:
                self.handle_training_progress(job_id, message)
                self.drain_worker_progress(job_id, worker)

        worker.join(timeout=5)
        self.drain_worker_progress(job_id, worker)

        result_payload = worker.read_result()
        if result_payload is None:
            if worker.exitcode not in (0, None) and not job_manager.should_stop(job_id):
                raise RuntimeError(
                    f"Training process exited with code {worker.exitcode}"
                )
            return {}
        if result_payload.get("error"):
            raise RuntimeError(str(result_payload.get("error")))
        if "result" in result_payload:
            return result_payload.get("result") or {}
        return {}

    # -------------------------------------------------------------------------
    def run_process_job(
        self,
        job_id: str,
        process_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        worker = ProcessWorker()
        self.session.worker = worker
        try:
            worker.start(
                target=run_training_process,
                kwargs=process_kwargs,
            )

            result = self.monitor_training_process(
                job_id,
                worker,
                stop_timeout_seconds=TRAINING_PROCESS_STOP_TIMEOUT_SECONDS,
            )

            if job_manager.should_stop(job_id):
                training_manager.handle_job_completion(job_id, "cancelled", None, None)
                return {}

            training_manager.handle_job_completion(job_id, "completed", result, None)
            return result
        except Exception as exc:  # noqa: BLE001
            if job_manager.should_stop(job_id):
                training_manager.handle_job_completion(job_id, "cancelled", None, None)
                return {}
            training_manager.handle_job_completion(job_id, "failed", None, str(exc))
            raise
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
            worker.cleanup()
            self.session.finish_session()


###############################################################################
training_session = TrainingSession()
training_job_runner = TrainingJobRunner(training_session)
