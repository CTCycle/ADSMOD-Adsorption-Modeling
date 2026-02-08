from __future__ import annotations

import os
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from ADSMOD.server.schemas.jobs import (
    JobListResponse,
    JobStartResponse,
    JobStatusResponse,
)
from ADSMOD.server.schemas.training import (
    CheckpointDetailInfo,
    CheckpointFullDetailsResponse,
    CheckpointsResponse,
    DatasetBuildRequest,
    DatasetInfoResponse,
    DatasetSourceInfo,
    DatasetSourcesResponse,
    ProcessedDatasetInfo,
    ProcessedDatasetsResponse,
    ResumeTrainingRequest,
    TrainingConfigRequest,
    TrainingDatasetResponse,
    TrainingMetadata,
    TrainingStartResponse,
    TrainingStatusResponse,
)
from ADSMOD.server.configurations.server import server_settings
from ADSMOD.server.common.utils.logger import logger
from ADSMOD.server.common.constants import CHECKPOINTS_PATH
from ADSMOD.server.services.jobs import job_manager
from ADSMOD.server.learning.training.manager import (
    run_training_process,
    training_manager,
)
from ADSMOD.server.learning.training.worker import ProcessWorker
from ADSMOD.server.services.data.builder import (
    DatasetBuilder,
    DatasetBuilderConfig,
)
from ADSMOD.server.services.data.composition import DatasetCompositionService

router = APIRouter(prefix="/training", tags=["training"])


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


training_session = TrainingSession()


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

    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    def drain_worker_progress(self, job_id: str, worker: ProcessWorker) -> None:
        while True:
            message = worker.poll(timeout=0.0)
            if message is None:
                return
            self.handle_training_progress(job_id, message)

    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    def run_training_job(
        self,
        configuration: dict[str, Any],
        job_id: str,
    ) -> dict[str, Any]:
        worker = ProcessWorker()
        self.session.worker = worker
        try:
            worker.start(
                target=run_training_process,
                kwargs={"configuration": configuration},
            )

            result = self.monitor_training_process(
                job_id,
                worker,
                stop_timeout_seconds=5.0,
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

    # ---------------------------------------------------------------------
    def run_resume_training_job(
        self,
        checkpoint: str,
        additional_epochs: int,
        job_id: str,
    ) -> dict[str, Any]:
        worker = ProcessWorker()
        self.session.worker = worker
        try:
            worker.start(
                target=run_training_process,
                kwargs={
                    "configuration": None,
                    "checkpoint": checkpoint,
                    "additional_epochs": additional_epochs,
                },
            )

            result = self.monitor_training_process(
                job_id,
                worker,
                stop_timeout_seconds=5.0,
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


training_job_runner = TrainingJobRunner(training_session)


###############################################################################
class TrainingEndpoint:
    DATASET_JOB_TYPE = "training_dataset"

    def __init__(self, router: APIRouter) -> None:
        self.router = router

    # -------------------------------------------------------------------------
    @staticmethod
    def determine_checkpoint_compatibility(
        checkpoint_name: str,
        metadata: TrainingMetadata | None,
        dataset_hashes: set[str],
        log_missing_metadata: bool = True,
    ) -> bool:
        return determine_checkpoint_compatibility(
            checkpoint_name=checkpoint_name,
            metadata=metadata,
            dataset_hashes=dataset_hashes,
            log_missing_metadata=log_missing_metadata,
        )

    # -------------------------------------------------------------------------
    def get_training_datasets(self) -> TrainingDatasetResponse:
        try:
            logger.info("Checking training dataset availability")
            info = DatasetBuilder.get_training_dataset_info()

            if info is None:
                return TrainingDatasetResponse(
                    available=False,
                    name=None,
                    train_samples=None,
                    validation_samples=None,
                )

            return TrainingDatasetResponse(
                available=True,
                name="Training Dataset",
                train_samples=info.get("train_samples"),
                validation_samples=info.get("validation_samples"),
            )

        except Exception as e:
            logger.error(f"Error checking training datasets: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # -------------------------------------------------------------------------
    def get_dataset_sources(self) -> DatasetSourcesResponse:
        try:
            composer = DatasetCompositionService()
            datasets = [DatasetSourceInfo(**entry) for entry in composer.list_sources()]
            return DatasetSourcesResponse(datasets=datasets)
        except Exception as e:
            logger.error(f"Error listing dataset sources: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # -------------------------------------------------------------------------
    def delete_dataset_source(self, source: str, dataset_name: str) -> dict[str, str]:
        try:
            composer = DatasetCompositionService()
            success, message = composer.delete_source(source, dataset_name)
            if success:
                return {"status": "success", "message": message}
            return {"status": "error", "message": message}
        except Exception as e:
            logger.error(f"Error deleting dataset source: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # -------------------------------------------------------------------------
    def run_dataset_build(self, request_data: dict[str, Any]) -> dict[str, Any]:
        request = DatasetBuildRequest(**request_data)
        logger.info("Building training dataset with config: %s", request.model_dump())

        reference_metadata = None
        if request.reference_checkpoint:
            checkpoint_path = os.path.join(
                CHECKPOINTS_PATH, request.reference_checkpoint
            )
            if not os.path.isdir(checkpoint_path):
                return {
                    "success": False,
                    "message": "Reference checkpoint not found.",
                }
            try:
                _, reference_metadata, _ = (
                    training_manager.model_serializer.load_training_configuration(
                        checkpoint_path
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to load reference checkpoint metadata from %s: %s",
                    request.reference_checkpoint,
                    exc,
                )
                return {
                    "success": False,
                    "message": "Failed to load reference checkpoint metadata.",
                }

        config = DatasetBuilderConfig(
            sample_size=request.sample_size,
            validation_size=request.validation_size,
            min_measurements=request.min_measurements,
            max_measurements=request.max_measurements,
            smile_sequence_size=request.smile_sequence_size,
            max_pressure=request.max_pressure,
            max_uptake=request.max_uptake,
        )
        composer = DatasetCompositionService(allow_pubchem_fetch=False)
        selections = [selection.model_dump() for selection in request.datasets]
        adsorption_data, guest_data, host_data, dataset_label = (
            composer.compose_datasets(selections)
        )

        builder = DatasetBuilder(config, dataset_label=request.dataset_label)
        result = builder.build_training_dataset(
            adsorption_data=adsorption_data,
            guest_data=guest_data,
            host_data=host_data,
            dataset_name=dataset_label,
            reference_metadata=reference_metadata,
        )

        if result.get("success"):
            return {
                "success": True,
                "message": "Training dataset built successfully.",
                "total_samples": result.get("total_samples"),
                "train_samples": result.get("train_samples"),
                "validation_samples": result.get("validation_samples"),
            }
        return {
            "success": False,
            "message": result.get("error", "Unknown error during dataset building."),
        }

    # -------------------------------------------------------------------------
    def build_training_dataset(self, request: DatasetBuildRequest) -> JobStartResponse:
        if job_manager.is_job_running(self.DATASET_JOB_TYPE):
            raise HTTPException(
                status_code=400,
                detail="A dataset build job is already running.",
            )

        job_id = job_manager.start_job(
            job_type=self.DATASET_JOB_TYPE,
            runner=self.run_dataset_build,
            args=(request.model_dump(),),
        )
        return JobStartResponse(
            job_id=job_id,
            job_type=self.DATASET_JOB_TYPE,
            status="running",
            message="Dataset build job started.",
            poll_interval=server_settings.jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def get_dataset_job_status(self, job_id: str) -> JobStatusResponse:
        job_status = job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
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
    def list_dataset_jobs(self) -> JobListResponse:
        all_jobs = job_manager.list_jobs(self.DATASET_JOB_TYPE)
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
    def cancel_dataset_job(self, job_id: str) -> dict[str, str]:
        success = job_manager.cancel_job(job_id)
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Job {job_id} cannot be cancelled.",
            )
        return {"status": "cancelled", "job_id": job_id}

    # -------------------------------------------------------------------------
    def get_processed_datasets(self) -> ProcessedDatasetsResponse:
        """Returns a list of all processed datasets with their metadata."""
        try:
            datasets_list = DatasetBuilder.list_processed_datasets()
            datasets = [ProcessedDatasetInfo(**entry) for entry in datasets_list]
            return ProcessedDatasetsResponse(datasets=datasets)
        except Exception as e:
            logger.error(f"Error listing processed datasets: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # -------------------------------------------------------------------------
    def get_dataset_info(self, dataset_label: str = "default") -> DatasetInfoResponse:
        try:
            resolved_label = training_manager.data_serializer.normalize_dataset_label(
                dataset_label
            )
            info = DatasetBuilder.get_training_dataset_info(resolved_label)

            if info is None:
                return DatasetInfoResponse(available=False)

            return DatasetInfoResponse(
                available=True,
                dataset_label=info.get("dataset_label"),
                created_at=info.get("created_at"),
                sample_size=info.get("sample_size"),
                validation_size=info.get("validation_size"),
                min_measurements=info.get("min_measurements"),
                max_measurements=info.get("max_measurements"),
                smile_sequence_size=info.get("smile_sequence_size"),
                max_pressure=info.get("max_pressure"),
                max_uptake=info.get("max_uptake"),
                total_samples=info.get("total_samples"),
                train_samples=info.get("train_samples"),
                validation_samples=info.get("validation_samples"),
            )

        except Exception as e:
            logger.error(f"Error getting dataset info: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # -------------------------------------------------------------------------
    def clear_training_dataset(
        self, dataset_label: str | None = None
    ) -> dict[str, str]:
        try:
            resolved_label = (
                training_manager.data_serializer.normalize_dataset_label(dataset_label)
                if dataset_label is not None
                else None
            )
            success = DatasetBuilder.clear_training_dataset(resolved_label)

            if success:
                msg = (
                    f"Training dataset '{resolved_label}' cleared."
                    if resolved_label
                    else "All training datasets cleared."
                )
                return {"status": "success", "message": msg}
            else:
                return {
                    "status": "error",
                    "message": "Failed to clear training dataset.",
                }

        except Exception as e:
            logger.error(f"Error clearing training dataset: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # -------------------------------------------------------------------------
    def get_checkpoints(self) -> CheckpointsResponse:
        try:
            logger.info("Scanning for available checkpoints")
            checkpoints = training_manager.model_serializer.scan_checkpoints_folder()
            dataset_hashes = training_manager.data_serializer.collect_dataset_hashes()
            detailed_checkpoints: list[CheckpointDetailInfo] = []

            for checkpoint in checkpoints:
                checkpoint_path = os.path.join(CHECKPOINTS_PATH, checkpoint)
                epochs_trained: int | None = None
                final_loss: float | None = None
                final_accuracy: float | None = None
                is_compatible = False
                metadata: TrainingMetadata | None = None
                metadata_load_failed = False

                try:
                    training_configuration, metadata, session = (
                        training_manager.model_serializer.load_training_configuration(
                            checkpoint_path
                        )
                    )
                    session_history = (
                        session.get("history") if isinstance(session, dict) else {}
                    )
                    if not isinstance(session_history, dict):
                        session_history = {}

                    epochs_value = (
                        session.get("epochs") if isinstance(session, dict) else None
                    )
                    if isinstance(epochs_value, int):
                        epochs_trained = epochs_value

                    loss_values = session_history.get("loss")
                    if isinstance(loss_values, list) and loss_values:
                        last_loss = loss_values[-1]
                        if isinstance(last_loss, (int, float)):
                            final_loss = float(last_loss)
                        if epochs_trained is None:
                            epochs_trained = len(loss_values)

                    if epochs_trained is None and isinstance(
                        training_configuration, dict
                    ):
                        configured_epochs = training_configuration.get("epochs")
                        if isinstance(configured_epochs, int):
                            epochs_trained = configured_epochs

                    for metric_key in [
                        "accuracy",
                        "MaskedR2",
                        "masked_r2",
                        "masked_r_squared",
                        "val_accuracy",
                    ]:
                        metric_values = session_history.get(metric_key)
                        if isinstance(metric_values, list) and metric_values:
                            last_value = metric_values[-1]
                            if isinstance(last_value, (int, float)):
                                final_accuracy = float(last_value)
                            break

                except Exception as exc:  # noqa: BLE001
                    metadata_load_failed = True
                    logger.warning(
                        "Failed to load checkpoint details for %s: %s",
                        checkpoint,
                        exc,
                    )

                is_compatible = self.determine_checkpoint_compatibility(
                    checkpoint,
                    metadata,
                    dataset_hashes,
                    log_missing_metadata=not metadata_load_failed,
                )

                detailed_checkpoints.append(
                    CheckpointDetailInfo(
                        name=checkpoint,
                        epochs_trained=epochs_trained,
                        final_loss=final_loss,
                        final_accuracy=final_accuracy,
                        is_compatible=is_compatible,
                    )
                )

            return CheckpointsResponse(checkpoints=detailed_checkpoints)

        except Exception as e:
            logger.error(f"Error scanning checkpoints: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # -------------------------------------------------------------------------
    def get_checkpoint_details(
        self, checkpoint_name: str
    ) -> CheckpointFullDetailsResponse:
        try:
            checkpoint_path = os.path.join(CHECKPOINTS_PATH, checkpoint_name)
            if not os.path.isdir(checkpoint_path):
                raise HTTPException(
                    status_code=404, detail=f"Checkpoint {checkpoint_name} not found"
                )

            configuration, metadata, history = (
                training_manager.model_serializer.load_training_configuration(
                    checkpoint_path
                )
            )

            return CheckpointFullDetailsResponse(
                name=checkpoint_name,
                configuration=configuration,
                metadata=metadata,
                history=history,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting checkpoint details: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # -------------------------------------------------------------------------
    def delete_checkpoint(self, checkpoint_name: str) -> dict[str, str]:
        try:
            success = training_manager.model_serializer.delete_checkpoint(
                checkpoint_name
            )
            if success:
                return {
                    "status": "success",
                    "message": f"Checkpoint {checkpoint_name} deleted.",
                }
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to delete checkpoint {checkpoint_name}.",
                )
        except Exception as e:
            logger.error(f"Error deleting checkpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # -------------------------------------------------------------------------
    def start_training(self, config: TrainingConfigRequest) -> TrainingStartResponse:
        state = training_manager.state.snapshot()
        if state["is_training"] or job_manager.is_job_running("training"):
            raise HTTPException(
                status_code=400,
                detail="Training is already in progress. Stop it first.",
            )

        try:
            # Force mixed precision setting from server configuration
            config.use_mixed_precision = server_settings.training.use_mixed_precision

            resolved_label = training_manager.data_serializer.normalize_dataset_label(
                config.dataset_label
            )
            configuration = config.model_dump()
            configuration["dataset_label"] = resolved_label
            configuration["polling_interval"] = server_settings.jobs.polling_interval

            logger.info("Starting training with config: %s", configuration)
            metadata = training_manager.data_serializer.load_training_metadata(
                resolved_label
            )
            requested_hash = configuration.get("dataset_hash")
            if requested_hash and metadata.dataset_hash:
                if requested_hash != metadata.dataset_hash:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Selected dataset does not match the stored training metadata. "
                            "Refresh the dataset list and try again."
                        ),
                    )
            info = DatasetBuilder.get_training_dataset_info(resolved_label)
            if info is None:
                raise HTTPException(
                    status_code=400,
                    detail="No training dataset available. Build the dataset first.",
                )

            job_id = job_manager.start_job(
                job_type="training",
                runner=training_job_runner.run_training_job,
                kwargs={
                    "configuration": configuration,
                },
            )

            total_epochs = int(configuration.get("epochs", 0))
            training_session.reset_for_new_session(
                total_epochs=total_epochs,
                job_id=job_id,
                current_epoch=0,
                message="Starting training session",
            )
            state_snapshot = training_manager.state.snapshot()
            job_manager.update_result(
                job_id,
                {
                    "current_epoch": state_snapshot["current_epoch"],
                    "total_epochs": state_snapshot["total_epochs"],
                    "progress": 0.0,
                    "metrics": state_snapshot["metrics"],
                },
            )

            return TrainingStartResponse(
                status="started",
                session_id=job_id,
                message=f"Training started with {config.epochs} epochs. Session: {job_id}",
                poll_interval=server_settings.jobs.polling_interval,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error starting training: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # -------------------------------------------------------------------------
    def resume_training(self, request: ResumeTrainingRequest) -> TrainingStartResponse:
        state = training_manager.state.snapshot()
        if state["is_training"] or job_manager.is_job_running("training"):
            raise HTTPException(
                status_code=400,
                detail="Training is already in progress. Stop it first.",
            )

        try:
            logger.info(
                f"Resuming training from checkpoint: {request.checkpoint_name} "
                f"with {request.additional_epochs} additional epochs"
            )
            available = training_manager.model_serializer.scan_checkpoints_folder()
            if request.checkpoint_name not in available:
                raise HTTPException(
                    status_code=404,
                    detail=f"Checkpoint '{request.checkpoint_name}' not found.",
                )

            checkpoint_path = os.path.join(CHECKPOINTS_PATH, request.checkpoint_name)
            try:
                _, _, session = (
                    training_manager.model_serializer.load_training_configuration(
                        checkpoint_path
                    )
                )
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to load checkpoint metadata: {exc}",
                ) from exc

            from_epoch = 0
            if isinstance(session, dict):
                from_epoch_value = session.get("epochs", 0)
                if isinstance(from_epoch_value, int):
                    from_epoch = from_epoch_value
            history_entries = training_manager.build_history_entries(
                session if isinstance(session, dict) else {}
            )
            last_metrics = training_manager.extract_last_metrics(history_entries)

            job_id = job_manager.start_job(
                job_type="training",
                runner=training_job_runner.run_resume_training_job,
                kwargs={
                    "checkpoint": request.checkpoint_name,
                    "additional_epochs": request.additional_epochs,
                },
            )

            total_epochs = from_epoch + request.additional_epochs
            training_session.reset_for_new_session(
                total_epochs=total_epochs,
                job_id=job_id,
                current_epoch=from_epoch,
                message="Resuming training session",
                history=history_entries,
                metrics=last_metrics,
            )
            state_snapshot = training_manager.state.snapshot()
            job_manager.update_result(
                job_id,
                {
                    "current_epoch": state_snapshot["current_epoch"],
                    "total_epochs": state_snapshot["total_epochs"],
                    "progress": (from_epoch / total_epochs * 100.0)
                    if total_epochs
                    else 0.0,
                    "metrics": state_snapshot["metrics"],
                },
            )

            return TrainingStartResponse(
                status="started",
                session_id=job_id,
                message=(
                    f"Resuming training from {request.checkpoint_name} "
                    f"with {request.additional_epochs} epochs. Session: {job_id}"
                ),
                poll_interval=server_settings.jobs.polling_interval,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error resuming training: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # -------------------------------------------------------------------------
    def stop_training(self) -> dict[str, str]:
        state = training_manager.state.snapshot()
        if not state["is_training"]:
            return {"status": "stopped", "message": "No training session is running."}

        try:
            logger.info("Stop requested for current training session")
            training_manager.state.update(stop_requested=True)
            training_manager.state.add_log("Stop requested by user...")
            if training_session.worker is not None:
                training_session.worker.stop()
            if training_session.current_job_id:
                job_manager.cancel_job(training_session.current_job_id)

            return {"status": "stopped", "message": "Training stop requested."}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error stopping training: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # -------------------------------------------------------------------------
    def get_training_status(self) -> TrainingStatusResponse:
        state = training_manager.state.snapshot()
        progress = state.get("progress", 0.0)
        if not isinstance(progress, (int, float)):
            progress = 0.0
        if progress <= 0.0 and state["total_epochs"] > 0:
            progress = (state["current_epoch"] / state["total_epochs"]) * 100

        return TrainingStatusResponse(
            is_training=state["is_training"],
            current_epoch=state["current_epoch"],
            total_epochs=state["total_epochs"],
            progress=progress,
            metrics=state["metrics"],
            history=state["history"],
            log=state["log"],
            poll_interval=server_settings.jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "/datasets",
            self.get_training_datasets,
            methods=["GET"],
            response_model=TrainingDatasetResponse,
        )
        self.router.add_api_route(
            "/dataset-sources",
            self.get_dataset_sources,
            methods=["GET"],
            response_model=DatasetSourcesResponse,
        )
        self.router.add_api_route(
            "/dataset-source",
            self.delete_dataset_source,
            methods=["DELETE"],
        )
        self.router.add_api_route(
            "/build-dataset",
            self.build_training_dataset,
            methods=["POST"],
            response_model=JobStartResponse,
        )
        self.router.add_api_route(
            "/processed-datasets",
            self.get_processed_datasets,
            methods=["GET"],
            response_model=ProcessedDatasetsResponse,
        )
        self.router.add_api_route(
            "/dataset-info",
            self.get_dataset_info,
            methods=["GET"],
            response_model=DatasetInfoResponse,
        )
        self.router.add_api_route(
            "/dataset",
            self.clear_training_dataset,
            methods=["DELETE"],
        )
        self.router.add_api_route(
            "/jobs",
            self.list_dataset_jobs,
            methods=["GET"],
            response_model=JobListResponse,
        )
        self.router.add_api_route(
            "/jobs/{job_id}",
            self.get_dataset_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
        )
        self.router.add_api_route(
            "/jobs/{job_id}",
            self.cancel_dataset_job,
            methods=["DELETE"],
        )
        self.router.add_api_route(
            "/checkpoints",
            self.get_checkpoints,
            methods=["GET"],
            response_model=CheckpointsResponse,
        )
        self.router.add_api_route(
            "/checkpoints/{checkpoint_name}",
            self.get_checkpoint_details,
            methods=["GET"],
            response_model=CheckpointFullDetailsResponse,
        )
        self.router.add_api_route(
            "/checkpoints/{checkpoint_name}",
            self.delete_checkpoint,
            methods=["DELETE"],
        )
        self.router.add_api_route(
            "/start",
            self.start_training,
            methods=["POST"],
            response_model=TrainingStartResponse,
        )
        self.router.add_api_route(
            "/resume",
            self.resume_training,
            methods=["POST"],
            response_model=TrainingStartResponse,
        )
        self.router.add_api_route(
            "/stop",
            self.stop_training,
            methods=["POST"],
        )
        self.router.add_api_route(
            "/status",
            self.get_training_status,
            methods=["GET"],
            response_model=TrainingStatusResponse,
        )


###############################################################################
training_endpoint = TrainingEndpoint(router=router)
training_endpoint.add_routes()
