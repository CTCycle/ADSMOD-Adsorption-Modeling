"""Training routes for ML model training and checkpoint management."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ADSMOD.server.database.database import database
from ADSMOD.server.utils.logger import logger
from ADSMOD.server.utils.services.builder import DatasetBuilder, DatasetBuilderConfig
from ADSMOD.server.utils.services.training_manager import training_manager

router = APIRouter(prefix="/training", tags=["training"])


###############################################################################
# Request/Response Models
###############################################################################
class TrainingConfigRequest(BaseModel):
    """Training configuration from frontend."""

    # Dataset settings
    sample_size: float = Field(default=1.0, ge=0.01, le=1.0)
    validation_size: float = Field(default=0.2, ge=0.05, le=0.5)
    batch_size: int = Field(default=32, ge=1, le=256)
    shuffle_dataset: bool = True
    shuffle_size: int = Field(default=1000, ge=100, le=10000)

    # Model settings
    selected_model: str = "SCADS Series"
    dropout_rate: float = Field(default=0.1, ge=0.0, le=0.5)
    num_attention_heads: int = Field(default=8, ge=1, le=16)
    num_encoders: int = Field(default=4, ge=1, le=12)
    molecular_embedding_size: int = Field(default=256, ge=64, le=1024)

    # Training settings
    epochs: int = Field(default=50, ge=1, le=500)

    # LR scheduler settings
    use_lr_scheduler: bool = True
    initial_lr: float = Field(default=1e-4, ge=1e-7, le=1e-2)
    target_lr: float = Field(default=1e-5, ge=1e-8, le=1e-3)
    constant_steps: int = Field(default=5, ge=0, le=50)
    decay_steps: int = Field(default=10, ge=1, le=100)

    # Callbacks
    save_checkpoints: bool = True
    checkpoints_frequency: int = Field(default=5, ge=1, le=50)


class ResumeTrainingRequest(BaseModel):
    """Request to resume training from a checkpoint."""

    checkpoint_name: str
    additional_epochs: int = Field(default=10, ge=1, le=100)


class TrainingDatasetResponse(BaseModel):
    """Response with training dataset availability info."""

    available: bool
    name: str | None = None
    train_samples: int | None = None
    validation_samples: int | None = None


class CheckpointsResponse(BaseModel):
    """Response with list of available checkpoints."""

    checkpoints: list[str]


class TrainingStartResponse(BaseModel):
    """Response after starting training."""

    status: str
    session_id: str
    message: str


class TrainingStatusResponse(BaseModel):
    """Response with current training status."""

    is_training: bool
    current_epoch: int
    total_epochs: int
    progress: float


class DatasetBuildRequest(BaseModel):
    """Request to build a training dataset."""

    sample_size: float = Field(default=1.0, ge=0.01, le=1.0)
    validation_size: float = Field(default=0.2, ge=0.05, le=0.5)
    min_measurements: int = Field(default=1, ge=1, le=100)
    max_measurements: int = Field(default=30, ge=5, le=500)
    smile_sequence_size: int = Field(default=20, ge=5, le=100)
    max_pressure: float = Field(default=10000.0, ge=100.0, le=100000.0)
    max_uptake: float = Field(default=20.0, ge=1.0, le=1000.0)
    source_datasets: list[str] = Field(default=["SINGLE_COMPONENT_ADSORPTION"])


class DatasetBuildResponse(BaseModel):
    """Response after building a dataset."""

    success: bool
    message: str
    total_samples: int | None = None
    train_samples: int | None = None
    validation_samples: int | None = None


class DatasetInfoResponse(BaseModel):
    """Response with training dataset info."""

    available: bool
    created_at: str | None = None
    sample_size: float | None = None
    validation_size: float | None = None
    min_measurements: int | None = None
    max_measurements: int | None = None
    smile_sequence_size: int | None = None
    max_pressure: float | None = None
    max_uptake: float | None = None
    total_samples: int | None = None
    train_samples: int | None = None
    validation_samples: int | None = None


###############################################################################
class TrainingEndpoint:
    """Endpoint for ML model training and checkpoint management operations."""

    def __init__(self, router: APIRouter) -> None:
        self.router = router

    # -------------------------------------------------------------------------
    async def get_training_datasets(self) -> TrainingDatasetResponse:
        """Check if training datasets are available in the database."""
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
    async def build_training_dataset(
        self, request: DatasetBuildRequest
    ) -> DatasetBuildResponse:
        """Build a new training dataset from raw adsorption data."""
        try:
            logger.info(f"Building training dataset with config: {request.model_dump()}")

            config = DatasetBuilderConfig(
                sample_size=request.sample_size,
                validation_size=request.validation_size,
                min_measurements=request.min_measurements,
                max_measurements=request.max_measurements,
                smile_sequence_size=request.smile_sequence_size,
                max_pressure=request.max_pressure,
                max_uptake=request.max_uptake,
            )

            builder = DatasetBuilder(config)

            source_datasets = [name.upper() for name in request.source_datasets]
            guest_data = None
            host_data = None
            if "SINGLE_COMPONENT_ADSORPTION" in source_datasets:
                adsorption_data = database.load_from_database(
                    "SINGLE_COMPONENT_ADSORPTION"
                )
                guest_data = database.load_from_database("ADSORBATES")
                host_data = database.load_from_database("ADSORBENTS")
                dataset_name = "nist"
            else:
                return DatasetBuildResponse(
                    success=False,
                    message="Unsupported dataset source. Use SINGLE_COMPONENT_ADSORPTION.",
                )

            if adsorption_data.empty:
                return DatasetBuildResponse(
                    success=False,
                    message="No adsorption data available. Please fetch NIST data first.",
                )

            result = builder.build_training_dataset(
                adsorption_data=adsorption_data,
                guest_data=guest_data,
                host_data=host_data,
                dataset_name=dataset_name,
            )

            if result.get("success"):
                return DatasetBuildResponse(
                    success=True,
                    message="Training dataset built successfully.",
                    total_samples=result.get("total_samples"),
                    train_samples=result.get("train_samples"),
                    validation_samples=result.get("validation_samples"),
                )
            else:
                return DatasetBuildResponse(
                    success=False,
                    message=result.get("error", "Unknown error during dataset building."),
                )

        except Exception as e:
            logger.error(f"Error building training dataset: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # -------------------------------------------------------------------------
    async def get_dataset_info(self) -> DatasetInfoResponse:
        """Get current training dataset info and metadata."""
        try:
            info = DatasetBuilder.get_training_dataset_info()

            if info is None:
                return DatasetInfoResponse(available=False)

            return DatasetInfoResponse(
                available=True,
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
    async def clear_training_dataset(self) -> dict[str, str]:
        """Clear the current training dataset."""
        try:
            success = DatasetBuilder.clear_training_dataset()

            if success:
                return {"status": "success", "message": "Training dataset cleared."}
            else:
                return {"status": "error", "message": "Failed to clear training dataset."}

        except Exception as e:
            logger.error(f"Error clearing training dataset: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # -------------------------------------------------------------------------
    async def get_checkpoints(self) -> CheckpointsResponse:
        """List available model checkpoints."""
        try:
            logger.info("Scanning for available checkpoints")
            checkpoints = training_manager.model_serializer.scan_checkpoints_folder()

            return CheckpointsResponse(checkpoints=checkpoints)

        except Exception as e:
            logger.error(f"Error scanning checkpoints: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # -------------------------------------------------------------------------
    async def start_training(self, config: TrainingConfigRequest) -> TrainingStartResponse:
        """Start a new training session."""
        state = training_manager.state.snapshot()
        if state["is_training"]:
            raise HTTPException(
                status_code=400, detail="Training is already in progress. Stop it first."
            )

        try:
            logger.info(f"Starting training with config: {config.model_dump()}")
            info = DatasetBuilder.get_training_dataset_info()
            if info is None:
                raise HTTPException(
                    status_code=400,
                    detail="No training dataset available. Build the dataset first.",
                )

            session_id = training_manager.start_training(config.model_dump())

            return TrainingStartResponse(
                status="started",
                session_id=session_id,
                message=f"Training started with {config.epochs} epochs. Session: {session_id}",
            )

        except Exception as e:
            logger.error(f"Error starting training: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # -------------------------------------------------------------------------
    async def resume_training(
        self, request: ResumeTrainingRequest
    ) -> TrainingStartResponse:
        """Resume training from a checkpoint."""
        state = training_manager.state.snapshot()
        if state["is_training"]:
            raise HTTPException(
                status_code=400, detail="Training is already in progress. Stop it first."
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
            session_id = training_manager.resume_training(
                request.checkpoint_name,
                request.additional_epochs,
            )

            return TrainingStartResponse(
                status="started",
                session_id=session_id,
                message=f"Resuming training from {request.checkpoint_name} with {request.additional_epochs} epochs. Session: {session_id}",
            )

        except Exception as e:
            logger.error(f"Error resuming training: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # -------------------------------------------------------------------------
    async def stop_training(self) -> dict[str, str]:
        """Stop the current training session."""
        state = training_manager.state.snapshot()
        if not state["is_training"]:
            return {"status": "stopped", "message": "No training session is running."}

        try:
            logger.info("Stop requested for current training session")
            training_manager.stop_training()

            return {"status": "stopped", "message": "Training stop requested."}

        except Exception as e:
            logger.error(f"Error stopping training: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # -------------------------------------------------------------------------
    async def get_training_status(self) -> TrainingStatusResponse:
        """Get current training status."""
        state = training_manager.state.snapshot()
        progress = 0.0
        if state["total_epochs"] > 0:
            progress = (state["current_epoch"] / state["total_epochs"]) * 100

        return TrainingStatusResponse(
            is_training=state["is_training"],
            current_epoch=state["current_epoch"],
            total_epochs=state["total_epochs"],
            progress=progress,
        )

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        """Register all training-related routes with the router."""
        self.router.add_api_route(
            "/datasets",
            self.get_training_datasets,
            methods=["GET"],
            response_model=TrainingDatasetResponse,
        )
        self.router.add_api_route(
            "/build-dataset",
            self.build_training_dataset,
            methods=["POST"],
            response_model=DatasetBuildResponse,
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
            "/checkpoints",
            self.get_checkpoints,
            methods=["GET"],
            response_model=CheckpointsResponse,
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
