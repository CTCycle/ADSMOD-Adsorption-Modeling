from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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


###############################################################################
class ResumeTrainingRequest(BaseModel):
    """Request to resume training from a checkpoint."""

    checkpoint_name: str
    additional_epochs: int = Field(default=10, ge=1, le=100)


###############################################################################
class TrainingDatasetResponse(BaseModel):
    """Response with training dataset availability info."""

    available: bool
    name: str | None = None
    train_samples: int | None = None
    validation_samples: int | None = None


###############################################################################
class CheckpointsResponse(BaseModel):
    """Response with list of available checkpoints."""

    checkpoints: list[str]


###############################################################################
class TrainingStartResponse(BaseModel):
    """Response after starting training."""

    status: str
    session_id: str
    message: str


###############################################################################
class TrainingStatusResponse(BaseModel):
    """Response with current training status."""

    is_training: bool
    current_epoch: int
    total_epochs: int
    progress: float


###############################################################################
class DatasetSelection(BaseModel):
    """Dataset selection for training dataset composition."""

    source: Literal["nist", "uploaded"]
    dataset_name: str = Field(min_length=1)


###############################################################################
class DatasetBuildRequest(BaseModel):
    """Request to build a training dataset."""

    sample_size: float = Field(default=1.0, ge=0.01, le=1.0)
    validation_size: float = Field(default=0.2, ge=0.05, le=0.5)
    min_measurements: int = Field(default=1, ge=1, le=100)
    max_measurements: int = Field(default=30, ge=5, le=500)
    smile_sequence_size: int = Field(default=20, ge=5, le=100)
    max_pressure: float = Field(default=10000.0, ge=100.0, le=100000.0)
    max_uptake: float = Field(default=20.0, ge=1.0, le=1000.0)
    datasets: list[DatasetSelection] = Field(default_factory=list, min_length=1)


###############################################################################
class DatasetSourceInfo(BaseModel):
    """Response entry for available dataset sources."""

    source: Literal["nist", "uploaded"]
    dataset_name: str
    display_name: str
    row_count: int


###############################################################################
class DatasetSourcesResponse(BaseModel):
    """Response listing available dataset sources for processing."""

    datasets: list[DatasetSourceInfo]


###############################################################################
class DatasetBuildResponse(BaseModel):
    """Response after building a dataset."""

    success: bool
    message: str
    total_samples: int | None = None
    train_samples: int | None = None
    validation_samples: int | None = None


###############################################################################
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
