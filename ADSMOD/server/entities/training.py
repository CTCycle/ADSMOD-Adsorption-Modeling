from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field


###############################################################################
class TrainingConfigRequest(BaseModel):
    # Dataset settings
    sample_size: float = Field(default=1.0, ge=0.01, le=1.0)
    validation_size: float = Field(default=0.2, ge=0.05, le=0.5)
    batch_size: int = Field(default=32, ge=1, le=256)
    shuffle_dataset: bool = True
    dataset_label: str | None = None
    dataset_hash: str | None = None

    # Model settings
    selected_model: str = "SCADS Series"
    dropout_rate: float = Field(default=0.1, ge=0.0, le=0.5)
    num_attention_heads: int = Field(default=8, ge=1, le=16)
    num_encoders: int = Field(default=4, ge=1, le=12)
    molecular_embedding_size: int = Field(default=256, ge=64, le=1024)

    # Training settings
    epochs: int = Field(default=50, ge=1, le=500)
    use_device_GPU: bool = True
    use_mixed_precision: bool = False

    # LR scheduler settings
    use_lr_scheduler: bool = True
    initial_lr: float = Field(default=1e-4, ge=1e-7, le=1e-2)
    target_lr: float = Field(default=1e-5, ge=1e-8, le=1e-3)
    constant_steps: int = Field(default=5, ge=0, le=50)
    decay_steps: int = Field(default=10, ge=1, le=100)

    # Callbacks
    save_checkpoints: bool = True
    checkpoints_frequency: int = Field(default=5, ge=1, le=50)
    custom_name: str | None = None


###############################################################################
class ResumeTrainingRequest(BaseModel):
    checkpoint_name: str
    additional_epochs: int = Field(default=10, ge=1, le=100)


###############################################################################
class TrainingDatasetResponse(BaseModel):
    available: bool
    name: str | None = None
    train_samples: int | None = None
    validation_samples: int | None = None


###############################################################################
class CheckpointDetailInfo(BaseModel):
    name: str
    epochs_trained: int | None = None
    final_loss: float | None = None
    final_accuracy: float | None = None
    is_compatible: bool = True


###############################################################################
class CheckpointFullDetailsResponse(BaseModel):
    name: str
    configuration: dict[str, Any] | None
    metadata: TrainingMetadata | None
    history: dict[str, Any] | None = None


###############################################################################
class CheckpointsResponse(BaseModel):
    checkpoints: list[CheckpointDetailInfo]


###############################################################################
class TrainingStartResponse(BaseModel):
    status: str
    session_id: str
    message: str
    poll_interval: float | None = None


###############################################################################
class TrainingStatusResponse(BaseModel):
    is_training: bool
    current_epoch: int
    total_epochs: int
    progress: float
    metrics: dict[str, float] = Field(default_factory=dict)
    history: list[dict[str, Any]] = Field(default_factory=list)
    log: list[str] = Field(default_factory=list)
    poll_interval: float | None = None


###############################################################################
@dataclass
class TrainingState:
    is_training: bool = False
    current_epoch: int = 0
    total_epochs: int = 0
    progress: float = 0.0
    session_id: str | None = None
    stop_requested: bool = False
    last_error: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    # ---------------------------------------------------------------------
    def update(self, **kwargs: Any) -> None:
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    # ---------------------------------------------------------------------
    def add_log(self, message: str) -> None:
        with self.lock:
            self.log.append(message)
            if len(self.log) > 1000:
                self.log = self.log[-1000:]

    # ---------------------------------------------------------------------
    def add_history(self, epoch_data: dict[str, Any]) -> None:
        with self.lock:
            self.history.append(epoch_data)

    # ---------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "is_training": self.is_training,
                "current_epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "progress": self.progress,
                "session_id": self.session_id,
                "stop_requested": self.stop_requested,
                "last_error": self.last_error,
                "metrics": self.metrics.copy(),
                "history": list(self.history),
                "log": list(self.log),
            }


###############################################################################
class DatasetSelection(BaseModel):
    source: Literal["nist", "uploaded"]
    dataset_name: str = Field(min_length=1)


###############################################################################
class DatasetBuildRequest(BaseModel):
    sample_size: float = Field(default=1.0, ge=0.01, le=1.0)
    validation_size: float = Field(default=0.2, ge=0.05, le=0.5)
    min_measurements: int = Field(default=1, ge=1, le=100)
    max_measurements: int = Field(default=30, ge=5, le=500)
    smile_sequence_size: int = Field(default=20, ge=5, le=100)
    max_pressure: float = Field(default=10000.0, ge=100.0, le=100000.0)
    max_uptake: float = Field(default=20.0, ge=1.0, le=1000.0)
    reference_checkpoint: str | None = None
    datasets: list[DatasetSelection] = Field(default_factory=list, min_length=1)
    dataset_label: str = Field(default="default", min_length=1, max_length=64)


###############################################################################
class DatasetSourceInfo(BaseModel):
    source: Literal["nist", "uploaded"]
    dataset_name: str
    display_name: str
    row_count: int


###############################################################################
class DatasetSourcesResponse(BaseModel):
    datasets: list[DatasetSourceInfo]


###############################################################################
class DatasetBuildResponse(BaseModel):
    success: bool
    message: str
    total_samples: int | None = None
    train_samples: int | None = None
    validation_samples: int | None = None


###############################################################################
class DatasetInfoResponse(BaseModel):
    available: bool
    dataset_label: str | None = None
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
    smile_vocabulary_size: int | None = None
    adsorbent_vocabulary_size: int | None = None
    normalization_stats: dict[str, Any] | None = None


###############################################################################
class ProcessedDatasetInfo(BaseModel):
    dataset_label: str
    dataset_hash: str | None = None
    train_samples: int
    validation_samples: int
    created_at: str | None = None


###############################################################################
class ProcessedDatasetsResponse(BaseModel):
    datasets: list[ProcessedDatasetInfo]


###############################################################################
class TrainingMetadata(BaseModel):
    created_at: str | None = None
    sample_size: float = 1.0
    validation_size: float = 0.2
    min_measurements: int = 1
    max_measurements: int = 30
    smile_sequence_size: int = 20
    max_pressure: float = 10000.0
    max_uptake: float = 20.0
    total_samples: int = 0
    train_samples: int = 0
    validation_samples: int = 0

    # Vocabularies
    smile_vocabulary: dict[str, int] = Field(default_factory=dict)
    adsorbent_vocabulary: dict[str, int] = Field(default_factory=dict)

    # Statistics
    normalization_stats: dict[str, list[float] | float | dict] = Field(
        default_factory=dict
    )
    # Also support 'normalization' key logic via this field or separate
    normalization: dict[str, list[float] | float | dict] = Field(default_factory=dict)

    # Integrity Check
    dataset_hash: str | None = None

    # Computed/Derived fields that might be stored
    smile_vocabulary_size: int = 0
    adsorbent_vocabulary_size: int = 0

    # Legacy/Frontend fields (optional, can be aliases)
    SMILE_sequence_size: int | None = None
    SMILE_vocabulary: dict[str, int] | None = None
    SMILE_vocabulary_size: int | None = None

    model_config = {"populate_by_name": True, "extra": "ignore"}
