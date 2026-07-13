from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from shared.common.constants import DEFAULT_DATASET_COLUMN_MAPPING

###############################################################################
@dataclass
class DatasetColumns:
    experiment: str = DEFAULT_DATASET_COLUMN_MAPPING["experiment"]
    temperature: str = DEFAULT_DATASET_COLUMN_MAPPING["temperature"]
    pressure: str = DEFAULT_DATASET_COLUMN_MAPPING["pressure"]
    uptake: str = DEFAULT_DATASET_COLUMN_MAPPING["uptake"]

    # -------------------------------------------------------------------------
    def as_dict(self) -> dict[str, str]:
        return {"experiment": self.experiment, "temperature": self.temperature, "pressure": self.pressure, "uptake": self.uptake}

###############################################################################
class DatasetMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tags: list[str] = Field(default_factory=list, max_length=32)
    description: str = Field(default="", max_length=2000)

###############################################################################
class DatasetSummary(BaseModel):
    name: str
    source: Literal["uploaded"] = "uploaded"
    created_at: str
    row_count: int
    column_count: int
    tags: list[str] = Field(default_factory=list)
    description: str = ""

###############################################################################
class DatasetListResponse(BaseModel):
    status: str = "success"
    datasets: list[DatasetSummary] = Field(default_factory=list)

###############################################################################
class DatasetUploadResponse(BaseModel):
    status: str = "success"
    dataset: DatasetSummary
    summary: str

###############################################################################
class DatasetRenameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    new_name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_. -]+$")

###############################################################################
class DatasetRowsPage(BaseModel):
    status: str = "success"
    dataset_name: str
    columns: list[str]
    rows: list[dict[str, Any]]
    offset: int
    limit: int
    total_rows: int

###############################################################################
class DatasetRowMutation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["insert", "update", "delete"]
    row_id: int | None = Field(default=None, ge=0)
    values: dict[str, Any] = Field(default_factory=dict)

###############################################################################
class DatasetRowsMutationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operations: list[DatasetRowMutation] = Field(min_length=1, max_length=500)

###############################################################################
class DatasetMutationResponse(BaseModel):
    status: str = "success"
    dataset: DatasetSummary