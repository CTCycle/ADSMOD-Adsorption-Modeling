from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


MAX_DATASET_COLUMNS = 256
MAX_DATASET_RECORDS = 200_000
MAX_DATASET_RECORD_COLUMNS = 256
MAX_COLUMN_NAME_LENGTH = 128
MAX_DATASET_NAME_LENGTH = 128
MAX_FITTING_ITERATIONS = 1_000_000


###############################################################################
class DatasetPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    dataset_name: str = Field(
        ...,
        min_length=1,
        max_length=MAX_DATASET_NAME_LENGTH,
        pattern=r"^[A-Za-z0-9_. -]+$",
    )
    columns: list[str] = Field(default_factory=list, max_length=MAX_DATASET_COLUMNS)
    records: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=MAX_DATASET_RECORDS,
    )

    # -------------------------------------------------------------------------
    @field_validator("columns")
    @classmethod
    def validate_columns(cls, values: list[str]) -> list[str]:
        for column in values:
            if not isinstance(column, str) or not column.strip():
                raise ValueError("Dataset columns must be non-empty strings.")
            if len(column.strip()) > MAX_COLUMN_NAME_LENGTH:
                raise ValueError(
                    f"Dataset column names must be <= {MAX_COLUMN_NAME_LENGTH} characters."
                )
        return values

    # -------------------------------------------------------------------------
    @field_validator("records")
    @classmethod
    def validate_records(cls, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for record in records:
            if len(record) > MAX_DATASET_RECORD_COLUMNS:
                raise ValueError(
                    f"Dataset records cannot contain more than {MAX_DATASET_RECORD_COLUMNS} columns."
                )
            for key in record:
                key_text = str(key).strip()
                if not key_text:
                    raise ValueError("Dataset record keys must be non-empty strings.")
                if len(key_text) > MAX_COLUMN_NAME_LENGTH:
                    raise ValueError(
                        f"Dataset record keys must be <= {MAX_COLUMN_NAME_LENGTH} characters."
                    )
        return records


###############################################################################
class ModelParameterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min: dict[str, float] = Field(default_factory=dict)
    max: dict[str, float] = Field(default_factory=dict)
    initial: dict[str, float] = Field(default_factory=dict)


###############################################################################
class FittingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_iterations: int = Field(..., ge=1, le=MAX_FITTING_ITERATIONS)
    optimization_method: Literal[
        "LSS",
        "BFGS",
        "L-BFGS-B",
        "Nelder-Mead",
        "Powell",
    ] = Field(default="LSS")
    parameter_bounds: dict[str, ModelParameterConfig] = Field(default_factory=dict, max_length=32)
    dataset: DatasetPayload


###############################################################################
class FittingResponse(BaseModel):
    status: str = Field(default="success")
    summary: str
    processed_rows: int
    models: list[str]
    best_model_saved: bool
    best_model_preview: list[dict[str, Any]] | None = None


###############################################################################
class NISTFittingDatasetPayload(BaseModel):
    dataset_name: str
    columns: list[str]
    records: list[dict[str, Any]]
    row_count: int


###############################################################################
class NISTFittingDatasetResponse(BaseModel):
    status: str = Field(default="success")
    dataset: NISTFittingDatasetPayload
