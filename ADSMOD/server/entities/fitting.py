from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


###############################################################################
class DatasetPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    dataset_name: str = Field(..., min_length=1, max_length=256)
    columns: list[str] = Field(default_factory=list, max_length=256)
    records: list[dict[str, Any]] = Field(default_factory=list)


###############################################################################
class ModelParameterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: dict[str, float] = Field(default_factory=dict)
    max: dict[str, float] = Field(default_factory=dict)
    initial: dict[str, float] = Field(default_factory=dict)


###############################################################################
class FittingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_iterations: int = Field(..., ge=1)
    optimization_method: Literal[
        "LSS",
        "BFGS",
        "L-BFGS-B",
        "Nelder-Mead",
        "Powell",
    ] = Field(default="LSS")
    parameter_bounds: dict[str, ModelParameterConfig]
    dataset: DatasetPayload


###############################################################################
class FittingResponse(BaseModel):
    status: str = Field(default="success")
    summary: str
    processed_rows: int
    models: list[str]
    best_model_saved: bool
    best_model_preview: list[dict[str, Any]] | None = None
