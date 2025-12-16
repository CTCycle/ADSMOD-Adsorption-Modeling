from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


###############################################################################
class DatasetPayload(BaseModel):
    columns: list[str] = Field(default_factory=list)
    records: list[dict[str, Any]] = Field(default_factory=list)


###############################################################################
class ModelParameterConfig(BaseModel):
    min: dict[str, float] = Field(default_factory=dict)
    max: dict[str, float] = Field(default_factory=dict)
    initial: dict[str, float] = Field(default_factory=dict)


###############################################################################
class FittingRequest(BaseModel):
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
