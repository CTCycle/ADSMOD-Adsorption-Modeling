# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

###############################################################################
class ParameterBounds(BaseModel):
    """Canonical-unit (Pa, mol/kg) search bounds for one model parameter."""

    model_config = ConfigDict(extra="forbid")

    lower: float
    upper: float

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_order(self) -> ParameterBounds:
        if not (math.isfinite(self.lower) and math.isfinite(self.upper)):
            raise ValueError("Parameter bounds must be finite numbers.")
        if self.lower >= self.upper:
            raise ValueError("Parameter lower bound must be smaller than upper bound.")
        return self

###############################################################################
class DisplayUnits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pressure: str = "bar"
    uptake: str = "mmol/g"

###############################################################################
class FittingSessionRequest(BaseModel):
    """Start a fitting session over every experiment of one dataset."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    dataset_id: int = Field(ge=1)
    models: list[str] = Field(min_length=1, max_length=32)
    optimizer: Literal["trf", "dogbox"] = "trf"
    max_evaluations: int = Field(default=10_000, ge=10)
    weighting: Literal["unweighted", "inverse_sigma"] = "unweighted"
    parameter_overrides: dict[str, dict[str, ParameterBounds]] = Field(
        default_factory=dict
    )
    display_units: DisplayUnits = Field(default_factory=DisplayUnits)
    skip_ineligible: bool = False

    # -------------------------------------------------------------------------
    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Session name cannot be empty.")
        return cleaned

###############################################################################
class ExperimentFitRequest(BaseModel):
    """Resolved per-experiment fitting request built from a session snapshot."""

    model_config = ConfigDict(extra="forbid")

    models: list[str]
    optimizer: Literal["trf", "dogbox"]
    max_evaluations: int
    weighting: Literal["unweighted", "inverse_sigma"]
    model_bounds: dict[str, dict[str, ParameterBounds]]

###############################################################################
class ModelParameterDefinition(BaseModel):
    name: str
    label: str
    lower: float
    upper: float
    canonical_lower: float
    canonical_upper: float
    unit: str
    customized: bool

###############################################################################
class ModelDefinition(BaseModel):
    key: str
    name: str
    equation_latex: str
    assumptions: str
    pressure_requirement: str
    requires_temperature: bool
    reference: str
    enabled: bool
    parameters: list[ModelParameterDefinition]

###############################################################################
class ModelCatalogResponse(BaseModel):
    status: Literal["success"] = "success"
    models: list[ModelDefinition]

###############################################################################
class ModelSettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    parameters: dict[str, ParameterBounds] = Field(default_factory=dict)

###############################################################################
class FittedParameter(BaseModel):
    name: str
    label: str
    value: float
    standard_error: float | None
    ci95_low: float | None
    ci95_high: float | None
    unit: str

###############################################################################
class FitMetrics(BaseModel):
    sse: float | None
    rmse: float | None
    mae: float | None
    r_squared: float | None
    adjusted_r_squared: float | None
    chi_square: float | None
    aic: float | None
    aicc: float | None
    bic: float | None

###############################################################################
class SessionModelResult(BaseModel):
    model: str
    name: str
    status: Literal["success", "warning", "failed"]
    convergence_message: str
    rank: int | None
    observation_count: int
    parameters: list[FittedParameter]
    metrics: FitMetrics
    warnings: list[str]

###############################################################################
class ExperimentFitResponse(BaseModel):
    isotherm_id: int | None
    experiment_name: str
    pressure_unit: str
    uptake_unit: str
    observation_count: int
    best_model: str | None
    results: list[SessionModelResult]

###############################################################################
class FittingExperimentSummary(BaseModel):
    isotherm_id: int
    name: str
    eligible: bool
    reason: str | None = None

###############################################################################
class FittingSessionSummary(BaseModel):
    id: int
    name: str
    dataset_id: int
    dataset_name: str
    status: Literal["running", "completed", "warning", "failed", "cancelled"]
    message: str
    models: list[str]
    experiment_count: int
    optimizer: str
    weighting: str
    max_evaluations: int
    created_at: str
    completed_at: str | None
    duration_seconds: float | None
    best_models: dict[str, str | None] = Field(default_factory=dict)
    stale: bool = False

###############################################################################
class FittingSessionListResponse(BaseModel):
    status: Literal["success"] = "success"
    items: list[FittingSessionSummary]
    total: int
    page: int
    page_size: int

###############################################################################
class FittingSessionDetail(FittingSessionSummary):
    configuration: dict[str, Any]
    log: list[str]
    experiments: list[ExperimentFitResponse]
    job_id: str | None = None
    stale_reason: str | None = None

###############################################################################
class FittingSessionStartResponse(BaseModel):
    status: Literal["success"] = "success"
    session_id: int
    job_id: str
    poll_interval: float

###############################################################################
class FittingDatasetReadiness(BaseModel):
    status: Literal["success"] = "success"
    dataset_id: int
    dataset_name: str
    experiment_count: int
    eligible_count: int
    ready: bool
    inverse_sigma_available: bool
    model_incompatibility: dict[str, str]
    experiments: list[FittingExperimentSummary]
