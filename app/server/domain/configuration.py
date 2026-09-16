from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

###############################################################################
class NumericBounds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum: int | float
    maximum: int | float

###############################################################################
class ParameterDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lower: float
    upper: float
    initial: float

###############################################################################
class DisplayUnitCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pressure: tuple[str, ...]
    uptake: tuple[str, ...]
    default_pressure: str
    default_uptake: str

###############################################################################
class FittingConfigurationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["success"] = "success"
    supported_optimizers: tuple[str, ...]
    default_optimizer: str
    default_max_evaluations: int
    max_evaluations_bounds: NumericBounds
    weighting_options: tuple[str, ...]
    default_weighting: str
    display_units: DisplayUnitCapabilities
    parameter_defaults: ParameterDefaults


###############################################################################
class RuntimeDeviceCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    keras_backend: str
    cuda_available: bool
    device_count: int
    devices: tuple[str, ...]


###############################################################################
class TrainingConfigurationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["success"] = "success"
    defaults: dict[str, Any]
    dataset_defaults: dict[str, Any]
    resume_defaults: dict[str, Any]
    numeric_constraints: dict[str, dict[str, int | float]]
    supported_models: tuple[str, ...]
    checkpoint_capabilities: dict[str, bool]
    runtime: RuntimeDeviceCapabilities


__all__ = [
    "DisplayUnitCapabilities",
    "FittingConfigurationResponse",
    "NumericBounds",
    "ParameterDefaults",
    "RuntimeDeviceCapabilities",
    "TrainingConfigurationResponse",
]
