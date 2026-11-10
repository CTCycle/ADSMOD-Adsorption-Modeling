# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DeepLearningStatusName = Literal[
    "not_installed", "installing", "verifying", "installed", "failed", "incompatible"
]
DeepLearningAction = Literal["install", "uninstall", "repair"]


###############################################################################
class DeepLearningError(BaseModel):
    code: str
    summary: str
    hint: str


###############################################################################
class DeepLearningOperation(BaseModel):
    id: str
    kind: DeepLearningAction
    status: Literal["running", "succeeded", "failed"]
    phase: str | None = None
    started_at: str
    finished_at: str | None = None
    error: DeepLearningError | None = None
    warnings: list[str] = Field(default_factory=list)
    restart_healthy: bool | None = None


###############################################################################
class DeepLearningEnvironment(BaseModel):
    python_version: str
    platform: str
    torch_version: str | None = None
    keras_version: str | None = None
    scikit_learn_version: str | None = None
    cuda_runtime: str | None = None
    cuda_available: bool | None = None
    devices: list[str] = Field(default_factory=list)
    gpu_names: list[str] = Field(default_factory=list)
    driver_version: str | None = None
    driver_cuda_level: str | None = None


###############################################################################
class DeepLearningStatusResponse(BaseModel):
    status: DeepLearningStatusName
    available: bool
    reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    environment: DeepLearningEnvironment
    operation: DeepLearningOperation | None = None
    last_validated_at: str | None = None
    log_path: str


###############################################################################
class DeepLearningCheck(BaseModel):
    name: str
    status: Literal["ok", "warning", "blocked"]
    detail: str


###############################################################################
class DeepLearningPlanResponse(BaseModel):
    action: DeepLearningAction
    allowed: bool
    checks: list[DeepLearningCheck]
    packages: list[str]
    download_bytes: int | None = None
    download_estimated: bool = False
    requires_restart: bool = True


###############################################################################
class DeepLearningOperationRequest(BaseModel):
    action: DeepLearningAction
    confirm: bool = False


###############################################################################
class DeepLearningOperationAccepted(BaseModel):
    operation_id: str
    action: DeepLearningAction


###############################################################################
class DeepLearningLogResponse(BaseModel):
    lines: list[str]


__all__ = [
    "DeepLearningAction",
    "DeepLearningCheck",
    "DeepLearningEnvironment",
    "DeepLearningError",
    "DeepLearningLogResponse",
    "DeepLearningOperation",
    "DeepLearningOperationAccepted",
    "DeepLearningOperationRequest",
    "DeepLearningPlanResponse",
    "DeepLearningStatusName",
    "DeepLearningStatusResponse",
]
