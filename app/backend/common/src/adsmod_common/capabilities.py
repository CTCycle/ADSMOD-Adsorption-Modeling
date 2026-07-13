from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

CapabilityState = Literal["ready", "starting", "not-ready", "failed", "unavailable", "not-configured"]

###############################################################################
class FeatureCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    datasets: bool
    nist: bool
    fitting: bool
    training: bool
    checkpoints: bool

###############################################################################
class ServiceCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    configured: bool
    health: CapabilityState
    readiness: CapabilityState
    version: str | None = None
    reason: str | None = None

###############################################################################
class CapabilitiesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    configured_mode: Literal["core", "core-ml"]
    version: str
    features: FeatureCapabilities
    services: dict[str, ServiceCapability]


__all__ = ["CapabilitiesResponse", "FeatureCapabilities", "ServiceCapability"]
