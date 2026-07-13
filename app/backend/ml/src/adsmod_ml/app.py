from __future__ import annotations

from fastapi import FastAPI

from adsmod_common.capabilities import CapabilitiesResponse, FeatureCapabilities, ServiceCapability
from adsmod_common.config import AdsmodConfig
from adsmod_common.health import HealthResponse
from adsmod_common.version import __version__

###############################################################################
def create_app(config: AdsmodConfig) -> FastAPI:
    application = FastAPI(title="ADSMOD ML Service", version=__version__)
    application.state.config = config

    @application.get("/health/live", response_model=HealthResponse, tags=["health"])
    def liveness() -> HealthResponse:
        return HealthResponse(service="ml", version=__version__, state="ready")

    @application.get("/health/ready", response_model=HealthResponse, tags=["health"])
    def readiness() -> HealthResponse:
        return HealthResponse(service="ml", version=__version__, state="ready")

    @application.get("/api/v1/system/capabilities", response_model=CapabilitiesResponse, tags=["system"])
    def capabilities() -> CapabilitiesResponse:
        return CapabilitiesResponse(
            configured_mode=config.runtime.mode,
            version=__version__,
            features=FeatureCapabilities(datasets=False, nist=False, fitting=False, training=True, checkpoints=True),
            services={"ml": ServiceCapability(configured=True, health="ready", readiness="ready", version=__version__)},
        )

    return application