from __future__ import annotations

from fastapi import APIRouter, FastAPI, Request

from adsmod_common.capabilities import CapabilitiesResponse, FeatureCapabilities, ServiceCapability
from adsmod_common.config import AdsmodConfig
from adsmod_common.health import HealthResponse
from adsmod_common.version import __version__

###############################################################################
def liveness() -> HealthResponse:
    return HealthResponse(service="ml", version=__version__, state="ready")

###############################################################################
def readiness() -> HealthResponse:
    return HealthResponse(service="ml", version=__version__, state="ready")

###############################################################################
def capabilities(request: Request) -> CapabilitiesResponse:
    config: AdsmodConfig = request.app.state.config
    return CapabilitiesResponse(
        configured_mode=config.runtime.mode,
        version=__version__,
        features=FeatureCapabilities(
            datasets=False,
            nist=False,
            fitting=False,
            training=True,
            checkpoints=True,
        ),
        services={
            "ml": ServiceCapability(
                configured=True,
                health="ready",
                readiness="ready",
                version=__version__,
            )
        },
    )

###############################################################################
def _build_router() -> APIRouter:
    router = APIRouter()
    router.add_api_route(
        "/health/live", liveness, methods=["GET"], response_model=HealthResponse, tags=["health"]
    )
    router.add_api_route(
        "/health/ready", readiness, methods=["GET"], response_model=HealthResponse, tags=["health"]
    )
    router.add_api_route(
        "/api/v1/system/capabilities",
        capabilities,
        methods=["GET"],
        response_model=CapabilitiesResponse,
        tags=["system"],
    )
    return router

###############################################################################
def create_app(config: AdsmodConfig) -> FastAPI:
    application = FastAPI(title="ADSMOD ML Service", version=__version__)
    application.state.config = config
    application.include_router(_build_router())
    return application
