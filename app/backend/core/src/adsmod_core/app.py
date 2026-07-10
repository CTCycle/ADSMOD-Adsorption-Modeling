from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from adsmod_common.capabilities import CapabilitiesResponse, FeatureCapabilities, ServiceCapability
from adsmod_common.config import AdsmodConfig, load_config
from adsmod_common.health import HealthResponse
from adsmod_common.version import __version__


def _capabilities(config: AdsmodConfig) -> CapabilitiesResponse:
    ml_configured = config.runtime.mode == "core-ml"
    return CapabilitiesResponse(
        configured_mode=config.runtime.mode,
        version=__version__,
        features=FeatureCapabilities(
            datasets=True,
            nist=True,
            fitting=True,
            training=ml_configured,
            checkpoints=ml_configured,
        ),
        services={
            "core": ServiceCapability(
                configured=True,
                health="ready",
                readiness="ready",
                version=__version__,
            ),
            "ml": ServiceCapability(
                configured=ml_configured,
                health="starting" if ml_configured else "unavailable",
                readiness="not-ready" if ml_configured else "not-configured",
                reason=None if ml_configured else "Machine learning service is not configured.",
            ),
        },
    )


def create_app(config: AdsmodConfig) -> FastAPI:
    application = FastAPI(title="ADSMOD Core Service", version=__version__)
    application.state.config = config

    @application.get("/health/live", response_model=HealthResponse, tags=["health"])
    def liveness() -> HealthResponse:
        return HealthResponse(service="core", version=__version__, state="ready")

    @application.get("/health/ready", response_model=HealthResponse, tags=["health"])
    def readiness() -> HealthResponse:
        return HealthResponse(service="core", version=__version__, state="ready")

    @application.get(
        "/api/v1/system/capabilities",
        response_model=CapabilitiesResponse,
        tags=["system"],
    )
    def capabilities() -> CapabilitiesResponse:
        return _capabilities(config)

    return application


def create_app_from_path(config_path: str | Path) -> FastAPI:
    return create_app(load_config(config_path))


__all__ = ["create_app", "create_app_from_path"]