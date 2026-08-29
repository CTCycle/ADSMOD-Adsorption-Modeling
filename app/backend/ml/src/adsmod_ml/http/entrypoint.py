from __future__ import annotations

from fastapi import APIRouter, Request

from adsmod_common.health import HealthResponse
from adsmod_common.version import __version__

health_router = APIRouter()


@health_router.get("/health/live", response_model=HealthResponse, tags=["health"])
def liveness() -> HealthResponse:
    return HealthResponse(service="ml", version=__version__, state="ready")


@health_router.get("/health/ready", response_model=HealthResponse, tags=["health"])
def readiness(request: Request) -> HealthResponse:
    ready = bool(getattr(request.app.state, "ready", False))
    return HealthResponse(
        service="ml",
        version=__version__,
        state="ready" if ready else "not-ready",
        details={} if ready else {"reason": "ML service initialization is pending"},
    )


__all__ = ["health_router", "liveness", "readiness"]
