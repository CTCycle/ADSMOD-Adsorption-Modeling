from __future__ import annotations

from fastapi import APIRouter

from ml_service.domain.bootstrap import ServiceStatusResponse

health_router = APIRouter()


@health_router.get(
    "/api/health",
    include_in_schema=True,
    response_model=ServiceStatusResponse,
)
def health_check() -> ServiceStatusResponse:
    return ServiceStatusResponse(status="ok")
