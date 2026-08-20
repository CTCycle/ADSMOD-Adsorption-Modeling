from __future__ import annotations

from fastapi import APIRouter, FastAPI
from fastapi.responses import RedirectResponse

from core_service.configurations.startup import public_host_mode_enabled
from core_service.contracts.bootstrap import ServiceStatusResponse

health_router = APIRouter()

###############################################################################
@health_router.get(
    "/api/health",
    include_in_schema=False,
    response_model=ServiceStatusResponse,
)
def health_check() -> ServiceStatusResponse:
    return ServiceStatusResponse(status="ok")

###############################################################################
def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url="/docs")

###############################################################################
def service_root() -> ServiceStatusResponse:
    return ServiceStatusResponse(status="ok")

###############################################################################
def register_root_routes(app: FastAPI) -> None:
    if not public_host_mode_enabled():
        app.add_api_route("/", redirect_to_docs, methods=["GET"])
        return

    app.add_api_route(
        "/",
        service_root,
        methods=["GET"],
        include_in_schema=False,
        response_model=ServiceStatusResponse,
    )


