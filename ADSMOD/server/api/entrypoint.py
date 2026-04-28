from __future__ import annotations

import os

from fastapi import APIRouter, FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from ADSMOD.server.configurations.startup import (
    direct_api_enabled,
    get_client_dist_path,
    packaged_client_available,
    resolve_spa_file_path,
)
from ADSMOD.server.domain.bootstrap import ServiceStatusResponse

health_router = APIRouter()


###############################################################################
@health_router.get(
    "/api/health",
    include_in_schema=False,
    response_model=ServiceStatusResponse,
)
def health_check() -> ServiceStatusResponse:
    return ServiceStatusResponse(status="ok")


# -----------------------------------------------------------------------------
def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url="/docs")


# -----------------------------------------------------------------------------
def service_root() -> ServiceStatusResponse:
    return ServiceStatusResponse(status="ok")


# -----------------------------------------------------------------------------
class SpaEntrypointHandlers:
    def __init__(self, client_dist_path: str) -> None:
        self.client_dist_path = client_dist_path

    # -------------------------------------------------------------------------
    def serve_spa_root(self) -> FileResponse:
        return FileResponse(os.path.join(self.client_dist_path, "index.html"))

    # -------------------------------------------------------------------------
    def serve_spa_entrypoint(self, full_path: str) -> FileResponse:
        requested_path = resolve_spa_file_path(self.client_dist_path, full_path)
        if requested_path is not None:
            return FileResponse(requested_path)
        return FileResponse(os.path.join(self.client_dist_path, "index.html"))


# -----------------------------------------------------------------------------
def register_root_routes(app: FastAPI) -> None:
    if packaged_client_available():
        client_dist_path = get_client_dist_path()
        assets_path = os.path.join(client_dist_path, "assets")
        handlers = SpaEntrypointHandlers(client_dist_path=client_dist_path)

        if os.path.isdir(assets_path):
            app.mount("/assets", StaticFiles(directory=assets_path), name="spa-assets")

        app.add_api_route(
            "/",
            handlers.serve_spa_root,
            methods=["GET"],
            include_in_schema=False,
        )
        app.add_api_route(
            "/{full_path:path}",
            handlers.serve_spa_entrypoint,
            methods=["GET"],
            include_in_schema=False,
        )
        return

    if direct_api_enabled():
        app.add_api_route("/", redirect_to_docs, methods=["GET"])
        return

    app.add_api_route(
        "/",
        service_root,
        methods=["GET"],
        include_in_schema=False,
        response_model=ServiceStatusResponse,
    )

