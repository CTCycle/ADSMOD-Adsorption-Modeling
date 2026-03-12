from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from ADSMOD.server.common.constants import (
    FASTAPI_DESCRIPTION,
    FASTAPI_TITLE,
    FASTAPI_VERSION,
)
from ADSMOD.server.common.utils.variables import env_variables  # noqa: F401
from ADSMOD.server.routes.datasets import router as dataset_router
from ADSMOD.server.routes.fitting import router as fit_router
from ADSMOD.server.routes.training import router as training_router
from ADSMOD.server.routes.nist import router as nist_router


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def public_host_mode_enabled() -> bool:
    host = os.getenv("FASTAPI_HOST", "").strip().lower()
    if not host:
        return False
    return host not in LOOPBACK_HOSTS


def direct_api_enabled() -> bool:
    return not public_host_mode_enabled()


def tauri_mode_enabled() -> bool:
    value = os.getenv("ADSMOD_TAURI_MODE", "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def get_client_dist_path() -> str:
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(project_path, "client", "dist")


def packaged_client_available() -> bool:
    return tauri_mode_enabled() and os.path.isdir(get_client_dist_path())


def resolve_spa_file_path(client_dist_path: str, requested_path: str) -> str | None:
    normalized_path = str(requested_path or "").lstrip("/\\")
    absolute_root = os.path.abspath(client_dist_path)
    candidate = os.path.abspath(os.path.join(absolute_root, normalized_path))
    if os.path.commonpath([absolute_root, candidate]) != absolute_root:
        return None
    if not os.path.isfile(candidate):
        return None
    return candidate


PUBLIC_HOST_MODE = public_host_mode_enabled()


###############################################################################
app = FastAPI(
    title=FASTAPI_TITLE,
    version=FASTAPI_VERSION,
    description=FASTAPI_DESCRIPTION,
    docs_url=None if PUBLIC_HOST_MODE else "/docs",
    redoc_url=None if PUBLIC_HOST_MODE else "/redoc",
    openapi_url=None if PUBLIC_HOST_MODE else "/openapi.json",
)

routers = [
    dataset_router,
    fit_router,
    training_router,
    nist_router,
]

for router in routers:
    if direct_api_enabled():
        app.include_router(router)
    app.include_router(router, prefix="/api", include_in_schema=False)


@app.get("/api/health", include_in_schema=False)
def health_check() -> dict[str, str]:
    return {"status": "ok"}


if packaged_client_available():
    client_dist_path = get_client_dist_path()
    assets_path = os.path.join(client_dist_path, "assets")

    if os.path.isdir(assets_path):
        app.mount("/assets", StaticFiles(directory=assets_path), name="spa-assets")

    @app.get("/", include_in_schema=False)
    def serve_spa_root() -> FileResponse:
        return FileResponse(os.path.join(client_dist_path, "index.html"))

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa_entrypoint(full_path: str) -> FileResponse:
        requested_path = resolve_spa_file_path(client_dist_path, full_path)
        if requested_path is not None:
            return FileResponse(requested_path)
        return FileResponse(os.path.join(client_dist_path, "index.html"))

else:

    if direct_api_enabled():

        @app.get("/")
        def redirect_to_docs() -> RedirectResponse:
            return RedirectResponse(url="/docs")

    else:

        @app.get("/", include_in_schema=False)
        def service_root() -> dict[str, str]:
            return {"status": "ok"}

