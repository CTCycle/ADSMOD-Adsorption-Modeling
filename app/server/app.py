from __future__ import annotations

import os
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from core_service.api.entrypoint import health_router
from core_service.api.routes import register_core_routes
from core_service.services.container import CoreServiceContainer
from shared.common.constants import (
    FASTAPI_DESCRIPTION,
    FASTAPI_TITLE,
    FASTAPI_VERSION,
)
from shared.common.paths import (
    CHECKPOINTS_DIR,
    CLIENT_ASSETS_DIR,
    CLIENT_DIST_DIR,
    CLIENT_INDEX_FILE,
    LOGS_DIR,
    RESOURCES_DIR,
    TEMPLATES_DIR,
)
from shared.common.settings import ServerSettings, get_runtime_config, get_server_settings
from shared.repositories.database.initializer import initialize_database

TRUTHY_VALUES = {"1", "true", "yes", "on"}

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore", category=FutureWarning)

###############################################################################
def _ml_registration_enabled() -> bool:
    return os.getenv("ADSMOD_ENABLE_ML", "false").strip().lower() in TRUTHY_VALUES

###############################################################################
def _load_ml_service_modules():
    from ml_service.api.routes import register_ml_routes
    from ml_service.services.container import MlServiceContainer

    return register_ml_routes, MlServiceContainer

###############################################################################
def _client_build_available() -> bool:
    return CLIENT_INDEX_FILE.is_file()

###############################################################################
def _resolve_client_file(full_path: str) -> Path | None:
    client_root = CLIENT_DIST_DIR.resolve()
    requested_path = (client_root / full_path).resolve()

    if not requested_path.is_relative_to(client_root):
        return None
    if requested_path.is_file():
        return requested_path
    return None

###############################################################################
def _build_cors_origins() -> list[str]:
    runtime = get_runtime_config()
    ui_host = str(runtime.get("host", "127.0.0.1"))
    hosts = {ui_host, "127.0.0.1", "localhost"}
    ports = {int(runtime["frontend_port"])}

    return sorted(f"http://{host}:{port}" for host in hosts for port in ports)

###############################################################################
def _ensure_runtime_directories() -> None:
    for path_value in (RESOURCES_DIR, LOGS_DIR, TEMPLATES_DIR, CHECKPOINTS_DIR):
        path_value.mkdir(parents=True, exist_ok=True)

###############################################################################
def _run_startup_validations(settings: ServerSettings) -> None:
    _ = settings
    _ensure_runtime_directories()

###############################################################################
def serve_client_root() -> FileResponse:
    return FileResponse(CLIENT_INDEX_FILE)

###############################################################################
def serve_client_path(full_path: str) -> FileResponse:
    client_file = _resolve_client_file(full_path)
    if client_file is not None:
        return FileResponse(client_file)
    return FileResponse(CLIENT_INDEX_FILE)

###############################################################################
def redirect_root_to_docs() -> RedirectResponse:
    return RedirectResponse("/docs")

###############################################################################
@asynccontextmanager
async def app_lifespan(application: FastAPI) -> AsyncIterator[None]:
    settings = get_server_settings()
    _run_startup_validations(settings)
    initialize_database()
    application.state.server_settings = settings
    yield

###############################################################################
def create_app() -> FastAPI:
    application = FastAPI(
        title=FASTAPI_TITLE,
        version=FASTAPI_VERSION,
        description=FASTAPI_DESCRIPTION,
        lifespan=app_lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=_build_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    core_container = CoreServiceContainer()
    application.state.core_container = core_container

    application.include_router(health_router)
    register_core_routes(application, core_container, prefix="/api", include_schema=False)
    if _ml_registration_enabled():
        register_ml_routes, MlServiceContainer = _load_ml_service_modules()
        ml_container = MlServiceContainer()
        application.state.ml_container = ml_container
        register_ml_routes(application, ml_container, prefix="/api", include_schema=True)

    if _client_build_available():
        if CLIENT_ASSETS_DIR.is_dir():
            application.mount(
                "/assets",
                StaticFiles(directory=str(CLIENT_ASSETS_DIR)),
                name="assets",
            )
        application.add_api_route(
            "/",
            serve_client_root,
            methods=["GET"],
            include_in_schema=False,
        )
        application.add_api_route(
            "/{full_path:path}",
            serve_client_path,
            methods=["GET"],
            include_in_schema=False,
        )
    else:
        application.add_api_route(
            "/",
            redirect_root_to_docs,
            methods=["GET"],
            include_in_schema=False,
        )

    return application


app = create_app()

__all__ = ["app", "create_app"]
