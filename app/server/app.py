from __future__ import annotations

import os
import sys
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent
APP_DIR = SERVER_DIR.parent
CLIENT_DIST_PATH = APP_DIR / "client" / "dist"
CLIENT_INDEX_FILE_PATH = CLIENT_DIST_PATH / "index.html"
CLIENT_ASSETS_PATH = CLIENT_DIST_PATH / "assets"
TRUTHY_VALUES = {"1", "true", "yes", "on"}

for package_dir in (
    SERVER_DIR / "shared",
    SERVER_DIR / "core_service",
    SERVER_DIR / "ml_service",
):
    resolved_package_dir = str(package_dir)
    if package_dir.is_dir() and resolved_package_dir not in sys.path:
        sys.path.insert(0, resolved_package_dir)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from shared.common.env import load_environment
from shared.common.settings import ServerSettings, get_server_settings
from shared.repositories.database.initializer import initialize_database

load_environment()
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore", category=FutureWarning)

from core_service.api.datasets import router as dataset_router
from core_service.api.entrypoint import health_router
from core_service.api.fitting import router as fitting_router
from core_service.api.nist import router as nist_router
from ml_service.api.training import router as training_router
from shared.common.constants import (
    CHECKPOINTS_PATH,
    FASTAPI_DESCRIPTION,
    FASTAPI_TITLE,
    FASTAPI_VERSION,
    LOGS_PATH,
    RESOURCES_PATH,
    TEMPLATES_PATH,
)


def _client_build_available() -> bool:
    return CLIENT_INDEX_FILE_PATH.is_file()


def _resolve_client_file(full_path: str) -> Path | None:
    client_root = CLIENT_DIST_PATH.resolve()
    requested_path = (client_root / full_path).resolve()

    if not requested_path.is_relative_to(client_root):
        return None
    if requested_path.is_file():
        return requested_path
    return None


def _build_cors_origins() -> list[str]:
    hosts = {"127.0.0.1", "localhost"}
    ui_host = os.getenv("UI_HOST", "").strip()
    if ui_host:
        hosts.add(ui_host)
        if ui_host == "127.0.0.1":
            hosts.add("localhost")
        elif ui_host == "localhost":
            hosts.add("127.0.0.1")

    ports = {5173, 5174}
    ui_port = os.getenv("UI_PORT", "").strip()
    if ui_port.isdigit():
        ports.add(int(ui_port))

    return sorted(f"http://{host}:{port}" for host in hosts for port in ports)


def _tauri_mode_enabled() -> bool:
    return os.getenv("ADSMOD_TAURI_MODE", "false").strip().lower() in TRUTHY_VALUES


def _ensure_runtime_directories() -> None:
    for path_value in (RESOURCES_PATH, LOGS_PATH, TEMPLATES_PATH, CHECKPOINTS_PATH):
        Path(path_value).mkdir(parents=True, exist_ok=True)


def _run_startup_validations(settings: ServerSettings) -> None:
    _ = settings
    _ensure_runtime_directories()

    if _tauri_mode_enabled() and not _client_build_available():
        raise RuntimeError(
            "Tauri mode requires a built frontend bundle at "
            f"{CLIENT_INDEX_FILE_PATH}."
        )


def serve_client_root() -> FileResponse:
    return FileResponse(CLIENT_INDEX_FILE_PATH)


def serve_client_path(full_path: str) -> FileResponse:
    client_file = _resolve_client_file(full_path)
    if client_file is not None:
        return FileResponse(client_file)
    return FileResponse(CLIENT_INDEX_FILE_PATH)


def redirect_root_to_docs() -> RedirectResponse:
    return RedirectResponse("/docs")


@asynccontextmanager
async def app_lifespan(application: FastAPI) -> AsyncIterator[None]:
    settings = get_server_settings()
    _run_startup_validations(settings)
    initialize_database()
    application.state.server_settings = settings
    yield


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

    application.include_router(health_router)
    application.include_router(dataset_router, prefix="/api", include_in_schema=False)
    application.include_router(fitting_router, prefix="/api", include_in_schema=False)
    application.include_router(nist_router, prefix="/api", include_in_schema=False)
    application.include_router(training_router, prefix="/api", include_in_schema=True)

    if _client_build_available():
        if CLIENT_ASSETS_PATH.is_dir():
            application.mount(
                "/assets",
                StaticFiles(directory=str(CLIENT_ASSETS_PATH)),
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
