from __future__ import annotations

import os
import warnings
from pathlib import Path

from fastapi import FastAPI

os.environ.setdefault(
    "ADSMOD_CONFIG_PATH",
    str(Path(__file__).resolve().parents[4] / "settings" / "core_service.json")
)

from core_service.api.entrypoint import health_router, register_root_routes
from core_service.api.routes import register_core_routes
from core_service.common.constants import (
    FASTAPI_DESCRIPTION,
    FASTAPI_TITLE,
    FASTAPI_VERSION,
)
from core_service.configurations.startup import (
    public_host_mode_enabled,
    resolve_spa_file_path,
)
from core_service.services.container import CoreServiceContainer

warnings.filterwarnings("ignore", category=FutureWarning)

PUBLIC_HOST_MODE = public_host_mode_enabled()

def create_app(container: CoreServiceContainer | None = None) -> FastAPI:
    application = FastAPI(
        title=FASTAPI_TITLE,
        version=FASTAPI_VERSION,
        description=FASTAPI_DESCRIPTION,
        docs_url=None if PUBLIC_HOST_MODE else "/docs",
        redoc_url=None if PUBLIC_HOST_MODE else "/redoc",
        openapi_url=None if PUBLIC_HOST_MODE else "/openapi.json",
    )
    resolved_container = container or CoreServiceContainer()
    application.state.container = resolved_container
    application.include_router(health_router)
    register_core_routes(application, resolved_container)
    register_root_routes(application)
    return application


app = create_app()

__all__ = [
    "app",
    "create_app",
    "PUBLIC_HOST_MODE",
    "public_host_mode_enabled",
    "resolve_spa_file_path",
]
