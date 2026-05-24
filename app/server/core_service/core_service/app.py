from __future__ import annotations

import warnings

from fastapi import FastAPI

from core_service.common.constants import (
    FASTAPI_DESCRIPTION,
    FASTAPI_TITLE,
    FASTAPI_VERSION,
)
from core_service.configurations.startup import (
    public_host_mode_enabled,
    resolve_spa_file_path,
)
from core_service.api.datasets import router as dataset_router
from core_service.api.entrypoint import health_router, register_root_routes
from core_service.api.fitting import router as fit_router
from core_service.api.nist import router as nist_router

warnings.filterwarnings("ignore", category=FutureWarning)

PUBLIC_HOST_MODE = public_host_mode_enabled()

app = FastAPI(
    title=FASTAPI_TITLE,
    version=FASTAPI_VERSION,
    description=FASTAPI_DESCRIPTION,
    docs_url=None if PUBLIC_HOST_MODE else "/docs",
    redoc_url=None if PUBLIC_HOST_MODE else "/redoc",
    openapi_url=None if PUBLIC_HOST_MODE else "/openapi.json",
)

routers = [
    health_router,
    dataset_router,
    fit_router,
    nist_router,
]

for router in routers:
    if router is health_router:
        app.include_router(router)
    else:
        app.include_router(router, prefix="/api", include_in_schema=False)

register_root_routes(app)

__all__ = [
    "app",
    "PUBLIC_HOST_MODE",
    "public_host_mode_enabled",
    "resolve_spa_file_path",
]
