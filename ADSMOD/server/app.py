from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from fastapi import FastAPI

from ADSMOD.server.common.constants import (
    FASTAPI_DESCRIPTION,
    FASTAPI_TITLE,
    FASTAPI_VERSION,
)
from ADSMOD.server.common.utils.variables import env_variables  # noqa: F401
from ADSMOD.server.configurations.runtime import (
    direct_api_enabled,
    public_host_mode_enabled,
)
from ADSMOD.server.api.datasets import router as dataset_router
from ADSMOD.server.api.entrypoint import health_router, register_root_routes
from ADSMOD.server.api.fitting import router as fit_router
from ADSMOD.server.api.nist import router as nist_router
from ADSMOD.server.api.training import router as training_router


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
    health_router,
    dataset_router,
    fit_router,
    training_router,
    nist_router,
]

for router in routers:
    if router is health_router or direct_api_enabled():
        app.include_router(router)
    if router is not health_router:
        app.include_router(router, prefix="/api", include_in_schema=False)

register_root_routes(app)

