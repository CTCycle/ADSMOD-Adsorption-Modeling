from __future__ import annotations

import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ml_service.api.entrypoint import health_router
from ml_service.api.routes import register_ml_routes
from ml_service.services.container import MlServiceContainer
from shared.common.constants import (
    FASTAPI_DESCRIPTION,
    FASTAPI_TITLE,
    FASTAPI_VERSION,
)
from shared.common.settings import get_server_settings
from shared.repositories.database.initializer import prepare_database_for_startup

warnings.filterwarnings("ignore", category=FutureWarning)

###############################################################################
@asynccontextmanager
async def app_lifespan(application: FastAPI) -> AsyncIterator[None]:
    settings = get_server_settings()
    prepare_database_for_startup(settings.database)
    application.state.server_settings = settings
    yield

###############################################################################
def create_app(container: MlServiceContainer | None = None) -> FastAPI:
    application = FastAPI(
        title=f"{FASTAPI_TITLE} ML Service",
        version=FASTAPI_VERSION,
        description=(
            f"{FASTAPI_DESCRIPTION} Machine learning service for dataset preparation, "
            "training, checkpoint management, and model workflows."
        ),
        lifespan=app_lifespan,
    )
    resolved_container = container or MlServiceContainer()
    application.state.container = resolved_container
    application.include_router(health_router)
    register_ml_routes(application, resolved_container)
    return application


app = create_app()

__all__ = ["app", "create_app"]
