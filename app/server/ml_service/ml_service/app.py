from __future__ import annotations

import os
import warnings

from fastapi import FastAPI

from ml_service.api.entrypoint import health_router
from ml_service.api.routes import register_ml_routes
from ml_service.common.constants import (
    FASTAPI_DESCRIPTION,
    FASTAPI_TITLE,
    FASTAPI_VERSION,
)
from ml_service.services.container import MlServiceContainer

# Ensure Keras loads with the intended backend even when env vars are absent.
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("MPLBACKEND", "Agg")

warnings.filterwarnings("ignore", category=FutureWarning)

def create_app(container: MlServiceContainer | None = None) -> FastAPI:
    application = FastAPI(
        title=f"{FASTAPI_TITLE} ML Service",
        version=FASTAPI_VERSION,
        description=(
            f"{FASTAPI_DESCRIPTION} Machine learning service for dataset preparation, "
            "training, checkpoint management, and model workflows."
        ),
    )
    resolved_container = container or MlServiceContainer()
    application.state.container = resolved_container
    application.include_router(health_router)
    register_ml_routes(application, resolved_container)
    return application


app = create_app()

__all__ = ["app", "create_app"]
