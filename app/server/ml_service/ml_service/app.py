from __future__ import annotations

import os
import warnings

from fastapi import FastAPI

from ml_service.common.constants import (
    FASTAPI_DESCRIPTION,
    FASTAPI_TITLE,
    FASTAPI_VERSION,
)
from ml_service.api.entrypoint import health_router

# Ensure Keras loads with the intended backend even when env vars are absent.
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("MPLBACKEND", "Agg")

from ml_service.api.training import router as training_router

warnings.filterwarnings("ignore", category=FutureWarning)

app = FastAPI(
    title=f"{FASTAPI_TITLE} ML Service",
    version=FASTAPI_VERSION,
    description=f"{FASTAPI_DESCRIPTION} Machine learning service for dataset preparation, training, checkpoint management, and model workflows.",
)

routers = [
    health_router,
    training_router,
]

for router in routers:
    if router is health_router:
        app.include_router(router)
    else:
        app.include_router(router, prefix="/api", include_in_schema=True)

__all__ = ["app"]
