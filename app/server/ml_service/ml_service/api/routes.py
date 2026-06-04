from __future__ import annotations

from fastapi import FastAPI

from ml_service.api.training import create_training_router
from ml_service.services.container import MlServiceContainer


def register_ml_routes(
    app: FastAPI,
    container: MlServiceContainer,
    *,
    prefix: str = "/api",
    include_schema: bool = True,
) -> None:
    app.include_router(
        create_training_router(container),
        prefix=prefix,
        include_in_schema=include_schema,
    )


__all__ = ["register_ml_routes"]
