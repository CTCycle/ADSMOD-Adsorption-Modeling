from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from server.services.container import CoreServiceContainer
from server.api.datasets import create_dataset_router
from server.api.fitting import create_fitting_router
from server.api.nist import create_nist_router
from server.api.public_data import create_public_data_router

###############################################################################
def register_core_routes(
    app: FastAPI,
    container: CoreServiceContainer,
    *,
    prefix: str = "/api/v1",
    include_schema: bool = True,
) -> None:
    for router_factory in (
        create_dataset_router,
        create_fitting_router,
        create_nist_router,
        create_public_data_router,
    ):
        router = router_factory(container)
        app.include_router(router, prefix=prefix, include_in_schema=include_schema)


###############################################################################
def register_ml_routes(
    app: FastAPI,
    container: Any,
    *,
    prefix: str = "/api/v1",
    include_schema: bool = True,
) -> None:
    """Register ML routes only after the optional ML profile is imported."""

    from server.api.training import create_training_router
    from server.api.training_configuration import create_configuration_router

    app.include_router(
        create_configuration_router(),
        prefix=prefix,
        include_in_schema=include_schema,
    )
    app.include_router(
        create_training_router(container),
        prefix=prefix,
        include_in_schema=include_schema,
    )


__all__ = ["register_core_routes", "register_ml_routes"]
