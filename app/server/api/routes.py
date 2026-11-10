# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI

from server.services.container import CoreServiceContainer
from server.api.analytics import create_analytics_router
from server.api.datasets import create_dataset_router
from server.api.fitting import create_fitting_router
from server.api.home import create_home_router
from server.api.jobs import create_jobs_router
from server.api.materials import create_material_router
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
        create_material_router,
        create_public_data_router,
        create_analytics_router,
        create_home_router,
        create_jobs_router,
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
    """Register ML routes; each request is gated on a verified deep learning install.

    The routers only import ML packages lazily, so registering them is safe on
    the lightweight profile.
    """

    from server.api.deep_learning import require_deep_learning
    from server.api.training import create_training_router
    from server.api.training_configuration import create_configuration_router

    guard = [Depends(require_deep_learning)]
    app.include_router(
        create_configuration_router(),
        prefix=prefix,
        include_in_schema=include_schema,
        dependencies=guard,
    )
    app.include_router(
        create_training_router(container),
        prefix=prefix,
        include_in_schema=include_schema,
        dependencies=guard,
    )


__all__ = ["register_core_routes", "register_ml_routes"]
