from __future__ import annotations

from fastapi import FastAPI

from core_service.api.datasets import create_dataset_router
from core_service.api.fitting import create_fitting_router
from core_service.api.nist import create_nist_router
from core_service.services.container import CoreServiceContainer


def register_core_routes(
    app: FastAPI,
    container: CoreServiceContainer,
    *,
    prefix: str = "/api",
    include_schema: bool = False,
) -> None:
    for router_factory in (
        create_dataset_router,
        create_fitting_router,
        create_nist_router,
    ):
        router = router_factory(container)
        app.include_router(router, prefix=prefix, include_in_schema=include_schema)


__all__ = ["register_core_routes"]
