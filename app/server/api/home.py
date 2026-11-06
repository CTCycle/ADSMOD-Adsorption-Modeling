# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from fastapi import APIRouter

from server.domain.home import HomeSummaryResponse
from server.repositories.home import HomeRepository
from server.services.container import CoreServiceContainer


###############################################################################
class HomeEndpoint:
    """Read-only application overview."""

    # -------------------------------------------------------------------------
    def __init__(self, router: APIRouter, repository: HomeRepository) -> None:
        self.router = router
        self.repository = repository

    # -------------------------------------------------------------------------
    def get_summary(self) -> HomeSummaryResponse:
        return HomeSummaryResponse.model_validate(self.repository.summary())

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "/summary",
            self.get_summary,
            methods=["GET"],
            response_model=HomeSummaryResponse,
        )


###############################################################################
def create_home_router(container: CoreServiceContainer) -> APIRouter:
    router = APIRouter(prefix="/home", tags=["home"])
    HomeEndpoint(router=router, repository=container.home).add_routes()
    return router


__all__ = ["create_home_router"]
