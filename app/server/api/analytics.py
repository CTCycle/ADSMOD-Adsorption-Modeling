# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from server.domain.analytics import (
    AnalyticsOverviewResponse,
    DistributionRequest,
    DistributionResponse,
    SelectionRequest,
    SelectionResponse,
    SeriesRequest,
    SeriesResponse,
    UptakeAtPressureRequest,
    UptakeAtPressureResponse,
)
from server.services.container import CoreServiceContainer
from server.services.data.analytics import AnalyticsService

###############################################################################
class AnalyticsEndpoint:
    """Read-only analytical queries; POST only carries structured filters."""

    # -------------------------------------------------------------------------
    def __init__(self, router: APIRouter, service: AnalyticsService) -> None:
        self.router = router
        self.service = service

    # -------------------------------------------------------------------------
    @staticmethod
    def bad_request(exc: ValueError) -> HTTPException:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    # -------------------------------------------------------------------------
    def get_overview(self) -> AnalyticsOverviewResponse:
        return self.service.overview()

    # -------------------------------------------------------------------------
    def query_selection(self, request: SelectionRequest) -> SelectionResponse:
        return self.service.selection(request)

    # -------------------------------------------------------------------------
    def query_series(self, request: SeriesRequest) -> SeriesResponse:
        try:
            return self.service.series(request)
        except ValueError as exc:
            raise self.bad_request(exc) from exc

    # -------------------------------------------------------------------------
    def query_uptake_at_pressure(
        self, request: UptakeAtPressureRequest
    ) -> UptakeAtPressureResponse:
        try:
            return self.service.uptake_at_pressure(request)
        except ValueError as exc:
            raise self.bad_request(exc) from exc

    # -------------------------------------------------------------------------
    def query_distribution(self, request: DistributionRequest) -> DistributionResponse:
        try:
            return self.service.distribution(request)
        except ValueError as exc:
            raise self.bad_request(exc) from exc

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "/overview",
            self.get_overview,
            methods=["GET"],
            response_model=AnalyticsOverviewResponse,
        )
        self.router.add_api_route(
            "/selection",
            self.query_selection,
            methods=["POST"],
            response_model=SelectionResponse,
        )
        self.router.add_api_route(
            "/series",
            self.query_series,
            methods=["POST"],
            response_model=SeriesResponse,
        )
        self.router.add_api_route(
            "/uptake-at-pressure",
            self.query_uptake_at_pressure,
            methods=["POST"],
            response_model=UptakeAtPressureResponse,
        )
        self.router.add_api_route(
            "/distribution",
            self.query_distribution,
            methods=["POST"],
            response_model=DistributionResponse,
        )

###############################################################################
def create_analytics_router(container: CoreServiceContainer) -> APIRouter:
    router = APIRouter(prefix="/analytics", tags=["analytics"])
    AnalyticsEndpoint(router=router, service=container.analytics_service).add_routes()
    return router


__all__ = ["create_analytics_router"]
