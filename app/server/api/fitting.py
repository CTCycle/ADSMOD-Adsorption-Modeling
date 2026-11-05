# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Response, status

from server.domain.fitting import (
    FittingDatasetReadiness,
    FittingSessionDetail,
    FittingSessionListResponse,
    FittingSessionRequest,
    FittingSessionStartResponse,
    ModelCatalogResponse,
    ModelDefinition,
    ModelSettingsUpdate,
)
from server.repositories.fitting import SessionQuery
from server.services.container import CoreServiceContainer
from server.services.fitting import FittingService
from server.common.constants import (
    FITTING_JOBS_ENDPOINT,
    FITTING_JOB_STATUS_ENDPOINT,
    FITTING_MODEL_ENDPOINT,
    FITTING_MODEL_RESET_ENDPOINT,
    FITTING_MODELS_ENDPOINT,
    FITTING_ROUTER_PREFIX,
    FITTING_SESSION_ENDPOINT,
    FITTING_SESSIONS_ENDPOINT,
)
from server.domain.jobs import (
    JobCancelResponse,
    JobListResponse,
    JobStatusResponse,
)

SessionStatus = Literal["running", "completed", "warning", "failed", "cancelled"]

###############################################################################
def _start_of_day(value: date | None, *, next_day: bool = False) -> datetime | None:
    if value is None:
        return None
    moment = datetime.combine(value, time.min, tzinfo=timezone.utc)
    return moment + timedelta(days=1) if next_day else moment

###############################################################################
class FittingEndpoint:

    # -------------------------------------------------------------------------
    def __init__(self, router: APIRouter, service: FittingService) -> None:
        self.router = router
        self.service = service

    # -------------------------------------------------------------------------
    @staticmethod
    def _bad_request(exc: Exception) -> HTTPException:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    # -------------------------------------------------------------------------
    @staticmethod
    def _not_found(exc: Exception) -> HTTPException:
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    # -------------------------------------------------------------------------
    def model_catalog(self) -> ModelCatalogResponse:
        return self.service.model_catalog()

    # -------------------------------------------------------------------------
    def update_model(
        self, model_key: str, payload: ModelSettingsUpdate
    ) -> ModelDefinition:
        try:
            return self.service.update_model(model_key, payload)
        except LookupError as exc:
            raise self._not_found(exc) from exc
        except ValueError as exc:
            raise self._bad_request(exc) from exc

    # -------------------------------------------------------------------------
    def reset_model(self, model_key: str) -> ModelDefinition:
        try:
            return self.service.reset_model(model_key)
        except LookupError as exc:
            raise self._not_found(exc) from exc

    # -------------------------------------------------------------------------
    def dataset_readiness(self, dataset_id: int) -> FittingDatasetReadiness:
        try:
            return self.service.dataset_readiness(dataset_id)
        except LookupError as exc:
            raise self._not_found(exc) from exc

    # -------------------------------------------------------------------------
    def start_session(
        self, payload: FittingSessionRequest
    ) -> FittingSessionStartResponse:
        try:
            return self.service.start_session(payload)
        except (ValueError, LookupError) as exc:
            raise self._bad_request(exc) from exc

    # -------------------------------------------------------------------------
    def list_sessions(
        self,
        query: str = "",
        session_status: list[SessionStatus] = Query(default=[], alias="status"),
        dataset_id: int | None = Query(default=None, ge=1),
        model: str | None = None,
        created_from: date | None = None,
        created_to: date | None = None,
        sort: Literal["created_at", "name", "status"] = "created_at",
        direction: Literal["asc", "desc"] = "desc",
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=100),
    ) -> FittingSessionListResponse:
        if created_from and created_to and created_from > created_to:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="created_from must not be after created_to.",
            )
        return self.service.list_sessions(
            SessionQuery(
                query=query,
                statuses=tuple(session_status),
                dataset_id=dataset_id,
                model=model,
                created_from=_start_of_day(created_from),
                created_to=_start_of_day(created_to, next_day=True),
                sort=sort,
                descending=direction == "desc",
                page=page,
                page_size=page_size,
            )
        )

    # -------------------------------------------------------------------------
    def get_session(self, session_id: int) -> FittingSessionDetail:
        try:
            return self.service.get_session(session_id)
        except LookupError as exc:
            raise self._not_found(exc) from exc

    # -------------------------------------------------------------------------
    def delete_session(self, session_id: int) -> Response:
        try:
            self.service.delete_session(session_id)
        except LookupError as exc:
            raise self._not_found(exc) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(exc)
            ) from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # -------------------------------------------------------------------------
    def get_job_status(self, job_id: str) -> JobStatusResponse:
        try:
            return self.service.get_job_status(job_id)
        except LookupError as exc:
            raise self._not_found(exc) from exc

    # -------------------------------------------------------------------------
    def list_jobs(self) -> JobListResponse:
        return self.service.list_jobs()

    # -------------------------------------------------------------------------
    def cancel_job(self, job_id: str) -> JobCancelResponse:
        try:
            return self.service.cancel_job(job_id)
        except ValueError as exc:
            raise self._bad_request(exc) from exc

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        routes = (
            (FITTING_MODELS_ENDPOINT, self.model_catalog, "GET", ModelCatalogResponse),
            (FITTING_MODEL_ENDPOINT, self.update_model, "PUT", ModelDefinition),
            (FITTING_MODEL_RESET_ENDPOINT, self.reset_model, "POST", ModelDefinition),
            (
                "/datasets/{dataset_id}/readiness",
                self.dataset_readiness,
                "GET",
                FittingDatasetReadiness,
            ),
            (FITTING_SESSIONS_ENDPOINT, self.list_sessions, "GET", FittingSessionListResponse),
            (
                FITTING_SESSIONS_ENDPOINT,
                self.start_session,
                "POST",
                FittingSessionStartResponse,
            ),
            (FITTING_SESSION_ENDPOINT, self.get_session, "GET", FittingSessionDetail),
            (FITTING_JOBS_ENDPOINT, self.list_jobs, "GET", JobListResponse),
            (FITTING_JOB_STATUS_ENDPOINT, self.get_job_status, "GET", JobStatusResponse),
            (FITTING_JOB_STATUS_ENDPOINT, self.cancel_job, "DELETE", JobCancelResponse),
        )
        for path, endpoint, method, response_model in routes:
            self.router.add_api_route(
                path, endpoint, methods=[method], response_model=response_model
            )
        self.router.add_api_route(
            FITTING_SESSION_ENDPOINT,
            self.delete_session,
            methods=["DELETE"],
            status_code=status.HTTP_204_NO_CONTENT,
            response_class=Response,
        )

###############################################################################
def create_fitting_router(container: CoreServiceContainer) -> APIRouter:
    router = APIRouter(prefix=FITTING_ROUTER_PREFIX, tags=["fitting"])
    FittingEndpoint(router=router, service=container.fitting_service).add_routes()
    return router
