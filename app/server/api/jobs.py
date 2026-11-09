# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Query, Request

from server.domain.jobs import UnifiedJobListResponse, UnifiedJobView
from server.domain.public_data import PublicDataSyncRunView
from server.services.container import CoreServiceContainer

ACTIVE_STATUSES = frozenset({"queued", "running"})
MANAGER_STATUS = {"pending": "queued"}


###############################################################################
def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


###############################################################################
def acquisition_job(run: dict[str, Any]) -> UnifiedJobView:
    view = PublicDataSyncRunView.model_validate(run)
    return UnifiedJobView(
        id=view.id,
        kind="acquisition",
        name=view.operation,
        source=view.source_key,
        status=view.status,
        progress=view.progress,
        created_at=_aware(view.created_at),
        finished_at=_aware(view.finished_at) if view.finished_at else None,
        saved_count=view.imported_count,
        skipped_count=view.skipped_count,
        failed_count=view.failed_count,
        error=view.error_detail,
        run=view,
    )


###############################################################################
def managed_job(snapshot: dict[str, Any]) -> UnifiedJobView:
    job_type = str(snapshot["job_type"])
    finished = snapshot.get("completed_at_utc")
    return UnifiedJobView(
        id=str(snapshot["job_id"]),
        kind="fitting" if job_type == "fitting" else "training",
        name=job_type,
        status=MANAGER_STATUS.get(snapshot["status"], snapshot["status"]),
        progress=min(100.0, max(0.0, float(snapshot["progress"]))),
        created_at=_aware(snapshot["created_at_utc"]),
        finished_at=_aware(finished) if finished else None,
        error=snapshot.get("error"),
    )


###############################################################################
class JobsEndpoint:
    """Read-only view over every background job in the application."""

    # -------------------------------------------------------------------------
    def __init__(self, router: APIRouter, container: CoreServiceContainer) -> None:
        self.router = router
        self.container = container

    # -------------------------------------------------------------------------
    async def list_jobs(
        self,
        request: Request,
        limit: int = Query(default=20, ge=1, le=100),
    ) -> UnifiedJobListResponse:
        sync = self.container.public_data_sync
        active_runs, _ = await sync.list(
            limit=100, offset=0, source_key=None, status_filter=None, status_scope="active"
        )
        recent_runs, _ = await sync.list(
            limit=limit, offset=0, source_key=None, status_filter=None, status_scope="history"
        )
        jobs = [acquisition_job(run) for run in (*active_runs, *recent_runs)]

        managers = [self.container.job_manager]
        ml_container = request.app.state.runtime.ml_container
        if ml_container is not None and ml_container.job_manager is not None:
            managers.append(ml_container.job_manager)
        jobs.extend(
            managed_job(snapshot)
            for manager in managers
            for snapshot in manager.list_jobs()
        )

        active = [job for job in jobs if job.status in ACTIVE_STATUSES]
        finished = sorted(
            (job for job in jobs if job.status not in ACTIVE_STATUSES),
            key=lambda job: job.created_at,
            reverse=True,
        )[:limit]
        ordered = sorted(
            (*active, *finished), key=lambda job: job.created_at, reverse=True
        )
        return UnifiedJobListResponse(jobs=ordered, active_count=len(active))

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "",
            self.list_jobs,
            methods=["GET"],
            response_model=UnifiedJobListResponse,
        )


###############################################################################
def create_jobs_router(container: CoreServiceContainer) -> APIRouter:
    router = APIRouter(prefix="/jobs", tags=["jobs"])
    JobsEndpoint(router=router, container=container).add_routes()
    return router


__all__ = ["create_jobs_router"]
