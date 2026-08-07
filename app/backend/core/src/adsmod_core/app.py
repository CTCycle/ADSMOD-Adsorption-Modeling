from __future__ import annotations

import hmac
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, FastAPI, Header, HTTPException, Query, Request

from adsmod_common.capabilities import CapabilitiesResponse, FeatureCapabilities, ServiceCapability
from adsmod_common.config import AdsmodConfig, load_config
from adsmod_common.health import HealthResponse
from adsmod_common.version import __version__

from .api import SnapshotCreateRequest, SnapshotCreateResponse, SnapshotPageResponse
from .persistence.paths import resolve_database_path
from .persistence.snapshots import SnapshotStore

###############################################################################
def _capabilities(config: AdsmodConfig) -> CapabilitiesResponse:
    ml_configured = config.runtime.mode == "core-ml"
    return CapabilitiesResponse(
        configured_mode=config.runtime.mode,
        version=__version__,
        features=FeatureCapabilities(
            datasets=True,
            nist=True,
            fitting=True,
            training=ml_configured,
            checkpoints=ml_configured,
        ),
        services={
            "core": ServiceCapability(configured=True, health="ready", readiness="ready", version=__version__),
            "ml": ServiceCapability(
                configured=ml_configured,
                health="starting" if ml_configured else "unavailable",
                readiness="not-ready" if ml_configured else "not-configured",
                reason=None if ml_configured else "Machine learning service is not configured.",
            ),
        },
    )

###############################################################################
@dataclass
class CoreApplicationState:
    config: AdsmodConfig
    internal_token: str | None
    snapshot_store: SnapshotStore | None = None

    # -------------------------------------------------------------------------
    def get_snapshot_store(self) -> SnapshotStore:
        if self.snapshot_store is None:
            self.snapshot_store = SnapshotStore(resolve_database_path(self.config))
        return self.snapshot_store

    # -------------------------------------------------------------------------
    def require_internal_token(self, provided_token: str | None) -> None:
        if (
            not self.internal_token
            or not provided_token
            or not hmac.compare_digest(provided_token, self.internal_token)
        ):
            raise HTTPException(status_code=401, detail="internal authentication required")

###############################################################################
def liveness() -> HealthResponse:
    return HealthResponse(service="core", version=__version__, state="ready")

###############################################################################
def readiness() -> HealthResponse:
    return HealthResponse(service="core", version=__version__, state="ready")

###############################################################################
def capabilities(request: Request) -> CapabilitiesResponse:
    return _capabilities(request.app.state.runtime.config)

###############################################################################
def create_snapshot(
    request: SnapshotCreateRequest,
    http_request: Request,
    x_adsmod_internal_token: str | None = Header(default=None),
) -> SnapshotCreateResponse:
    runtime: CoreApplicationState = http_request.app.state.runtime
    runtime.require_internal_token(x_adsmod_internal_token)
    record = runtime.get_snapshot_store().create(request.rows)
    return SnapshotCreateResponse(
        snapshot_id=record.snapshot_id,
        content_hash=record.content_hash,
        created_at=record.created_at,
        row_count=record.row_count,
    )

###############################################################################
def get_snapshot_page(
    snapshot_id: str,
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=1000),
    x_adsmod_internal_token: str | None = Header(default=None),
) -> SnapshotPageResponse:
    runtime: CoreApplicationState = request.app.state.runtime
    runtime.require_internal_token(x_adsmod_internal_token)
    try:
        snapshot = runtime.get_snapshot_store().get_page(snapshot_id, page, page_size)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="snapshot not found") from exc
    return SnapshotPageResponse(
        snapshot_id=snapshot.snapshot_id,
        content_hash=snapshot.content_hash,
        page=snapshot.page,
        page_size=snapshot.page_size,
        total_rows=snapshot.total_rows,
        rows=list(snapshot.rows),
    )

###############################################################################
def _build_router() -> APIRouter:
    router = APIRouter()
    router.add_api_route(
        "/health/live", liveness, methods=["GET"], response_model=HealthResponse, tags=["health"]
    )
    router.add_api_route(
        "/health/ready", readiness, methods=["GET"], response_model=HealthResponse, tags=["health"]
    )
    router.add_api_route(
        "/api/v1/system/capabilities",
        capabilities,
        methods=["GET"],
        response_model=CapabilitiesResponse,
        tags=["system"],
    )
    router.add_api_route(
        "/api/v1/internal/snapshots",
        create_snapshot,
        methods=["POST"],
        response_model=SnapshotCreateResponse,
        tags=["internal"],
    )
    router.add_api_route(
        "/api/v1/internal/snapshots/{snapshot_id}",
        get_snapshot_page,
        methods=["GET"],
        response_model=SnapshotPageResponse,
        tags=["internal"],
    )
    return router

###############################################################################
def create_app(config: AdsmodConfig, *, internal_token: str | None = None) -> FastAPI:
    application = FastAPI(title="ADSMOD Core Service", version=__version__)
    application.state.config = config
    application.state.runtime = CoreApplicationState(
        config=config,
        internal_token=internal_token,
    )
    application.include_router(_build_router())
    return application

###############################################################################
def create_app_from_path(config_path: str | Path, *, internal_token: str | None = None) -> FastAPI:
    return create_app(load_config(config_path), internal_token=internal_token)


__all__ = ["create_app", "create_app_from_path"]
