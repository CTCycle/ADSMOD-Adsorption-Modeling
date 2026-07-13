from __future__ import annotations

import hmac
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query

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
def create_app(config: AdsmodConfig, *, internal_token: str | None = None) -> FastAPI:
    application = FastAPI(title="ADSMOD Core Service", version=__version__)
    application.state.config = config
    snapshot_store: SnapshotStore | None = None

    def get_snapshot_store() -> SnapshotStore:
        nonlocal snapshot_store
        if snapshot_store is None:
            snapshot_store = SnapshotStore(resolve_database_path(config))
        return snapshot_store

    def require_internal_token(provided_token: str | None) -> None:
        if not internal_token or not provided_token or not hmac.compare_digest(provided_token, internal_token):
            raise HTTPException(status_code=401, detail="internal authentication required")

    @application.get("/health/live", response_model=HealthResponse, tags=["health"])
    def liveness() -> HealthResponse:
        return HealthResponse(service="core", version=__version__, state="ready")

    @application.get("/health/ready", response_model=HealthResponse, tags=["health"])
    def readiness() -> HealthResponse:
        return HealthResponse(service="core", version=__version__, state="ready")

    @application.get("/api/v1/system/capabilities", response_model=CapabilitiesResponse, tags=["system"])
    def capabilities() -> CapabilitiesResponse:
        return _capabilities(config)

    @application.post("/api/v1/internal/snapshots", response_model=SnapshotCreateResponse, tags=["internal"])
    def create_snapshot(
        request: SnapshotCreateRequest,
        x_adsmod_internal_token: str | None = Header(default=None),
    ) -> SnapshotCreateResponse:
        require_internal_token(x_adsmod_internal_token)
        record = get_snapshot_store().create(request.rows)
        return SnapshotCreateResponse(
            snapshot_id=record.snapshot_id,
            content_hash=record.content_hash,
            created_at=record.created_at,
            row_count=record.row_count,
        )

    @application.get("/api/v1/internal/snapshots/{snapshot_id}", response_model=SnapshotPageResponse, tags=["internal"])
    def get_snapshot_page(
        snapshot_id: str,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=100, ge=1, le=1000),
        x_adsmod_internal_token: str | None = Header(default=None),
    ) -> SnapshotPageResponse:
        require_internal_token(x_adsmod_internal_token)
        try:
            snapshot = get_snapshot_store().get_page(snapshot_id, page, page_size)
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

    return application


###############################################################################
def create_app_from_path(config_path: str | Path, *, internal_token: str | None = None) -> FastAPI:
    return create_app(load_config(config_path), internal_token=internal_token)


__all__ = ["create_app", "create_app_from_path"]