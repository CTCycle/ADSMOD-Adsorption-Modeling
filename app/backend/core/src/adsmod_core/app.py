from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import hmac
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from fastapi import APIRouter, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from adsmod_common.capabilities import (
    CapabilitiesResponse,
    FeatureCapabilities,
    ServiceCapability,
)
from adsmod_common.config import AdsmodConfig, load_config
from adsmod_common.units import UnitRegistry
from adsmod_common.version import __version__

from .api import (
    SnapshotCreateRequest,
    SnapshotCreateResponse,
    SnapshotFromSelectionsRequest,
    SnapshotPageResponse,
)
from .common.constants import FASTAPI_DESCRIPTION, FASTAPI_TITLE
from .common.utils.logger import close_file_logging, configure_logging
from .contracts.configuration import (
    DisplayUnitCapabilities,
    FittingConfigurationResponse,
    NumericBounds,
    ParameterDefaults,
)
from .contracts.fitting import FittingRequest
from .http.entrypoint import health_router, register_root_routes
from .http.routes import register_core_routes
from .persistence.paths import resolve_storage_root
from .persistence.snapshots import SnapshotStore
from .repositories.database.initializer import prepare_database_for_startup
from .services.container import CoreServiceContainer


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
            "core": ServiceCapability(
                configured=True,
                health="ready",
                readiness="ready",
                version=__version__,
            ),
            "ml": ServiceCapability(
                configured=ml_configured,
                health="starting" if ml_configured else "unavailable",
                readiness="not-ready" if ml_configured else "not-configured",
                reason=None
                if ml_configured
                else "Machine learning service is not configured.",
            ),
        },
    )


@dataclass
class CoreApplicationState:
    config: AdsmodConfig
    container: CoreServiceContainer
    internal_token: str | None
    snapshot_store: SnapshotStore

    def require_internal_token(self, provided_token: str | None) -> None:
        if not self.config.security.internal_token_required:
            return
        if (
            not self.internal_token
            or not provided_token
            or not hmac.compare_digest(provided_token, self.internal_token)
        ):
            raise HTTPException(status_code=401, detail="internal authentication required")


def capabilities(request: Request) -> CapabilitiesResponse:
    return _capabilities(request.app.state.runtime.config)


def configuration(request: Request) -> FittingConfigurationResponse:
    config: AdsmodConfig = request.app.state.runtime.config
    fitting = config.application.fitting
    pressure_units = tuple(UnitRegistry.PRESSURE_TO_PA)
    uptake_units = tuple(
        dict.fromkeys(UnitRegistry.UPTAKE_ALIASES.values())
    )
    return FittingConfigurationResponse(
        supported_optimizers=tuple(
            get_args(FittingRequest.model_fields["optimizer"].annotation)
        ),
        default_optimizer="trf",
        default_max_evaluations=fitting.default_max_iterations,
        max_evaluations_bounds=NumericBounds(
            minimum=10,
            maximum=fitting.max_iterations_upper_bound,
        ),
        weighting_options=tuple(
            get_args(FittingRequest.model_fields["weighting"].annotation)
        ),
        default_weighting="unweighted",
        display_units=DisplayUnitCapabilities(
            pressure=pressure_units + ("1", "%"),
            uptake=uptake_units,
            default_pressure="bar",
            default_uptake="mmol/g",
        ),
        parameter_defaults=ParameterDefaults(
            lower=fitting.default_parameter_min,
            upper=fitting.default_parameter_max,
            initial=fitting.default_parameter_initial,
        ),
    )


def create_snapshot(
    request: SnapshotCreateRequest,
    http_request: Request,
    x_adsmod_internal_token: str | None = Header(default=None),
) -> SnapshotCreateResponse:
    runtime: CoreApplicationState = http_request.app.state.runtime
    runtime.require_internal_token(x_adsmod_internal_token)
    try:
        record = runtime.snapshot_store.create(request.rows, metadata=request.metadata)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SnapshotCreateResponse(
        snapshot_id=record.snapshot_id,
        content_hash=record.content_hash,
        created_at=record.created_at,
        row_count=record.row_count,
    )


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
        snapshot = runtime.snapshot_store.get_page(snapshot_id, page, page_size)
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


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _uploaded_snapshot_rows(
    runtime: CoreApplicationState,
    dataset_name: str,
    dataset_id: int | None,
) -> list[dict[str, object]]:
    summaries = runtime.container.datasets.list_summaries()
    dataset = next(
        (
            item
            for item in summaries
            if item["source"] == "uploaded"
            and (dataset_id is None and item["name"] == dataset_name or item["id"] == dataset_id)
        ),
        None,
    )
    if dataset is None:
        raise LookupError(f"Uploaded dataset '{dataset_name}' does not exist.")
    frame = runtime.container.datasets.observation_frame(int(dataset["id"]))
    if frame.empty:
        raise ValueError(f"Uploaded dataset '{dataset_name}' contains no observations.")
    rows: list[dict[str, object]] = []
    for index, record in enumerate(frame.to_dict(orient="records")):
        record = {str(key): _json_safe(value) for key, value in record.items()}
        experiment = str(record.get("experiment") or f"row-{index}")
        record.update(
            {
                "filename": f"{dataset_name}:{experiment}",
                "temperature": record.get("temperature"),
                "adsorbent_name": record.get("adsorbent_name"),
                "adsorbate_name": record.get("adsorbate_name"),
                "pressure_units": "Pa",
                "adsorption_units": "mol/kg",
            }
        )
        rows.append(record)
    return rows


def _nist_snapshot_rows(
    runtime: CoreApplicationState,
    dataset_name: str,
) -> list[dict[str, object]]:
    adsorption, guests, _ = runtime.container.nist_repository.load_adsorption_datasets()
    if adsorption.empty:
        raise ValueError(f"NIST dataset '{dataset_name}' contains no observations.")
    guest_properties = {
        str(row.get("name", "")).strip().casefold(): row
        for row in guests.to_dict(orient="records")
    }
    rows: list[dict[str, object]] = []
    for index, raw_record in enumerate(adsorption.to_dict(orient="records")):
        record = {str(key): _json_safe(value) for key, value in raw_record.items()}
        adsorbate = str(record.get("adsorbate") or "").strip()
        adsorbent = str(record.get("adsorbent") or "").strip()
        guest = guest_properties.get(adsorbate.casefold(), {})
        record.update(
            {
                "filename": f"{dataset_name}:{record.get('external_key') or index}",
                "temperature": record.get("temperature_k"),
                "adsorbent_name": adsorbent,
                "adsorbate_name": adsorbate,
                "adsorbate_molecular_weight": _json_safe(guest.get("molecular_weight")),
                "adsorbate_SMILE": _json_safe(guest.get("smile_code")),
                "pressure_units": "Pa",
                "adsorption_units": "mol/kg",
            }
        )
        rows.append(record)
    return rows


def training_sources(
    request: Request,
    x_adsmod_internal_token: str | None = Header(default=None),
) -> dict[str, list[dict[str, object]]]:
    runtime: CoreApplicationState = request.app.state.runtime
    runtime.require_internal_token(x_adsmod_internal_token)
    sources: list[dict[str, object]] = []
    if runtime.container.nist_repository.count_nist_rows().get("single_component_rows", 0) > 0:
        sources.append(
            {
                "source": "nist",
                "dataset_name": "NIST ISODB",
                "display_name": "NIST Single Component",
                "row_count": runtime.container.nist_repository.count_nist_rows().get(
                    "single_component_rows", 0
                ),
                "dataset_id": None,
            }
        )
    sources.extend(
        {
            "source": item["source"],
            "dataset_name": item["name"],
            "display_name": item["name"],
            "row_count": item["observation_count"],
            "dataset_id": item["id"],
        }
        for item in runtime.container.datasets.list_summaries()
        if item["source"] == "uploaded"
    )
    return {"datasets": sources}


def create_snapshot_from_selections(
    payload: SnapshotFromSelectionsRequest,
    request: Request,
    x_adsmod_internal_token: str | None = Header(default=None),
) -> SnapshotCreateResponse:
    runtime: CoreApplicationState = request.app.state.runtime
    runtime.require_internal_token(x_adsmod_internal_token)
    rows: list[dict[str, object]] = []
    for selection in payload.selections:
        if selection.source == "uploaded":
            rows.extend(
                _uploaded_snapshot_rows(
                    runtime,
                    selection.dataset_name,
                    selection.dataset_id,
                )
            )
        else:
            rows.extend(_nist_snapshot_rows(runtime, selection.dataset_name))
    try:
        record = runtime.snapshot_store.create(
            rows,
            metadata={
                **payload.metadata,
                "selections": [item.model_dump(mode="json") for item in payload.selections],
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SnapshotCreateResponse(
        snapshot_id=record.snapshot_id,
        content_hash=record.content_hash,
        created_at=record.created_at,
        row_count=record.row_count,
    )


def _build_internal_router() -> APIRouter:
    router = APIRouter()
    router.add_api_route(
        "/system/capabilities",
        capabilities,
        methods=["GET"],
        response_model=CapabilitiesResponse,
        tags=["system"],
    )
    router.add_api_route(
        "/system/configuration",
        configuration,
        methods=["GET"],
        response_model=FittingConfigurationResponse,
        tags=["system"],
    )
    router.add_api_route(
        "/internal/snapshots",
        create_snapshot,
        methods=["POST"],
        response_model=SnapshotCreateResponse,
        tags=["internal"],
    )
    router.add_api_route(
        "/internal/snapshots/{snapshot_id}",
        get_snapshot_page,
        methods=["GET"],
        response_model=SnapshotPageResponse,
        tags=["internal"],
    )
    router.add_api_route(
        "/internal/training/sources",
        training_sources,
        methods=["GET"],
        tags=["internal"],
    )
    router.add_api_route(
        "/internal/training/snapshots",
        create_snapshot_from_selections,
        methods=["POST"],
        response_model=SnapshotCreateResponse,
        tags=["internal"],
    )
    return router


@asynccontextmanager
async def app_lifespan(application: FastAPI) -> AsyncIterator[None]:
    config: AdsmodConfig = application.state.config
    storage_root = resolve_storage_root(config)
    storage_root.mkdir(parents=True, exist_ok=True)
    configure_logging(storage_root / "logs")
    prepare_database_for_startup(
        config.application.database,
        storage_root=storage_root,
    )
    application.state.ready = True
    try:
        yield
    finally:
        application.state.ready = False
        application.state.core_container.database.dispose()
        close_file_logging()


def _internal_token(config: AdsmodConfig, explicit: str | None) -> str | None:
    if explicit is not None:
        return explicit
    if not config.security.internal_token_required:
        return None
    return os.environ.get(config.security.internal_token_env)


def create_app(config: AdsmodConfig, *, internal_token: str | None = None) -> FastAPI:
    container = CoreServiceContainer(config)
    application = FastAPI(
        title=FASTAPI_TITLE,
        description=FASTAPI_DESCRIPTION,
        version=__version__,
        lifespan=app_lifespan,
    )
    application.state.config = config
    application.state.ready = False
    application.state.core_container = container
    application.state.runtime = CoreApplicationState(
        config=config,
        container=container,
        internal_token=_internal_token(config, internal_token),
        snapshot_store=SnapshotStore(container.database),
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            f"http://{config.runtime.host}:{config.runtime.frontend_port}",
            f"http://127.0.0.1:{config.runtime.frontend_port}",
            f"http://localhost:{config.runtime.frontend_port}",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(health_router)
    application.include_router(_build_internal_router(), prefix="/api/v1")
    register_core_routes(application, container, prefix="/api/v1")
    register_root_routes(application)
    return application


def create_app_from_path(
    config_path: str | Path,
    *,
    internal_token: str | None = None,
) -> FastAPI:
    return create_app(load_config(config_path), internal_token=internal_token)


__all__ = ["app_lifespan", "create_app", "create_app_from_path"]
