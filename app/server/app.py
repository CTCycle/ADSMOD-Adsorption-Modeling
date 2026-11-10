# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import get_args

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from server.common.units import UnitRegistry
from server.common.utils.logger import close_file_logging, configure_logging, logger
from server.common.version import __version__
from server.configurations.settings import AdsmodConfig, UserSettings
from server.configurations.store import ApplicationSettingsStore
from server.domain.capabilities import CapabilitiesResponse, FeatureCapabilities
from server.domain.configuration import (
    DisplayUnitCapabilities,
    FittingConfigurationResponse,
    NumericBounds,
)
from server.domain.fitting import FittingSessionRequest
from server.domain.system import SystemStatusResponse
from server.api.entrypoint import health_router, register_root_routes
from server.api.frontend import register_frontend_routes
from server.api.database import create_database_router
from server.api.routes import register_core_routes, register_ml_routes
from server.common.constants import FASTAPI_DESCRIPTION, FASTAPI_TITLE
from server.common.path import resolve_storage_root
from server.repositories.database.initializer import prepare_database_for_startup
from server.services.container import CoreServiceContainer
from server.services.system_status import SystemStatusSampler
from server.services.training_data import TrainingDataService
from server.api.settings import create_settings_router
from server.api.deep_learning import create_deep_learning_router
from server.services.deep_learning import DeepLearningManager
from server.services.ml_bootstrap import configure_environment
from server.services.ml_lazy import LazyMlContainer


###############################################################################
class DatabaseRequestGate:
    """Drain ordinary HTTP work before a live database connection swap."""

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._accepting = True
        self._active = 0

    # -------------------------------------------------------------------------
    async def enter(self) -> bool:
        async with self._condition:
            if not self._accepting:
                return False
            self._active += 1
            return True

    # -------------------------------------------------------------------------
    async def leave(self) -> None:
        async with self._condition:
            self._active -= 1
            self._condition.notify_all()

    # -------------------------------------------------------------------------
    async def close(self, timeout: float) -> bool:
        async with self._condition:
            self._accepting = False
            try:
                await asyncio.wait_for(
                    self._condition.wait_for(lambda: self._active == 0), timeout=timeout
                )
                return True
            except TimeoutError:
                self._accepting = True
                self._condition.notify_all()
                return False

    # -------------------------------------------------------------------------
    async def open(self) -> None:
        async with self._condition:
            self._accepting = True
            self._condition.notify_all()


DATABASE_INDEPENDENT_PREFIXES = (
    "/api/v1/database",
    "/api/v1/deep-learning",
    "/api/v1/settings",
    "/api/v1/system",
)


###############################################################################
@dataclass
class ApplicationRuntime:
    config: AdsmodConfig
    container: CoreServiceContainer
    training_data: TrainingDataService
    settings_store: ApplicationSettingsStore
    startup_settings: UserSettings
    deep_learning: DeepLearningManager
    database_ready: bool = False
    ml_container: LazyMlContainer | None = None


###############################################################################
def capabilities(request: Request) -> CapabilitiesResponse:
    runtime: ApplicationRuntime = request.app.state.runtime
    available, _ = runtime.deep_learning.availability()
    return CapabilitiesResponse(
        version=__version__,
        features=FeatureCapabilities(
            datasets=True,
            nist=True,
            fitting=True,
            machine_learning=available,
            training=available,
            checkpoints=available,
        ),
    )


###############################################################################
def configuration(request: Request) -> FittingConfigurationResponse:
    config: AdsmodConfig = request.app.state.runtime.config
    fitting = config.application.fitting
    pressure_units = tuple(UnitRegistry.PRESSURE_TO_PA)
    uptake_units = tuple(dict.fromkeys(UnitRegistry.UPTAKE_ALIASES.values()))
    return FittingConfigurationResponse(
        supported_optimizers=tuple(
            get_args(FittingSessionRequest.model_fields["optimizer"].annotation)
        ),
        default_optimizer="trf",
        default_max_evaluations=fitting.default_max_iterations,
        max_evaluations_bounds=NumericBounds(
            minimum=10, maximum=fitting.max_iterations_upper_bound
        ),
        weighting_options=tuple(
            get_args(FittingSessionRequest.model_fields["weighting"].annotation)
        ),
        default_weighting="unweighted",
        display_units=DisplayUnitCapabilities(
            pressure=pressure_units + ("1", "%"),
            uptake=uptake_units,
            default_pressure="bar",
            default_uptake="mmol/g",
        ),
    )


###############################################################################
async def system_status(request: Request) -> SystemStatusResponse:
    runtime: ApplicationRuntime = request.app.state.runtime
    config = runtime.config
    return await asyncio.to_thread(
        request.app.state.system_sampler.sample,
        config.application.database,
        runtime.container.database,
        resolve_storage_root(config),
    )


###############################################################################
def _build_system_router() -> APIRouter:
    router = APIRouter()
    router.add_api_route(
        "/system/status",
        system_status,
        methods=["GET"],
        response_model=SystemStatusResponse,
        tags=["system"],
    )
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
    return router


###############################################################################
def _register_ml(application: FastAPI, runtime: ApplicationRuntime) -> None:
    # Routes are always mounted but gated per request on the verified deep
    # learning installation; the ML packages load on the first training call.
    configure_environment()
    ml_container = LazyMlContainer(runtime.config, snapshot_access=runtime.training_data)
    register_ml_routes(application, ml_container, prefix="/api/v1")
    runtime.ml_container = ml_container
    application.state.ml_container = ml_container


###############################################################################
@asynccontextmanager
async def app_lifespan(application: FastAPI) -> AsyncIterator[None]:
    config: AdsmodConfig = application.state.config
    storage_root = resolve_storage_root(config)
    storage_root.mkdir(parents=True, exist_ok=True)
    configure_logging(storage_root / "logs")
    try:
        prepare_database_for_startup(
            config.application.database,
            storage_root=storage_root,
        )
        settings = application.state.settings_store.initialize(config.user_settings())
        config = AdsmodConfig.assemble(
            database=config.application.database,
            storage_root=storage_root,
            settings=settings,
            runtime=config.runtime,
        )
        application.state.config = config
        application.state.runtime.config = config
        application.state.runtime.startup_settings = settings
        application.state.core_container.apply_settings(config)
        application.state.core_container.public_data_service.ensure_sources()
        application.state.core_container.fitting_service.fail_interrupted_sessions()
        await application.state.core_container.public_data_sync.start()
        application.state.runtime.database_ready = True
    except Exception as exc:  # noqa: BLE001
        application.state.runtime.database_ready = False
        logger.warning(
            "Configured database is unavailable during startup (%s); database-dependent services are paused.",
            type(exc).__name__,
        )
    application.state.deep_learning.start()
    application.state.ready = True
    try:
        yield
    finally:
        application.state.ready = False
        ml_container = getattr(application.state, "ml_container", None)
        if ml_container is not None:
            ml_container.shutdown()
        await application.state.core_container.shutdown_async()
        application.state.core_container.database.dispose()
        close_file_logging()


###############################################################################
async def database_switch_admission(
    request: Request, call_next: RequestResponseEndpoint
) -> Response:
    path = request.url.path
    if path.startswith("/api/v1/database"):
        return await call_next(request)
    if (
        path.startswith("/api/v1/")
        and not path.startswith(DATABASE_INDEPENDENT_PREFIXES)
        and not request.app.state.runtime.database_ready
    ):
        return JSONResponse(
            status_code=503,
            content={
                "detail": "The configured database is unavailable; "
                "open Database settings to repair or switch it."
            },
        )
    gate: DatabaseRequestGate = request.app.state.database_request_gate
    if not await gate.enter():
        return JSONResponse(
            status_code=503,
            content={"detail": "Database is switching; retry the request shortly."},
        )
    try:
        return await call_next(request)
    finally:
        await gate.leave()


###############################################################################
def create_app(
    config: AdsmodConfig,
    *,
    settings_store: ApplicationSettingsStore | None = None,
) -> FastAPI:
    database_manager = settings_store.database if settings_store is not None else None
    container = CoreServiceContainer(config, database_manager=database_manager)
    store = settings_store or ApplicationSettingsStore(container.database)
    def is_busy() -> bool:
        if container.job_manager.is_job_running():
            return True
        ml_jobs = runtime.ml_container.job_manager if runtime.ml_container is not None else None
        return ml_jobs is not None and ml_jobs.is_job_running()

    runtime = ApplicationRuntime(
        config=config,
        container=container,
        training_data=TrainingDataService(container),
        settings_store=store,
        startup_settings=config.user_settings(),
        deep_learning=DeepLearningManager(config, is_busy=is_busy),
    )
    application = FastAPI(
        title=FASTAPI_TITLE,
        description=FASTAPI_DESCRIPTION,
        version=__version__,
        lifespan=app_lifespan,
    )
    application.state.config = config
    application.state.ready = False
    application.state.core_container = container
    application.state.runtime = runtime
    application.state.deep_learning = runtime.deep_learning
    application.state.request_shutdown = None
    application.state.system_sampler = SystemStatusSampler()
    application.state.settings_store = store
    application.state.settings_update_lock = asyncio.Lock()
    application.state.database_switch_lock = asyncio.Lock()
    application.state.database_request_gate = DatabaseRequestGate()

    application.add_middleware(BaseHTTPMiddleware, dispatch=database_switch_admission)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            *config.runtime.cors_origins,
        ],
        allow_credentials=config.runtime.cors_allow_credentials,
        allow_methods=list(config.runtime.cors_allow_methods),
        allow_headers=list(config.runtime.cors_allow_headers),
    )
    application.include_router(health_router)
    application.include_router(_build_system_router(), prefix="/api/v1")
    application.include_router(create_settings_router(), prefix="/api/v1")
    application.include_router(create_database_router(), prefix="/api/v1")
    application.include_router(create_deep_learning_router(), prefix="/api/v1")
    register_core_routes(application, container, prefix="/api/v1")
    _register_ml(application, runtime)
    if not register_frontend_routes(application):
        register_root_routes(application)
    return application


__all__ = ["app_lifespan", "create_app"]
