from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, Request

from adsmod_common.capabilities import (
    CapabilitiesResponse,
    FeatureCapabilities,
    ServiceCapability,
)
from adsmod_common.config import AdsmodConfig, load_config
from adsmod_common.paths import resolve_checkpoint_root, resolve_log_root, resolve_storage_root
from adsmod_common.version import __version__

from adsmod_ml.common.utils.logger import close_file_logging, configure_logging
from adsmod_ml.contracts.configuration import (
    RuntimeDeviceCapabilities,
    TrainingConfigurationResponse,
)
from adsmod_ml.contracts.training import (
    DatasetBuildRequest,
    ResumeTrainingRequest,
    TrainingConfigRequest,
)
from adsmod_ml.http.entrypoint import health_router
from adsmod_ml.http.routes import register_ml_routes
from adsmod_ml.services.container import MlServiceContainer


def capabilities(request: Request) -> CapabilitiesResponse:
    config: AdsmodConfig = request.app.state.config
    return CapabilitiesResponse(
        configured_mode=config.runtime.mode,
        version=__version__,
        features=FeatureCapabilities(
            datasets=False,
            nist=False,
            fitting=False,
            training=True,
            checkpoints=True,
        ),
        services={
            "ml": ServiceCapability(
                configured=True,
                health="ready",
                readiness="ready",
                version=__version__,
            )
        },
    )


def _numeric_constraints(model: type[Any]) -> dict[str, dict[str, int | float]]:
    constraints: dict[str, dict[str, int | float]] = {}
    for name, field in model.model_fields.items():
        values: dict[str, int | float] = {}
        for metadata in field.metadata:
            for attribute, key in (
                ("ge", "minimum"),
                ("gt", "exclusive_minimum"),
                ("le", "maximum"),
                ("lt", "exclusive_maximum"),
            ):
                value = getattr(metadata, attribute, None)
                if value is not None:
                    values[key] = value
        if values:
            constraints[name] = values
    return constraints


def _device_capabilities() -> RuntimeDeviceCapabilities:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    devices = tuple(
        str(torch.cuda.get_device_name(index)) for index in range(device_count)
    )
    return RuntimeDeviceCapabilities(
        keras_backend="torch",
        cuda_available=cuda_available,
        device_count=device_count,
        devices=devices,
    )


def configuration(request: Request) -> TrainingConfigurationResponse:
    config: AdsmodConfig = request.app.state.config
    training_defaults = TrainingConfigRequest().model_dump(mode="json")
    training_defaults.update(
        {
            "use_jit": config.application.training.use_jit,
            "jit_backend": config.application.training.jit_backend,
            "use_mixed_precision": config.application.training.use_mixed_precision,
            "dataloader_workers": config.application.training.dataloader_workers,
        }
    )
    dataset_defaults = DatasetBuildRequest.model_construct(
        datasets=[]
    ).model_dump(mode="json", exclude={"datasets"})
    return TrainingConfigurationResponse(
        defaults=training_defaults,
        dataset_defaults=dataset_defaults,
        resume_defaults=ResumeTrainingRequest.model_construct(
            checkpoint_name="configuration"
        ).model_dump(mode="json", exclude={"checkpoint_name"}),
        numeric_constraints={
            **_numeric_constraints(TrainingConfigRequest),
            **{
                f"dataset_{name}": value
                for name, value in _numeric_constraints(DatasetBuildRequest).items()
            },
            "additional_epochs": _numeric_constraints(ResumeTrainingRequest)[
                "additional_epochs"
            ],
        },
        supported_models=("SCADS Series", "SCADS Atomic"),
        checkpoint_capabilities={
            "save": True,
            "resume": True,
            "delete": True,
        },
        runtime=_device_capabilities(),
    )


def _build_router() -> APIRouter:
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
        response_model=TrainingConfigurationResponse,
        tags=["system"],
    )
    return router


@asynccontextmanager
async def app_lifespan(application: FastAPI) -> AsyncIterator[None]:
    config: AdsmodConfig = application.state.config
    resolve_storage_root(config).mkdir(parents=True, exist_ok=True)
    resolve_checkpoint_root(config).mkdir(parents=True, exist_ok=True)
    configure_logging(resolve_log_root(config))
    application.state.ready = True
    try:
        yield
    finally:
        application.state.ready = False
        close_file_logging()


def create_app(config: AdsmodConfig, *, internal_token: str | None = None) -> FastAPI:
    application = FastAPI(title="ADSMOD ML Service", version=__version__, lifespan=app_lifespan)
    application.state.config = config
    application.state.ready = False
    application.state.container = MlServiceContainer(config, internal_token=internal_token)
    application.include_router(health_router)
    application.include_router(_build_router(), prefix="/api/v1")
    register_ml_routes(application, application.state.container, prefix="/api/v1")
    return application


def create_app_from_path(
    config_path: str | Path,
    *,
    internal_token: str | None = None,
) -> FastAPI:
    return create_app(load_config(config_path), internal_token=internal_token)


__all__ = ["app_lifespan", "create_app", "create_app_from_path"]
