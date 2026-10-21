# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, model_validator

from server.configurations.settings import (
    DatasetConfig,
    FittingConfig,
    JobConfig,
    NISTConfig,
    PublicDataConfig,
    TrainingConfig,
    UserSettings,
)
from server.configurations.store import ApplicationSettingsStore
from server.configurations.settings import AdsmodConfig

RESTART_REQUIRED_FIELDS = (
    "public_data.sync_resume_on_startup",
)


###############################################################################
class SettingsUpdateRequest(UserSettings):
    """Full settings replacement; partial updates are intentionally rejected."""

    # -------------------------------------------------------------------------
    @model_validator(mode="before")
    @classmethod
    def require_full_settings_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            raise ValueError("A complete settings object is required.")
        expected = set(UserSettings.model_fields)
        if set(value) != expected:
            raise ValueError("A complete settings object is required.")
        section_models = {
            "datasets": DatasetConfig,
            "nist": NISTConfig,
            "public_data": PublicDataConfig,
            "fitting": FittingConfig,
            "jobs": JobConfig,
            "training": TrainingConfig,
        }
        for section, model in section_models.items():
            raw_section = value.get(section)
            if not isinstance(raw_section, dict) or set(raw_section) != set(model.model_fields):
                raise ValueError(f"A complete {section} settings section is required.")
        return value


###############################################################################
class SettingsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str = "success"
    current: UserSettings
    defaults: UserSettings
    restart_required: bool
    restart_required_fields: tuple[str, ...]
    pending_restart_fields: tuple[str, ...]


###############################################################################
def _store(request: Request) -> ApplicationSettingsStore:
    return request.app.state.settings_store


###############################################################################
def _current_settings(request: Request) -> UserSettings:
    runtime = request.app.state.runtime
    return _store(request).initialize(runtime.config.user_settings())


###############################################################################
def _response(request: Request, current: UserSettings) -> dict[str, Any]:
    runtime = request.app.state.runtime
    startup_settings = runtime.startup_settings
    restart_fields = []
    if current.public_data.sync_resume_on_startup != startup_settings.public_data.sync_resume_on_startup:
        restart_fields.append("public_data.sync_resume_on_startup")
    return {
        "status": "success",
        "current": current.model_dump(mode="json"),
        "defaults": UserSettings().model_dump(mode="json"),
        "restart_required": bool(restart_fields),
        "restart_required_fields": RESTART_REQUIRED_FIELDS,
        "pending_restart_fields": tuple(restart_fields),
    }


###############################################################################
def _apply(request: Request, settings: UserSettings) -> None:
    runtime = request.app.state.runtime
    config = AdsmodConfig.assemble(
        database=runtime.config.application.database,
        storage_root=runtime.config.storage.root,
        settings=settings,
        runtime=runtime.config.runtime,
    )
    runtime.config = config
    request.app.state.config = config
    runtime.container.apply_settings(config)
    ml_container = runtime.ml_container
    if ml_container is not None:
        ml_container.apply_settings(config)


###############################################################################
async def _save_and_apply(request: Request, settings: UserSettings) -> UserSettings:
    store = _store(request)
    runtime = request.app.state.runtime
    previous = await asyncio.to_thread(store.initialize, runtime.config.user_settings())
    saved = await asyncio.to_thread(store.save, settings)
    try:
        _apply(request, saved)
    except Exception as apply_error:  # noqa: BLE001
        try:
            restored = await asyncio.to_thread(store.save, previous)
            _apply(request, restored)
        except Exception as rollback_error:  # noqa: BLE001
            raise HTTPException(
                status_code=503,
                detail="Settings could not be applied or restored. Restart ADSMOD and verify the saved settings.",
            ) from rollback_error
        raise HTTPException(
            status_code=503,
            detail="Settings could not be applied. The previous settings were restored.",
        ) from apply_error
    return saved


###############################################################################
def create_settings_router() -> APIRouter:
    router = APIRouter(prefix="/settings", tags=["settings"])

    @router.get("", response_model=SettingsResponse)
    def get_settings(request: Request) -> dict[str, Any]:
        try:
            return _response(request, _current_settings(request))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail="Settings are unavailable.") from exc

    @router.put("", response_model=SettingsResponse)
    async def update_settings(
        payload: SettingsUpdateRequest, request: Request
    ) -> dict[str, Any]:
        try:
            async with request.app.state.settings_update_lock:
                saved = await _save_and_apply(
                    request,
                    UserSettings.model_validate(payload.model_dump()),
                )
                return _response(request, saved)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=503,
                detail="Settings could not be saved. Check the database and retry.",
            ) from exc

    @router.post("/reset", response_model=SettingsResponse)
    async def reset_settings(request: Request) -> dict[str, Any]:
        defaults = UserSettings()
        try:
            async with request.app.state.settings_update_lock:
                saved = await _save_and_apply(request, defaults)
                return _response(request, saved)
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(
                status_code=503,
                detail="Settings could not be reset. Check the database and retry.",
            ) from exc

    return router


__all__ = ["RESTART_REQUIRED_FIELDS", "SettingsUpdateRequest", "create_settings_router"]
