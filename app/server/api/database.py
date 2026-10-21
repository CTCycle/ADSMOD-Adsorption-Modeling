# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from server.common.utils.logger import logger

from server.configurations.database_config import write_database_config
from server.configurations.settings import AdsmodConfig, DatabaseConfig, UserSettings
from server.configurations.store import ApplicationSettingsStore
from server.repositories.database.initializer import prepare_database_for_startup
from server.repositories.database.manager import DatabaseManager
from server.repositories.database.migrator import CURRENT_TABLES


###############################################################################
class DatabaseSettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["sqlite", "postgres"]
    sqlite_path: str = "data/database.db"
    engine: Literal["postgres", "postgresql", "postgresql+psycopg"] = "postgresql+psycopg"
    host: str | None = None
    port: int = Field(default=5432, ge=1, le=65535)
    database_name: str | None = None
    username: str | None = None
    password: str | None = None
    clear_password: bool = False
    ssl: bool = False
    ssl_ca: str | None = None
    connect_timeout: int = Field(default=30, ge=1, le=300)

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_postgres(self) -> "DatabaseSettingsRequest":
        if self.backend == "postgres" and not all(
            (self.host, self.database_name, self.username)
        ):
            raise ValueError("Host, database name, and username are required for PostgreSQL.")
        if not self.sqlite_path.strip() or self.sqlite_path.strip() == ":memory:":
            raise ValueError("Enter a persistent SQLite file path; in-memory databases are reserved for tests.")
        return self


###############################################################################
class DatabaseSettingsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    backend: Literal["sqlite", "postgres"]
    connected: bool
    sqlite_path: str | None = None
    engine: str | None = None
    host: str | None = None
    port: int | None = None
    database_name: str | None = None
    username: str | None = None
    ssl: bool = False
    ssl_ca: str | None = None
    connect_timeout: int
    password_configured: bool


###############################################################################
class DatabaseTestResponse(BaseModel):
    success: Literal[True] = True
    message: str


###############################################################################
def _response(request: Request) -> dict[str, Any]:
    runtime = request.app.state.runtime
    database = runtime.config.application.database
    return {
        "backend": "sqlite" if database.embedded_database else "postgres",
        "connected": runtime.database_ready,
        "sqlite_path": database.sqlite_path if database.embedded_database else None,
        "engine": None if database.embedded_database else database.engine,
        "host": None if database.embedded_database else database.host,
        "port": None if database.embedded_database else database.port,
        "database_name": None if database.embedded_database else database.database_name,
        "username": None if database.embedded_database else database.username,
        "ssl": database.ssl if not database.embedded_database else False,
        "ssl_ca": database.ssl_ca if not database.embedded_database else None,
        "connect_timeout": database.connect_timeout,
        "password_configured": bool(database.password),
    }


###############################################################################
def _build_config(payload: DatabaseSettingsRequest, current: DatabaseConfig) -> DatabaseConfig:
    same_backend = (payload.backend == "sqlite") == current.embedded_database
    password = None
    if payload.backend == "postgres":
        if payload.password:
            password = payload.password
        elif payload.clear_password:
            password = None
        elif same_backend:
            password = current.password
    return DatabaseConfig.model_validate(
        {
            "embedded_database": payload.backend == "sqlite",
            "engine": payload.engine,
            "host": payload.host,
            "port": payload.port,
            "database_name": payload.database_name,
            "username": payload.username,
            "password": password,
            "ssl": payload.ssl,
            "ssl_ca": payload.ssl_ca,
            "connect_timeout": payload.connect_timeout,
            "sqlite_path": payload.sqlite_path,
        }
    )


###############################################################################
def _test_connection(database: DatabaseConfig, storage_root) -> None:
    manager = DatabaseManager(database, storage_root=storage_root)
    try:
        with manager.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except (SQLAlchemyError, OSError, ValueError) as exc:
        raise RuntimeError(type(exc).__name__) from None
    finally:
        manager.dispose()


###############################################################################
def get_database_settings(request: Request) -> dict[str, Any]:
    return _response(request)


###############################################################################
async def test_database_connection(
    payload: DatabaseSettingsRequest, request: Request
) -> dict[str, Any]:
    current = request.app.state.runtime.config.application.database
    candidate = _build_config(payload, current)
    try:
        await asyncio.to_thread(
            _test_connection, candidate, request.app.state.runtime.config.storage.root
        )
    except Exception as exc:  # noqa: BLE001
        detail = (
            "SQLite could not open the configured file. Check the path and folder permissions."
            if candidate.embedded_database
            else "PostgreSQL connection failed. Check the host, database, credentials, and TLS settings."
        )
        raise HTTPException(status_code=400, detail=detail) from exc
    return {"success": True, "message": "Connection succeeded."}


###############################################################################
class _DatabaseSwitch:
    """Coordinate a guarded database change and restore the old runtime on failure."""

    # -------------------------------------------------------------------------
    def __init__(self, payload: DatabaseSettingsRequest, request: Request) -> None:
        self.request = request
        self.application = request.app
        self.runtime = self.application.state.runtime
        self.container = self.application.state.core_container
        self.manager = self.container.database
        self.current_config = self.runtime.config
        self.current_database = self.current_config.application.database
        self.candidate_database = _build_config(payload, self.current_database)
        self.storage_root = self.current_config.storage.root
        self.sync = self.container.public_data_sync
        self.candidate: DatabaseManager | None = None
        self.snapshot = None
        self.file_written = False
        self.request_gate_closed = False
        self.admission_closed = False
        self.config_restore_failed = False
        self.old_database_ready = self.runtime.database_ready
        self.old_startup_settings = self.runtime.startup_settings

    # -------------------------------------------------------------------------
    async def run(self) -> dict[str, Any]:
        if self.container.job_manager.is_job_running():
            raise HTTPException(
                status_code=409,
                detail="Wait for active jobs to finish before switching databases.",
            )
        if not await self.sync.pause_for_database_switch():
            raise HTTPException(
                status_code=409,
                detail="Wait for active public-data sync runs to finish before switching databases.",
            )
        try:
            return await self._activate()
        except HTTPException:
            if self.snapshot is not None:
                detail = await self._rollback()
                raise HTTPException(status_code=503, detail=detail)
            await self._release_without_switch()
            raise
        except Exception as exc:  # noqa: BLE001
            detail = await self._rollback()
            raise HTTPException(status_code=503, detail=detail) from exc

    # -------------------------------------------------------------------------
    async def _activate(self) -> dict[str, Any]:
        self.candidate = await asyncio.to_thread(
            _prepare_candidate, self.candidate_database, self.storage_root
        )
        target_settings = await asyncio.to_thread(
            ApplicationSettingsStore(self.candidate).initialize, UserSettings()
        )
        await self._quiesce_requests()
        self.snapshot = self.manager.adopt(self.candidate)
        self._apply_candidate_settings(target_settings)
        self.manager.open_admission()
        self.admission_closed = False
        self.container.public_data_service.ensure_sources()
        if not self.sync._started:
            await self.sync.start()
        write_database_config(self.candidate_database, self.storage_root)
        self.file_written = True
        await self._open_request_gate()
        self.sync.resume_after_database_switch()
        self._dispose_previous_engine()
        return _response(self.request)

    # -------------------------------------------------------------------------
    async def _quiesce_requests(self) -> None:
        drained = await self.application.state.database_request_gate.close(30.0)
        if not drained:
            raise HTTPException(
                status_code=409,
                detail="Requests are still using the database. Retry the switch shortly.",
            )
        self.request_gate_closed = True
        self.manager.close_admission()
        self.admission_closed = True
        idle = await asyncio.to_thread(self.manager.wait_until_idle, 30.0)
        if not idle:
            self.manager.open_admission()
            self.admission_closed = False
            raise HTTPException(
                status_code=409,
                detail="Database operations are still in progress. Retry the switch shortly.",
            )

    # -------------------------------------------------------------------------
    def _apply_candidate_settings(self, target_settings: UserSettings) -> None:
        next_config = AdsmodConfig.assemble(
            database=self.candidate_database,
            storage_root=self.storage_root,
            settings=target_settings,
            runtime=self.current_config.runtime,
        )
        self.runtime.config = next_config
        self.application.state.config = next_config
        self.container.apply_settings(next_config)
        if self.runtime.ml_container is not None:
            self.runtime.ml_container.apply_settings(next_config)
        self.runtime.database_ready = True

    # -------------------------------------------------------------------------
    async def _open_request_gate(self) -> None:
        await self.application.state.database_request_gate.open()
        self.request_gate_closed = False

    # -------------------------------------------------------------------------
    def _dispose_previous_engine(self) -> None:
        try:
            self.snapshot.engine.dispose()
        except Exception:
            logger.exception("Could not dispose the previous database engine after switching.")

    # -------------------------------------------------------------------------
    async def _release_without_switch(self) -> None:
        if self.admission_closed:
            self.manager.open_admission()
            self.admission_closed = False
        if self.request_gate_closed:
            await self._open_request_gate()
        self.sync.resume_after_database_switch()
        if self.candidate is not None:
            self.candidate.dispose()

    # -------------------------------------------------------------------------
    async def _rollback(self) -> str:
        if self.snapshot is not None:
            if not self.admission_closed:
                self.manager.close_admission()
                self.admission_closed = True
                await asyncio.to_thread(self.manager.wait_until_idle, 30.0)
            failed_engine = self.manager.restore(self.snapshot)
            failed_engine.dispose()
            self.admission_closed = True
        self._restore_database_file()
        self.runtime.config = self.current_config
        self.application.state.config = self.current_config
        self.runtime.database_ready = self.old_database_ready
        self.runtime.startup_settings = self.old_startup_settings
        services_restore_failed = self._restore_services()
        if self.admission_closed:
            self.manager.open_admission()
            self.admission_closed = False
        if self.request_gate_closed:
            await self._open_request_gate()
        self.sync.resume_after_database_switch()
        if self.candidate is not None:
            self.candidate.dispose()
        if self.config_restore_failed:
            return (
                "The previous database was restored for this run, but database.json could not be restored. "
                "Correct data-directory permissions before restarting ADSMOD."
            )
        if services_restore_failed:
            return (
                "The previous database connection was restored, but one or more services could not be "
                "reconfigured. Restart ADSMOD before continuing."
            )
        return "The new database could not be activated. The previous database remains selected."

    # -------------------------------------------------------------------------
    def _restore_database_file(self) -> None:
        self.config_restore_failed = False
        if not self.file_written:
            return
        try:
            write_database_config(self.current_database, self.storage_root)
        except Exception as exc:  # noqa: BLE001
            self.config_restore_failed = True
            logger.warning(
                "Could not restore database.json after a failed switch (%s).",
                type(exc).__name__,
            )

    # -------------------------------------------------------------------------
    def _restore_services(self) -> bool:
        failed = False
        try:
            self.container.apply_settings(self.current_config)
        except Exception as exc:  # noqa: BLE001
            failed = True
            logger.warning(
                "Could not restore core services after a failed database switch (%s).",
                type(exc).__name__,
            )
        if self.runtime.ml_container is not None:
            try:
                self.runtime.ml_container.apply_settings(self.current_config)
            except Exception as exc:  # noqa: BLE001
                failed = True
                logger.warning(
                    "Could not restore ML services after a failed database switch (%s).",
                    type(exc).__name__,
                )
        return failed


###############################################################################
async def switch_database(
    payload: DatabaseSettingsRequest, request: Request
) -> dict[str, Any]:
    transaction = _DatabaseSwitch(payload, request)
    async with request.app.state.settings_update_lock:
        async with request.app.state.database_switch_lock:
            return await transaction.run()


###############################################################################
def create_database_router() -> APIRouter:
    router = APIRouter(prefix="/database", tags=["database"])
    router.add_api_route(
        "", get_database_settings, methods=["GET"], response_model=DatabaseSettingsResponse
    )
    router.add_api_route(
        "/test",
        test_database_connection,
        methods=["POST"],
        response_model=DatabaseTestResponse,
    )
    router.add_api_route(
        "/switch",
        switch_database,
        methods=["POST"],
        response_model=DatabaseSettingsResponse,
    )
    return router


###############################################################################
def _prepare_candidate(database: DatabaseConfig, storage_root) -> DatabaseManager:
    prepare_database_for_startup(database, storage_root=storage_root)
    candidate = DatabaseManager(database, storage_root=storage_root)
    try:
        with candidate.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            tables = set(inspect(connection).get_table_names())
        if not CURRENT_TABLES.issubset(tables):
            raise RuntimeError("The target database schema is incomplete.")
        return candidate
    except Exception:
        candidate.dispose()
        raise


__all__ = ["DatabaseSettingsRequest", "create_database_router"]
