# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from server.configurations.settings import PublicDataConfig
from server.repositories.public_data import PublicDataRepository
from server.services.data.nist_service import NISTDataService
from server.services.data.public_data import PublicDataService
from server.services.providers.public_data import (
    ProviderHealth,
    ProviderError,
    ProviderUnavailableError,
)

NIST_CATEGORIES = ("experiments", "guest", "host")


ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]
CancelCheck = Callable[[], Awaitable[bool]]

###############################################################################
class PublicDataSyncCoordinator:
    """Run bounded public-data acquisitions backed by durable database state."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        repository: PublicDataRepository,
        public_data: PublicDataService,
        nist: NISTDataService,
        config: PublicDataConfig,
    ) -> None:
        self.repository = repository
        self.public_data = public_data
        self.nist = nist
        self.config = config
        self._wake = asyncio.Event()
        self._closing = False
        self._workers: list[asyncio.Task[None]] = []
        self._executing: set[asyncio.Task[None]] = set()
        self._scheduler: asyncio.Task[None] | None = None
        self._started = False
        self._switch_paused = False
        self._switch_lock = asyncio.Lock()
        self._target_worker_count = config.sync_max_concurrent_jobs
        self._next_worker_index = 0
        self._run_settings: ContextVar[
            tuple[PublicDataConfig, PublicDataService, NISTDataService] | None
        ] = ContextVar("public_data_sync_settings", default=None)

    # -------------------------------------------------------------------------
    def _active_settings(
        self,
    ) -> tuple[PublicDataConfig, PublicDataService, NISTDataService]:
        return self._run_settings.get() or (self.config, self.public_data, self.nist)

    # -------------------------------------------------------------------------
    @property
    def has_active_work(self) -> bool:
        return any(not task.done() for task in self._executing)

    # -------------------------------------------------------------------------
    async def pause_for_database_switch(self) -> bool:
        async with self._switch_lock:
            if self.has_active_work:
                return False
            self._switch_paused = True
            return True

    # -------------------------------------------------------------------------
    def resume_after_database_switch(self) -> None:
        self._switch_paused = False
        self._wake.set()

    # -------------------------------------------------------------------------
    def apply_settings(
        self,
        *,
        config: PublicDataConfig,
        public_data: PublicDataService,
        nist: NISTDataService,
    ) -> None:
        self.config = config
        self.public_data = public_data
        self.nist = nist
        self._target_worker_count = config.sync_max_concurrent_jobs
        self._wake.set()
        if not self._started:
            return
        while self._next_worker_index < self._target_worker_count:
            index = self._next_worker_index
            self._next_worker_index += 1
            self._workers.append(
                asyncio.create_task(
                    self._worker_loop(index), name=f"public-data-sync-{index}"
                )
            )
        if self._scheduler is not None:
            self._scheduler.cancel()
            self._scheduler = None
        if config.auto_sync_enabled and config.auto_sync_categories:
            self._scheduler = asyncio.create_task(
                self._schedule_nist_refreshes(), name="public-data-auto-sync"
            )

    # -------------------------------------------------------------------------
    async def start(self) -> None:
        self._started = True
        await asyncio.to_thread(
            self.repository.mark_unfinished_sync_runs,
            resume=self.config.sync_resume_on_startup,
        )
        self._closing = False
        self._workers = []
        self._next_worker_index = 0
        while self._next_worker_index < self._target_worker_count:
            index = self._next_worker_index
            self._next_worker_index += 1
            self._workers.append(
                asyncio.create_task(
                    self._worker_loop(index), name=f"public-data-sync-{index}"
                )
            )
        if self.config.auto_sync_enabled and self.config.auto_sync_categories:
            self._scheduler = asyncio.create_task(
                self._schedule_nist_refreshes(), name="public-data-auto-sync"
            )
        self._wake.set()

    # -------------------------------------------------------------------------
    async def close(self) -> None:
        self._closing = True
        if self._scheduler is not None:
            self._scheduler.cancel()
            await asyncio.gather(self._scheduler, return_exceptions=True)
            self._scheduler = None
        # Idle workers stop on their own once their current database call
        # returns; cancelling them would leave that call's thread holding a
        # connection after the engine is disposed. Only running acquisitions
        # are cancelled, which records them as interrupted.
        self._wake.set()
        for worker in self._executing:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    # -------------------------------------------------------------------------
    async def _schedule_nist_refreshes(self) -> None:
        config, _, _ = self._active_settings()
        interval = config.auto_sync_interval_hours * 60 * 60
        while not self._closing:
            await asyncio.sleep(interval)
            if self._closing:
                return
            if self._switch_paused:
                continue
            for category in config.auto_sync_categories:
                await self.enqueue(
                    source_key="nist",
                    operation="nist_category",
                    parameters={
                        "category": category,
                        "fraction": config.auto_sync_fraction,
                        "trigger": "scheduled",
                    },
                )

    # -------------------------------------------------------------------------
    async def enqueue(
        self, *, source_key: str, operation: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        if self._switch_paused:
            raise RuntimeError("Database switching is in progress; retry the sync request shortly.")
        run = await asyncio.to_thread(
            self.repository.create_sync_run,
            source_key=source_key,
            operation=operation,
            parameters=parameters,
        )
        self._wake.set()
        return run

    # -------------------------------------------------------------------------
    async def get(self, run_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self.repository.get_sync_run, run_id)

    # -------------------------------------------------------------------------
    async def list(
        self,
        *,
        limit: int,
        offset: int,
        source_key: str | None,
        status_filter: str | None,
    ) -> tuple[list[dict[str, Any]], int]:
        return await asyncio.to_thread(
            self.repository.list_sync_runs,
            limit=limit,
            offset=offset,
            source_key=source_key,
            status_filter=status_filter,
        )

    # -------------------------------------------------------------------------
    async def cancel(self, run_id: str) -> dict[str, Any]:
        run = await asyncio.to_thread(self.repository.request_sync_cancel, run_id)
        self._wake.set()
        return run

    # -------------------------------------------------------------------------
    async def failures(
        self, run_id: str, *, limit: int, offset: int
    ) -> tuple[list[dict[str, Any]], int]:
        await asyncio.to_thread(self.repository.get_sync_run, run_id)
        return await asyncio.to_thread(
            self.repository.list_sync_failures,
            run_id,
            limit=limit,
            offset=offset,
        )

    # -------------------------------------------------------------------------
    async def nist_collections(self) -> list[dict[str, Any]]:
        """Local NIST counts with the latest remote availability seen by a run."""
        local_counts = await self.nist.local_counts()
        availability = await asyncio.to_thread(
            self.repository.latest_nist_availability
        )
        return [
            {
                "category": category,
                "local_count": int(local_counts.get(category, 0)),
                **availability.get(category, {}),
            }
            for category in NIST_CATEGORIES
        ]

    # -------------------------------------------------------------------------
    async def _worker_loop(self, worker_index: int) -> None:
        worker = asyncio.current_task()
        while not self._closing and worker_index < self._target_worker_count:
            if self._switch_paused:
                await self._wait_for_worker_wake()
                continue
            run, paused = await self._claim_worker_run(worker)
            if paused:
                continue
            if run is None:
                if self._closing:
                    return
                await self._wait_for_worker_wake()
                continue
            if self._closing:
                await self._requeue_worker_run(run, worker)
                return
            await self._execute_worker_run(run, worker)

    # -------------------------------------------------------------------------
    async def _claim_worker_run(
        self, worker: asyncio.Task[None] | None
    ) -> tuple[dict[str, Any] | None, bool]:
        async with self._switch_lock:
            if self._switch_paused:
                return None, True
            run = await asyncio.to_thread(self.repository.claim_next_sync_run)
            if run is not None and worker is not None:
                self._executing.add(worker)
            return run, False

    # -------------------------------------------------------------------------
    async def _wait_for_worker_wake(self) -> None:
        self._wake.clear()
        try:
            await asyncio.wait_for(self._wake.wait(), timeout=1.0)
        except TimeoutError:
            pass

    # -------------------------------------------------------------------------
    async def _requeue_worker_run(
        self, run: dict[str, Any], worker: asyncio.Task[None] | None
    ) -> None:
        await asyncio.to_thread(
            self.repository.update_sync_run,
            str(run["id"]),
            status="queued",
            started_at=None,
        )
        self._executing.discard(worker)

    # -------------------------------------------------------------------------
    async def _execute_worker_run(
        self, run: dict[str, Any], worker: asyncio.Task[None] | None
    ) -> None:
        try:
            await self._execute(run)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._executing.discard(worker)

    # -------------------------------------------------------------------------
    async def _execute(self, run: dict[str, Any]) -> None:
        token = self._run_settings.set((self.config, self.public_data, self.nist))
        try:
            await self._execute_with_settings(run)
        finally:
            self._run_settings.reset(token)

    # -------------------------------------------------------------------------
    async def _execute_with_settings(self, run: dict[str, Any]) -> None:
        run_id = str(run["id"])
        source_key = str(run["source_key"])
        operation = str(run["operation"])
        parameters = dict(run["parameters"])
        state: dict[str, Any] = {
            "counts": {
            key: int(run.get(f"{key}_count", 0))
            for key in ("processed", "imported", "skipped", "failed")
            },
            "checkpoint": dict(run.get("checkpoint", {})),
            "progress": float(run.get("progress", 0.0)),
        }
        _, public_data, _ = self._active_settings()
        provider = public_data.providers.get(source_key)
        state["statistics_reader"] = getattr(provider, "request_statistics", None)
        state["statistics_before"] = (
            state["statistics_reader"]()
            if callable(state["statistics_reader"])
            else {}
        )

        async def update(**values: Any) -> None:
            await asyncio.to_thread(self.repository.update_sync_run, run_id, **values)

        async def should_cancel() -> bool:
            return await asyncio.to_thread(
                self.repository.sync_cancel_requested, run_id
            ) or self._closing

        async def on_progress(payload: dict[str, Any]) -> None:
            state["progress"] = max(
                state["progress"], float(payload.get("progress", 0.0))
            )
            state["checkpoint"] = {
                **state["checkpoint"],
                **self._checkpoint_values(payload),
            }
            counts = state["counts"]
            await update(
                progress=state["progress"],
                processed_count=counts["processed"] + int(payload.get("processed_count", 0)),
                imported_count=counts["imported"] + int(payload.get("imported_count", 0)),
                skipped_count=counts["skipped"] + int(payload.get("skipped_count", 0)),
                failed_count=counts["failed"] + int(payload.get("failed_count", 0)),
                checkpoint=state["checkpoint"],
            )

        try:
            await self._execute_claimed_run(
                run=run,
                state=state,
                on_progress=on_progress,
                should_cancel=should_cancel,
                update=update,
            )
        except asyncio.CancelledError:
            await asyncio.shield(
                update(
                    status="interrupted",
                    progress=state["progress"],
                    error_detail="Application stopped before the run completed.",
                    finished_at=datetime.now(timezone.utc),
                )
            )
            raise
        except Exception as exc:  # noqa: BLE001
            await self._record_operation_failure(
                run_id=run_id,
                source_key=source_key,
                operation=operation,
                parameters=parameters,
                error=exc,
                failed_count=state["counts"]["failed"],
                progress=state["progress"],
                checkpoint=state["checkpoint"],
                update=update,
                statistics_reader=state["statistics_reader"],
                statistics_before=state["statistics_before"],
            )

    # -------------------------------------------------------------------------
    async def _execute_claimed_run(
        self,
        *,
        run: dict[str, Any],
        state: dict[str, Any],
        on_progress: ProgressCallback,
        should_cancel: CancelCheck,
        update: Callable[..., Awaitable[None]],
    ) -> None:
        if await should_cancel():
            await update(status="cancelled", finished_at=datetime.now(timezone.utc))
            return
        outcome = await self._execute_operation(
            run=run,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )
        counts = state["counts"]
        for key in counts:
            counts[key] += int(outcome.get(f"{key}_count", 0))
        state["checkpoint"].update(outcome.get("checkpoint", {}))
        statistics_reader = state["statistics_reader"]
        if "request_stats" not in state["checkpoint"] and callable(statistics_reader):
            state["checkpoint"]["request_stats"] = self._statistics_delta(
                statistics_reader(), state["statistics_before"]
            )
        terminal_status = self._terminal_status(
            counts, cancelled=await should_cancel()
        )
        finished_at = datetime.now(timezone.utc)
        if counts["imported"]:
            await asyncio.to_thread(
                self.repository.mark_source_sync_success,
                str(run["source_key"]),
                finished_at,
            )
        await update(
            status=terminal_status,
            progress=(
                state["progress"] if terminal_status == "cancelled" else 100.0
            ),
            **{f"{key}_count": value for key, value in counts.items()},
            checkpoint=state["checkpoint"],
            finished_at=finished_at,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _terminal_status(counts: dict[str, int], *, cancelled: bool) -> str:
        if cancelled:
            return "cancelled"
        if counts["failed"] and counts["imported"]:
            return "partially_completed"
        if counts["failed"]:
            return "failed"
        return "completed"

    # -------------------------------------------------------------------------
    async def _execute_operation(
        self,
        *,
        run: dict[str, Any],
        on_progress: ProgressCallback,
        should_cancel: CancelCheck,
    ) -> dict[str, Any]:
        operation = str(run["operation"])
        if operation == "health_check":
            return await self._run_health_check(str(run["source_key"]))
        if operation == "nist_index":
            return await self._run_nist_index(run)
        if operation == "nist_category":
            return await self._run_nist_category(
                run, on_progress=on_progress, should_cancel=should_cancel
            )
        if operation == "pubchem_resolve":
            return await self._run_pubchem_resolve(run)
        if operation == "pubchem_enrich":
            return await self._run_pubchem_enrich(
                run, on_progress=on_progress, should_cancel=should_cancel
            )
        if operation == "cod_import":
            return await self._run_cod_import(
                run, on_progress=on_progress, should_cancel=should_cancel
            )
        raise ValueError(f"Unsupported public-data operation: {operation}")

    # -------------------------------------------------------------------------
    async def _run_health_check(self, source_key: str) -> dict[str, Any]:
        _, public_data, _ = self._active_settings()
        provider = public_data.providers[source_key]
        try:
            health = await provider.health()
        except Exception as exc:  # noqa: BLE001
            health = ProviderHealth(
                status="unavailable",
                detail=str(exc),
                checked_at=datetime.now(timezone.utc),
            )
        await asyncio.to_thread(
            self.repository.update_source_health,
            source_key,
            status=health.status,
            detail=health.detail,
            checked_at=health.checked_at,
        )
        return {"processed_count": 1, "checkpoint": {"request_stats": health.request_stats}}

    # -------------------------------------------------------------------------
    async def _run_nist_index(self, run: dict[str, Any]) -> dict[str, Any]:
        _, _, nist = self._active_settings()
        result = await nist.index_category(str(run["parameters"]["category"]))
        return {
            "processed_count": 1,
            "checkpoint": {
                "category": result["category"],
                "available_count": result["available_count"],
                "local_count": result["local_count"],
                "request_stats": dict(result.get("request_stats") or {}),
            },
        }

    # -------------------------------------------------------------------------
    async def _run_nist_category(
        self,
        run: dict[str, Any],
        *,
        on_progress: ProgressCallback,
        should_cancel: CancelCheck,
    ) -> dict[str, Any]:
        run_id = str(run["id"])
        source_key = str(run["source_key"])
        parameters = dict(run["parameters"])
        run_checkpoint = dict(run.get("checkpoint", {}))
        config, _, nist = self._active_settings()

        async def on_failure(external_id: str, reason: str) -> None:
            retry_at = datetime.now(timezone.utc) + timedelta(
                seconds=min(30.0, config.retry_backoff_max_seconds)
            )
            await self._store_failure(run_id, source_key, external_id, reason, retry_at)

        async def on_skip(external_id: str, reason: str) -> None:
            await self._store_failure(run_id, source_key, external_id, reason, None)

        result = await nist.sync_category_records(
            category=str(parameters["category"]),
            fraction=float(parameters.get("fraction", 1.0)),
            batch_size=config.sync_batch_size,
            max_records=config.sync_max_records_per_run,
            processed_before=int(run.get("processed_count", 0)),
            resume_after_external_id=(
                str(run_checkpoint["last_external_id"])
                if run_checkpoint.get("last_external_id")
                else None
            ),
            on_progress=on_progress,
            should_cancel=should_cancel,
            on_failure=on_failure,
            on_skip=on_skip,
        )
        return {
            f"{key}_count": int(result.get(f"{key}_count", 0))
            for key in ("processed", "imported", "skipped", "failed")
        } | {
            "checkpoint": {
                "category": str(parameters["category"]),
                "available_count": int(result.get("available_count", 0)),
                "local_count": int(result.get("local_count", 0)),
                "request_stats": dict(result.get("request_stats") or {}),
            }
        }

    # -------------------------------------------------------------------------
    async def _run_pubchem_resolve(self, run: dict[str, Any]) -> dict[str, Any]:
        parameters = dict(run["parameters"])
        adsorbate_id = parameters.get("adsorbate_id")
        _, public_data, _ = self._active_settings()
        entity_id, imported = await public_data.resolve_pubchem(
            str(parameters["query"]),
            int(adsorbate_id) if adsorbate_id is not None else None,
        )
        return {
            "processed_count": 1,
            "imported_count" if imported else "skipped_count": 1,
            "checkpoint": {"entity_id": entity_id},
        }

    # -------------------------------------------------------------------------
    async def _run_pubchem_enrich(
        self,
        run: dict[str, Any],
        *,
        on_progress: ProgressCallback,
        should_cancel: CancelCheck,
    ) -> dict[str, Any]:
        """Resolve stored chemicals that have no PubChem identity yet."""
        run_id = str(run["id"])
        config, public_data, _ = self._active_settings()
        candidates = await asyncio.to_thread(
            self.repository.pubchem_enrichment_candidates,
            config.sync_max_records_per_run,
        )
        counts = {"processed": 0, "imported": 0, "skipped": 0, "failed": 0}
        entity_ids: list[int] = []
        for candidate in candidates:
            if await should_cancel():
                break
            try:
                entity_id, imported = await public_data.resolve_pubchem(
                    str(candidate["query"]), int(candidate["id"])
                )
                entity_ids.append(entity_id)
                counts["imported" if imported else "skipped"] += 1
            except (ProviderError, ValueError, LookupError) as exc:
                counts["failed"] += 1
                await self._store_failure(
                    run_id,
                    "pubchem",
                    f"adsorbate:{candidate['id']}",
                    f"{candidate['query']}: {exc or exc.__class__.__name__}",
                    self._retry_time(exc),
                )
            counts["processed"] += 1
            await on_progress(
                {
                    "progress": 100.0 * counts["processed"] / len(candidates),
                    **{f"{key}_count": value for key, value in counts.items()},
                    "pending_count": len(candidates) - counts["processed"],
                    "entity_ids": entity_ids,
                }
            )
        return {
            **{f"{key}_count": value for key, value in counts.items()},
            "checkpoint": {
                "entity_ids": entity_ids,
                "candidate_count": len(candidates),
                "pending_count": len(candidates) - counts["processed"],
            },
        }

    # -------------------------------------------------------------------------
    def _retry_time(self, error: Exception) -> datetime | None:
        if not isinstance(error, ProviderUnavailableError):
            return None
        return datetime.now(timezone.utc) + timedelta(
            seconds=min(30.0, self._active_settings()[0].retry_backoff_max_seconds)
        )

    # -------------------------------------------------------------------------
    async def _run_cod_import(
        self,
        run: dict[str, Any],
        *,
        on_progress: ProgressCallback,
        should_cancel: CancelCheck,
    ) -> dict[str, Any]:
        parameters = dict(run["parameters"])
        requested_ids = parameters.get("cod_ids") or [parameters["cod_id"]]
        config, public_data, _ = self._active_settings()
        selected_ids = [
            str(cod_id)
            for cod_id in requested_ids[: config.sync_max_records_per_run]
        ]
        run_id = str(run["id"])
        checkpoint = dict(run.get("checkpoint", {}))
        processed_before = int(run.get("processed_count", 0))
        counts = {"processed": 0, "imported": 0, "skipped": 0, "failed": 0}
        entity_ids = [int(item) for item in checkpoint.get("entity_ids", [])]
        last_external_id = checkpoint.get("last_external_id")
        if last_external_id in selected_ids:
            start_index = selected_ids.index(str(last_external_id)) + 1
        elif last_external_id is None:
            start_index = 0
        else:
            raise ValueError(
                "The saved COD checkpoint does not match the selected identifiers."
            )
        for cod_id in selected_ids[start_index:]:
            if await should_cancel():
                break
            last_external_id = cod_id
            try:
                entity_id, imported = await public_data.import_cod(
                    cod_id, parameters.get("adsorbent_id")
                )
                entity_ids.append(entity_id)
                counts["imported" if imported else "skipped"] += 1
            except (ProviderError, ValueError, LookupError) as exc:
                counts["failed"] += 1
                await self._store_failure(
                    run_id,
                    "cod",
                    cod_id,
                    str(exc) or exc.__class__.__name__,
                    self._retry_time(exc),
                )
            counts["processed"] += 1
            processed_count = processed_before + counts["processed"]
            await on_progress(
                {
                    "progress": 100.0 * processed_count / max(1, len(selected_ids)),
                    **{f"{key}_count": value for key, value in counts.items()},
                    "pending_count": max(0, len(requested_ids) - processed_count),
                    "last_external_id": last_external_id,
                    "entity_ids": entity_ids,
                }
            )
        return {
            **{f"{key}_count": value for key, value in counts.items()},
            "checkpoint": {
                **({"entity_id": entity_ids[0]} if len(requested_ids) == 1 and entity_ids else {}),
                "entity_ids": entity_ids,
                "last_external_id": last_external_id,
                "pending_count": max(
                    0,
                    len(requested_ids) - (processed_before + counts["processed"]),
                ),
            },
        }

    # -------------------------------------------------------------------------
    async def _store_failure(
        self,
        run_id: str,
        source_key: str,
        external_id: str | None,
        reason: str,
        retry_eligible_at: datetime | None,
    ) -> None:
        await asyncio.to_thread(
            self.repository.add_sync_failure,
            sync_run_id=run_id,
            source_key=source_key,
            external_id=external_id,
            reason=reason,
            retry_eligible_at=retry_eligible_at,
        )

    # -------------------------------------------------------------------------
    async def _record_operation_failure(
        self,
        *,
        run_id: str,
        source_key: str,
        operation: str,
        parameters: dict[str, Any],
        error: Exception,
        failed_count: int,
        progress: float,
        checkpoint: dict[str, Any],
        update: Callable[..., Awaitable[None]],
        statistics_reader: Any,
        statistics_before: dict[str, int],
    ) -> None:
        message = str(error) or error.__class__.__name__
        now = datetime.now(timezone.utc)
        await self._store_failure(
            run_id,
            source_key,
            self._failure_identifier(operation, parameters, checkpoint),
            message,
            self._retry_time(error),
        )
        if callable(statistics_reader) and "request_stats" not in checkpoint:
            checkpoint["request_stats"] = self._statistics_delta(
                statistics_reader(), statistics_before
            )
        await update(
            status="failed",
            progress=progress,
            failed_count=max(1, failed_count),
            checkpoint=checkpoint,
            error_detail=message,
            finished_at=now,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _checkpoint_values(payload: dict[str, Any]) -> dict[str, Any]:
        counters = {
            "progress",
            "processed_count",
            "imported_count",
            "skipped_count",
            "failed_count",
        }
        return {key: value for key, value in payload.items() if key not in counters}

    # -------------------------------------------------------------------------
    @staticmethod
    def _statistics_delta(
        current: dict[str, int], previous: dict[str, int]
    ) -> dict[str, int]:
        return {
            key: max(0, int(value) - int(previous.get(key, 0)))
            for key, value in current.items()
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def _failure_identifier(
        operation: str, parameters: dict[str, Any], checkpoint: dict[str, Any]
    ) -> str | None:
        """Name the record a run-level failure concerns, when one is known."""
        if operation == "pubchem_resolve":
            return str(parameters.get("query") or "") or None
        if operation == "cod_import":
            identifiers = parameters.get("cod_ids") or [parameters.get("cod_id")]
            return ", ".join(str(item) for item in identifiers if item) or None
        if operation in ("nist_index", "nist_category"):
            return (
                str(checkpoint.get("last_external_id") or "")
                or str(parameters.get("category") or "")
                or None
            )
        return None


__all__ = ["PublicDataSyncCoordinator"]
