# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from server.configurations.settings import AdsmodConfig, DatabaseConfig, PublicDataConfig, UserSettings
from server.domain.public_data import PublicDataSyncCreateRequest
from server.repositories.database.manager import DatabaseManager
from server.repositories.public_data import PublicDataRepository
from server.repositories.schemas import Base
from server.services.data.public_data import PublicDataService
from server.services.data.public_data_sync import PublicDataSyncCoordinator
from server.services.data.nist_service import NISTDataService
from server.services.data.nistads import NISTApiClient
from server.services.providers.nist import NISTPublicDataProvider
from server.services.providers import public_data as public_data_provider_module
from server.services.providers.pubchem import PubChemProvider
from server.services.providers.public_data import (
    ProviderCapability,
    ProviderUnavailableError,
    ProviderHealth,
    PublicDataProvider,
    RequestBudget,
)
from server.services.providers import PROVIDER_CLASSES

PUBLIC_SOURCE_DEFINITIONS = [provider.source_definition() for provider in PROVIDER_CLASSES]

###############################################################################
def _database(path: Path) -> DatabaseManager:
    manager = DatabaseManager(
        DatabaseConfig(
            embedded_database=True,
            connect_timeout=5,
            sqlite_path=str(path / "public-data-sync.db"),
        )
    )
    Base.metadata.create_all(manager.engine)
    return manager

###############################################################################
class _Provider(PublicDataProvider):
    key = "nist"
    name = "NIST test provider"
    description = "Deterministic provider for coordinator tests."
    homepage_url = "https://adsorption.nist.gov/"
    license_name = None
    license_url = None
    terms_url = None
    capabilities = (ProviderCapability.ADSORPTION,)

    # -------------------------------------------------------------------------
    async def health(self) -> ProviderHealth:
        return ProviderHealth(
            status="available", detail=None, checked_at=datetime.now(timezone.utc)
        )

###############################################################################
class _Nist:

    # -------------------------------------------------------------------------
    async def sync_category_records(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        raise NotImplementedError

###############################################################################
async def _wait_for_terminal(
    coordinator: PublicDataSyncCoordinator, run_id: str
) -> dict[str, Any]:
    for _ in range(300):
        run = await coordinator.get(run_id)
        if run["status"] not in {"queued", "running"}:
            return run
        await asyncio.sleep(0.01)
    raise AssertionError(f"Sync run {run_id} did not reach a terminal state.")

###############################################################################
def test_coordinator_persists_checkpoints_partial_failures_and_skip_reasons(
    tmp_path: Path,
) -> None:

    ###############################################################################
    class PartialNist:

        # -------------------------------------------------------------------------
        async def sync_category_records(self, **kwargs: Any) -> dict[str, Any]:
            await kwargs["on_failure"]("upstream-404", "NIST record request failed.")
            await kwargs["on_skip"]("unsupported-1", "Unsupported uptake unit: foo.")
            await kwargs["on_progress"](
                {
                    "progress": 50.0,
                    "processed_count": 2,
                    "imported_count": 1,
                    "skipped_count": 1,
                    "failed_count": 1,
                    "category": "experiments",
                    "pending_count": 3,
                    "last_external_id": "record-2",
                }
            )
            return {
                "processed_count": 2,
                "imported_count": 1,
                "skipped_count": 1,
                "failed_count": 1,
                "request_stats": {
                    "requests": 4,
                    "retries": 1,
                    "rate_limits": 0,
                    "failures": 1,
                },
            }

    async def scenario() -> None:
        manager = _database(tmp_path)
        repository = PublicDataRepository(manager)
        repository.ensure_sources(PUBLIC_SOURCE_DEFINITIONS)
        public_data = PublicDataService(repository=repository, providers=[_Provider()])
        coordinator = PublicDataSyncCoordinator(
            repository=repository,
            public_data=public_data,
            nist=PartialNist(),  # type: ignore[arg-type]
            config=PublicDataConfig(sync_max_concurrent_jobs=1),
        )
        await coordinator.start()
        try:
            queued = await coordinator.enqueue(
                source_key="nist",
                operation="nist_category",
                parameters={"category": "experiments", "fraction": 1.0},
            )
            run = await _wait_for_terminal(coordinator, queued["id"])
            failures, total = await coordinator.failures(run["id"], limit=10, offset=0)

            assert run["status"] == "partially_completed"
            assert run["processed_count"] == 2
            assert run["imported_count"] == 1
            assert run["skipped_count"] == 1
            assert run["failed_count"] == 1
            assert run["checkpoint"]["pending_count"] == 3
            assert run["checkpoint"]["last_external_id"] == "record-2"
            assert run["checkpoint"]["request_stats"]["retries"] == 1
            assert total == 2
            assert {failure["external_id"] for failure in failures} == {
                "upstream-404",
                "unsupported-1",
            }
            assert next(
                failure["reason"]
                for failure in failures
                if failure["external_id"] == "unsupported-1"
            ) == "Unsupported uptake unit: foo."
        finally:
            await coordinator.close()
            manager.dispose()

    asyncio.run(scenario())

###############################################################################
def test_coordinator_cancels_between_nist_batches(tmp_path: Path) -> None:
    async def scenario() -> None:
        started = asyncio.Event()

        ###############################################################################
        class BlockingNist:

            # -------------------------------------------------------------------------
            async def sync_category_records(self, **kwargs: Any) -> dict[str, Any]:
                started.set()
                while not await kwargs["should_cancel"]():
                    await asyncio.sleep(0.01)
                return {
                    "processed_count": 1,
                    "imported_count": 0,
                    "skipped_count": 0,
                    "failed_count": 0,
                    "request_stats": {"requests": 1, "retries": 0, "rate_limits": 0, "failures": 0},
                }

        manager = _database(tmp_path)
        repository = PublicDataRepository(manager)
        repository.ensure_sources(PUBLIC_SOURCE_DEFINITIONS)
        public_data = PublicDataService(repository=repository, providers=[_Provider()])
        coordinator = PublicDataSyncCoordinator(
            repository=repository,
            public_data=public_data,
            nist=BlockingNist(),  # type: ignore[arg-type]
            config=PublicDataConfig(),
        )
        await coordinator.start()
        try:
            queued = await coordinator.enqueue(
                source_key="nist",
                operation="nist_category",
                parameters={"category": "guest", "fraction": 1.0},
            )
            await asyncio.wait_for(started.wait(), timeout=1.0)
            await coordinator.cancel(queued["id"])
            run = await _wait_for_terminal(coordinator, queued["id"])
            assert run["status"] == "cancelled"
            assert run["processed_count"] == 1
        finally:
            await coordinator.close()
            manager.dispose()

    asyncio.run(scenario())

###############################################################################
def test_coordinator_reuses_local_pubchem_identity_without_a_provider_call(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    async def scenario() -> None:
        manager = _database(tmp_path)
        repository = PublicDataRepository(manager)
        repository.ensure_sources(PUBLIC_SOURCE_DEFINITIONS)
        adsorbate_id = repository.upsert_pubchem_compound(
            {
                "cid": "297",
                "name": "Methane",
                "formula": "CH4",
                "inchi_key": "VNWKTOKETHGBQD-UHFFFAOYSA-N",
                "source_url": "https://pubchem.ncbi.nlm.nih.gov/compound/297",
                "raw_metadata": {},
            }
        )
        provider = PubChemProvider(parallel_requests=1)
        resolved: list[str] = []

        async def unexpected_resolution(query: str) -> dict[str, Any]:
            resolved.append(query)
            raise AssertionError("local PubChem identity should have been reused")

        monkeypatch.setattr(provider, "resolve", unexpected_resolution)
        public_data = PublicDataService(repository=repository, providers=[provider])
        coordinator = PublicDataSyncCoordinator(
            repository=repository,
            public_data=public_data,
            nist=_Nist(),
            config=PublicDataConfig(),
        )
        await coordinator.start()
        try:
            queued = await coordinator.enqueue(
                source_key="pubchem",
                operation="pubchem_resolve",
                parameters={"query": "VNWKTOKETHGBQD-UHFFFAOYSA-N"},
            )
            run = await _wait_for_terminal(coordinator, queued["id"])
            assert run["status"] == "completed"
            assert run["imported_count"] == 0
            assert run["skipped_count"] == 1
            assert run["checkpoint"]["entity_id"] == adsorbate_id
            assert resolved == []
        finally:
            await coordinator.close()
            await public_data.close()
            manager.dispose()

    asyncio.run(scenario())

###############################################################################
def test_cod_batch_import_is_bounded_and_keeps_record_failures(tmp_path: Path) -> None:
    async def scenario() -> None:
        manager = _database(tmp_path)
        repository = PublicDataRepository(manager)
        repository.ensure_sources(PUBLIC_SOURCE_DEFINITIONS)
        public_data = PublicDataService(repository=repository, providers=[_Provider()])
        imported: list[str] = []

        async def import_cod(cod_id: str, adsorbent_id: int | None = None) -> Any:
            del adsorbent_id
            if cod_id == "9012346":
                raise ProviderUnavailableError("COD CIF request timed out.")
            imported.append(cod_id)
            return len(imported), True

        public_data.import_cod = import_cod  # type: ignore[method-assign]
        coordinator = PublicDataSyncCoordinator(
            repository=repository,
            public_data=public_data,
            nist=_Nist(),  # type: ignore[arg-type]
            config=PublicDataConfig(sync_max_records_per_run=2),
        )
        await coordinator.start()
        try:
            request = PublicDataSyncCreateRequest.model_validate(
                {
                    "source_key": "cod",
                    "operation": "cod_import",
                    "cod_ids": ["9012345", "9012346", "9012347"],
                }
            )
            queued = await coordinator.enqueue(
                source_key=request.source_key,
                operation=request.operation,
                parameters=request.parameters(),
            )
            run = await _wait_for_terminal(coordinator, queued["id"])
            failures, total = await coordinator.failures(
                run["id"], limit=10, offset=0
            )

            assert run["status"] == "partially_completed"
            assert run["processed_count"] == 2
            assert run["imported_count"] == 1
            assert run["failed_count"] == 1
            assert run["checkpoint"]["pending_count"] == 1
            assert run["checkpoint"]["entity_ids"] == [1]
            assert imported == ["9012345"]
            assert total == 1
            assert failures[0]["external_id"] == "9012346"
            assert failures[0]["retry_eligible_at"] is not None
        finally:
            await coordinator.close()
            manager.dispose()

    asyncio.run(scenario())

###############################################################################
def test_cod_batch_request_rejects_duplicates_and_more_than_fifty_ids() -> None:
    valid = PublicDataSyncCreateRequest.model_validate(
        {
            "source_key": "cod",
            "operation": "cod_import",
            "cod_ids": ["9012345", "9012346"],
        }
    )
    assert valid.cod_ids == ["9012345", "9012346"]

    for cod_ids in (["9012345", "9012345"], [str(9000000 + index) for index in range(51)]):
        try:
            PublicDataSyncCreateRequest.model_validate(
                {
                    "source_key": "cod",
                    "operation": "cod_import",
                    "cod_ids": cod_ids,
                }
            )
        except ValueError:
            continue
        raise AssertionError("Invalid COD batch request was accepted.")

###############################################################################
def test_concurrent_sync_run_creation_serializes_sqlite_writes(tmp_path: Path) -> None:
    manager = _database(tmp_path)
    try:
        repository = PublicDataRepository(manager)
        repository.ensure_sources(PUBLIC_SOURCE_DEFINITIONS)

        async def create_runs() -> list[dict[str, Any]]:
            return await asyncio.gather(
                *(
                    asyncio.to_thread(
                        repository.create_sync_run,
                        source_key="nist",
                        operation="health_check",
                        parameters={"probe": index},
                    )
                    for index in range(12)
                )
            )

        runs = asyncio.run(create_runs())

        assert len(runs) == 12
        assert len({run["id"] for run in runs}) == 12
        assert all(run["status"] == "queued" for run in runs)
    finally:
        manager.dispose()

###############################################################################
def test_identical_active_requests_reuse_the_existing_sync_run(
    tmp_path: Path,
) -> None:
    manager = _database(tmp_path)
    try:
        repository = PublicDataRepository(manager)
        repository.ensure_sources(PUBLIC_SOURCE_DEFINITIONS)

        first = repository.create_sync_run(
            source_key="nist",
            operation="nist_category",
            parameters={
                "category": "experiments",
                "fraction": 0.25,
                "trigger": "manual",
            },
        )
        duplicate = repository.create_sync_run(
            source_key="nist",
            operation="nist_category",
            parameters={
                "category": "experiments",
                "fraction": 0.25,
                "trigger": "scheduled",
            },
        )
        different_scope = repository.create_sync_run(
            source_key="nist",
            operation="nist_category",
            parameters={"category": "experiments", "fraction": 1.0},
        )
        assert duplicate["id"] == first["id"]
        assert different_scope["id"] != first["id"]

        repository.update_sync_run(first["id"], status="completed")
        after_completion = repository.create_sync_run(
            source_key="nist",
            operation="nist_category",
            parameters={"category": "experiments", "fraction": 0.25},
        )
        assert after_completion["id"] != first["id"]

        first_cod = repository.create_sync_run(
            source_key="cod",
            operation="cod_import",
            parameters={"cod_ids": ["1000002", "1000001"]},
        )
        reordered_cod = repository.create_sync_run(
            source_key="cod",
            operation="cod_import",
            parameters={"cod_ids": ["1000001", "1000002"]},
        )
        assert reordered_cod["id"] == first_cod["id"]
    finally:
        manager.dispose()

###############################################################################
def test_sync_run_scopes_and_overview_exclude_availability_probes(
    tmp_path: Path,
) -> None:
    manager = _database(tmp_path)
    try:
        repository = PublicDataRepository(manager)
        repository.ensure_sources(PUBLIC_SOURCE_DEFINITIONS)
        active = repository.create_sync_run(
            source_key="nist",
            operation="nist_category",
            parameters={"category": "guest"},
        )
        completed = repository.create_sync_run(
            source_key="nist",
            operation="nist_category",
            parameters={"category": "host"},
        )
        failed = repository.create_sync_run(
            source_key="pubchem",
            operation="pubchem_resolve",
            parameters={"query": "methane"},
        )
        health = repository.create_sync_run(
            source_key="cod",
            operation="health_check",
            parameters={},
        )
        index = repository.create_sync_run(
            source_key="nist",
            operation="nist_index",
            parameters={"category": "experiments"},
        )
        failed_index = repository.create_sync_run(
            source_key="nist",
            operation="nist_index",
            parameters={"category": "guest"},
        )
        repository.update_sync_run(completed["id"], status="completed")
        repository.update_sync_run(failed["id"], status="failed")
        repository.update_sync_run(health["id"], status="completed")
        repository.update_sync_run(index["id"], status="completed")
        repository.update_sync_run(failed_index["id"], status="failed")

        active_runs, active_total = repository.list_sync_runs(
            status_scope="active"
        )
        history_runs, history_total = repository.list_sync_runs(
            status_scope="history"
        )

        assert active_total == 1
        assert [run["id"] for run in active_runs] == [active["id"]]
        assert history_total == 2
        assert {run["id"] for run in history_runs} == {
            completed["id"],
            failed["id"],
        }
        assert all(
            run["operation"] not in ("health_check", "nist_index")
            for run in history_runs
        )
        overview = repository.overview_counts()
        assert overview["active_sync_runs"] == 1
        assert overview["failed_sync_runs"] == 1
        assert overview["latest_successful_sync_at"] is None
    finally:
        manager.dispose()

###############################################################################
def test_nist_availability_survives_failed_refresh(tmp_path: Path) -> None:
    manager = _database(tmp_path)
    try:
        repository = PublicDataRepository(manager)
        repository.ensure_sources(PUBLIC_SOURCE_DEFINITIONS)
        good = repository.create_sync_run(
            source_key="nist",
            operation="nist_index",
            parameters={"category": "guest"},
        )
        repository.update_sync_run(
            good["id"],
            status="completed",
            checkpoint={"category": "guest", "available_count": 42},
            finished_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        failed = repository.create_sync_run(
            source_key="nist",
            operation="nist_index",
            parameters={"category": "guest"},
        )
        repository.update_sync_run(
            failed["id"], status="failed", finished_at=datetime(2026, 1, 2, tzinfo=timezone.utc)
        )

        cached = repository.latest_nist_availability()["guest"]

        assert cached["available_count"] == 42
        assert cached["indexed_at"] == datetime(2026, 1, 1, tzinfo=timezone.utc)
    finally:
        manager.dispose()

###############################################################################
def test_resume_interrupted_sync_run_preserves_checkpoint_and_counters(
    tmp_path: Path,
) -> None:
    manager = _database(tmp_path)
    try:
        repository = PublicDataRepository(manager)
        repository.ensure_sources(PUBLIC_SOURCE_DEFINITIONS)
        run = repository.create_sync_run(
            source_key="nist",
            operation="nist_category",
            parameters={"category": "experiments", "fraction": 1.0},
        )
        repository.update_sync_run(
            run["id"],
            status="running",
            progress=42.0,
            processed_count=100,
            imported_count=90,
            skipped_count=6,
            failed_count=4,
            checkpoint={
                "last_external_id": "record-100",
                "pending_count": 50,
            },
        )
        assert repository.mark_unfinished_sync_runs(resume=False) == 1

        resumed = repository.resume_sync_run(run["id"])
        assert resumed["status"] == "queued"
        assert resumed["progress"] == 42.0
        assert resumed["processed_count"] == 100
        assert resumed["imported_count"] == 90
        assert resumed["skipped_count"] == 6
        assert resumed["failed_count"] == 4
        assert resumed["checkpoint"] == {
            "last_external_id": "record-100",
            "pending_count": 50,
        }
        assert resumed["started_at"] is None
        assert resumed["finished_at"] is None
        assert resumed["error_detail"] is None

        with pytest.raises(ValueError):
            repository.resume_sync_run(run["id"])

        repository.update_sync_run(
            run["id"], status="interrupted", error_detail="stopped"
        )
        repository.create_sync_run(
            source_key="nist",
            operation="nist_category",
            parameters={"category": "experiments", "fraction": 1.0},
        )
        with pytest.raises(ValueError):
            repository.resume_sync_run(run["id"])
    finally:
        manager.dispose()

###############################################################################
def test_nist_restart_resumes_after_checkpoint_without_recounting_prior_batches(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manager = _database(tmp_path)
        repository = PublicDataRepository(manager)
        repository.ensure_sources(PUBLIC_SOURCE_DEFINITIONS)
        run = repository.create_sync_run(
            source_key="nist",
            operation="nist_category",
            parameters={"category": "experiments", "fraction": 1.0},
        )
        repository.update_sync_run(
            run["id"],
            status="running",
            progress=50.0,
            processed_count=100,
            imported_count=90,
            skipped_count=5,
            failed_count=5,
            checkpoint={"last_external_id": "record-100", "pending_count": 100},
        )
        assert repository.mark_unfinished_sync_runs(resume=True) == 1
        public_data = PublicDataService(repository=repository, providers=[_Provider()])
        observed: dict[str, Any] = {}

        ###############################################################################
        class ResumingNist:

            # -------------------------------------------------------------------------
            async def sync_category_records(self, **kwargs: Any) -> dict[str, Any]:
                observed.update(
                    processed_before=kwargs["processed_before"],
                    resume_after_external_id=kwargs["resume_after_external_id"],
                    max_records=kwargs["max_records"],
                )
                await kwargs["on_progress"](
                    {
                        "progress": 100.0,
                        "processed_count": 2,
                        "imported_count": 1,
                        "skipped_count": 1,
                        "failed_count": 0,
                        "last_external_id": "record-102",
                        "pending_count": 0,
                    }
                )
                return {
                    "processed_count": 2,
                    "imported_count": 1,
                    "skipped_count": 1,
                    "failed_count": 0,
                    "request_stats": {"requests": 2, "retries": 0},
                }

        coordinator = PublicDataSyncCoordinator(
            repository=repository,
            public_data=public_data,
            nist=ResumingNist(),  # type: ignore[arg-type]
            config=PublicDataConfig(
                sync_max_records_per_run=200,
                sync_resume_on_startup=True,
            ),
        )
        await coordinator.start()
        try:
            resumed = await _wait_for_terminal(coordinator, run["id"])
            assert observed == {
                "processed_before": 100,
                "resume_after_external_id": "record-100",
                "max_records": None,
            }
            assert resumed["status"] == "partially_completed"
            assert resumed["progress"] == 100.0
            assert resumed["processed_count"] == 102
            assert resumed["imported_count"] == 91
            assert resumed["skipped_count"] == 6
            assert resumed["failed_count"] == 5
            assert resumed["checkpoint"]["last_external_id"] == "record-102"
        finally:
            await coordinator.close()
            await public_data.close()
            manager.dispose()

    asyncio.run(scenario())

###############################################################################
def test_cod_restart_continues_after_checkpoint_without_reimporting_saved_ids(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manager = _database(tmp_path)
        repository = PublicDataRepository(manager)
        repository.ensure_sources(PUBLIC_SOURCE_DEFINITIONS)
        run = repository.create_sync_run(
            source_key="cod",
            operation="cod_import",
            parameters={
                "cod_ids": ["9012345", "9012346", "9012347"],
            },
        )
        repository.update_sync_run(
            run["id"],
            status="running",
            progress=33.0,
            processed_count=1,
            imported_count=1,
            checkpoint={
                "last_external_id": "9012345",
                "entity_ids": [77],
                "pending_count": 2,
            },
        )
        assert repository.mark_unfinished_sync_runs(resume=True) == 1
        local_ids = {"9012345": 77}
        imported: list[str] = []

        def find_local(cod_id: str) -> int | None:
            return local_ids.get(cod_id)

        repository.find_cod_structure_id = find_local  # type: ignore[method-assign]
        public_data = PublicDataService(repository=repository, providers=[_Provider()])

        async def import_cod(cod_id: str, adsorbent_id: int | None = None) -> Any:
            del adsorbent_id
            imported.append(cod_id)
            structure_id = 100 + len(imported)
            local_ids[cod_id] = structure_id
            return structure_id, True

        public_data.import_cod = import_cod  # type: ignore[method-assign]
        coordinator = PublicDataSyncCoordinator(
            repository=repository,
            public_data=public_data,
            nist=_Nist(),  # type: ignore[arg-type]
            config=PublicDataConfig(sync_resume_on_startup=True),
        )
        await coordinator.start()
        try:
            resumed = await _wait_for_terminal(coordinator, run["id"])
            assert imported == ["9012346", "9012347"]
            assert resumed["status"] == "completed"
            assert resumed["processed_count"] == 3
            assert resumed["imported_count"] == 3
            assert resumed["skipped_count"] == 0
            assert resumed["checkpoint"]["entity_ids"] == [77, 101, 102]
            assert len(local_ids) == 3
        finally:
            await coordinator.close()
            await public_data.close()
            manager.dispose()

###############################################################################
def test_restart_policy_preserves_runs_and_exposes_resume_choice(tmp_path: Path) -> None:
    manager = _database(tmp_path)
    try:
        repository = PublicDataRepository(manager)
        repository.ensure_sources(PUBLIC_SOURCE_DEFINITIONS)
        queued = repository.create_sync_run(
            source_key="nist", operation="nist_category", parameters={"category": "guest"}
        )
        running = repository.create_sync_run(
            source_key="nist", operation="nist_category", parameters={"category": "host"}
        )
        repository.update_sync_run(running["id"], status="running", progress=35.0)

        assert repository.mark_unfinished_sync_runs(resume=False) == 2
        assert repository.get_sync_run(queued["id"])["status"] == "interrupted"
        assert repository.get_sync_run(running["id"])["status"] == "interrupted"

        resumable = repository.create_sync_run(
            source_key="nist", operation="nist_category", parameters={"category": "experiments"}
        )
        repository.update_sync_run(resumable["id"], status="running", progress=35.0)
        assert repository.mark_unfinished_sync_runs(resume=True) == 1
        restored = repository.get_sync_run(resumable["id"])
        assert restored["status"] == "queued"
        assert restored["progress"] == 35.0
    finally:
        manager.dispose()

###############################################################################
def test_pubchem_request_budget_is_shared_across_provider_instances(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    clock = [0.0]
    delays: list[float] = []
    budget = RequestBudget(minimum_interval_seconds=0.22)

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)
        clock[0] += delay

    monkeypatch.setattr(public_data_provider_module, "monotonic", lambda: clock[0])
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    first = PubChemProvider(parallel_requests=1, request_budget=budget)
    second = PubChemProvider(parallel_requests=1, request_budget=budget)

    async def scenario() -> None:
        await first._before_attempt()
        await second._before_attempt()

    asyncio.run(scenario())
    assert delays == [0.22]

###############################################################################
def test_optional_scheduler_enqueues_only_configured_nist_collections(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    config = PublicDataConfig(
        auto_sync_enabled=True,
        auto_sync_categories=("guest", "host"),
        auto_sync_interval_hours=1,
        auto_sync_fraction=0.5,
    )
    coordinator = PublicDataSyncCoordinator(
        repository=object(),  # type: ignore[arg-type]
        public_data=object(),  # type: ignore[arg-type]
        nist=object(),  # type: ignore[arg-type]
        config=config,
    )
    scheduled: list[dict[str, Any]] = []

    async def fake_sleep(_seconds: float) -> None:
        return None

    async def capture_enqueue(**kwargs: Any) -> dict[str, Any]:
        scheduled.append(kwargs)
        coordinator._closing = True
        return {}

    coordinator.enqueue = capture_enqueue  # type: ignore[method-assign]
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    asyncio.run(coordinator._schedule_nist_refreshes())

    assert [run["parameters"]["category"] for run in scheduled] == ["guest", "host"]
    assert all(run["parameters"]["fraction"] == 0.5 for run in scheduled)
    assert all(run["parameters"]["trigger"] == "scheduled" for run in scheduled)

###############################################################################
def test_nist_streaming_client_retries_with_retry_after_and_records_statistics(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    url = "https://adsorption.nist.gov/isodb/api/gases/test.json"
    responses = [
        httpx.Response(429, headers={"Retry-After": "1.5"}),
        httpx.Response(200, json={"name": "methane"}),
    ]
    delays: list[float] = []

    ###############################################################################
    class _Stream:

        # -------------------------------------------------------------------------
        def __init__(self, response: Any) -> None:
            self.response = response

        # -------------------------------------------------------------------------
        async def __aenter__(self) -> Any:
            return self.response

        # -------------------------------------------------------------------------
        async def __aexit__(self, *_args: Any) -> None:
            return None

    ###############################################################################
    class _Client:

        # -------------------------------------------------------------------------
        def stream(self, *_args: Any, **_kwargs: Any) -> _Stream:
            return _Stream(responses.pop(0))

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    client = NISTApiClient(2, retry_attempts=2)
    payload = asyncio.run(client.fetch_json(_Client(), url))  # type: ignore[arg-type]

    assert payload == {"name": "methane"}
    assert delays == [1.5]
    assert client.request_statistics() == {
        "requests": 2,
        "retries": 1,
        "rate_limits": 1,
        "failures": 0,
    }

###############################################################################
def test_nist_streaming_client_rejects_oversized_response() -> None:

    ###############################################################################
    class _Stream:

        # -------------------------------------------------------------------------
        async def __aenter__(self) -> httpx.Response:
            return httpx.Response(
                200,
                text="x" * 2048,
                request=httpx.Request(
                    "GET", "https://adsorption.nist.gov/isodb/api/gases/test.json"
                ),
            )

        # -------------------------------------------------------------------------
        async def __aexit__(self, *_args: Any) -> None:
            return None

    ###############################################################################
    class _Client:

        # -------------------------------------------------------------------------
        def stream(self, *_args: Any, **_kwargs: Any) -> _Stream:
            return _Stream()

    client = NISTApiClient(1, retry_attempts=1, max_response_bytes=1024)
    assert (
        asyncio.run(
            client.fetch_json(
                _Client(), "https://adsorption.nist.gov/isodb/api/gases/test.json"  # type: ignore[arg-type]
            )
        )
        is None
    )
    assert client.request_statistics()["failures"] == 1

###############################################################################
def test_nist_health_probe_requests_only_a_single_byte_range(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[tuple[str, str, dict[str, str]]] = []

    ###############################################################################
    class _Stream:

        # -------------------------------------------------------------------------
        async def __aenter__(self) -> httpx.Response:
            return httpx.Response(
                206,
                headers={"Content-Range": "bytes 0-0/2000000"},
                request=httpx.Request(
                    "GET", "https://adsorption.nist.gov/isodb/api/isotherms.json"
                ),
            )

        # -------------------------------------------------------------------------
        async def __aexit__(self, *_args: Any) -> None:
            return None

    ###############################################################################
    class _Client:

        # -------------------------------------------------------------------------
        def __init__(self, **_kwargs: Any) -> None:
            pass

        # -------------------------------------------------------------------------
        def stream(
            self, method: str, url: str, *, headers: dict[str, str]
        ) -> _Stream:
            calls.append((method, url, headers))
            return _Stream()

        # -------------------------------------------------------------------------
        async def __aenter__(self) -> "_Client":
            return self

        # -------------------------------------------------------------------------
        async def __aexit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    service = NISTDataService(
        config=AdsmodConfig.assemble(
            database=DatabaseConfig(),
            storage_root=Path("/tmp/adsmod"),
            settings=UserSettings(),
        ),
        repository=object(),  # type: ignore[arg-type]
        mapper=object(),  # type: ignore[arg-type]
    )
    health = asyncio.run(NISTPublicDataProvider(service).health())

    assert health.status == "available"
    assert calls == [
        (
            "GET",
            "https://adsorption.nist.gov/isodb/api/isotherms.json",
            {"Range": "bytes=0-0"},
        )
    ]

###############################################################################
def test_coordinator_close_waits_for_in_flight_claim_and_requeues_it() -> None:
    claim_started = threading.Event()
    release_claim = threading.Event()
    updates: list[tuple[str, dict[str, Any]]] = []

    ###############################################################################
    class _BlockingRepository:

        # -------------------------------------------------------------------------
        def mark_unfinished_sync_runs(self, *, resume: bool) -> int:
            del resume
            return 0

        # -------------------------------------------------------------------------
        def claim_next_sync_run(self) -> dict[str, Any]:
            claim_started.set()
            release_claim.wait(timeout=5)
            return {"id": "run-claimed-at-shutdown"}

        # -------------------------------------------------------------------------
        def update_sync_run(self, run_id: str, **values: Any) -> None:
            updates.append((run_id, values))

    coordinator = PublicDataSyncCoordinator(
        repository=_BlockingRepository(),  # type: ignore[arg-type]
        public_data=object(),  # type: ignore[arg-type]
        nist=object(),  # type: ignore[arg-type]
        config=PublicDataConfig(sync_max_concurrent_jobs=1),
    )

    async def scenario() -> None:
        await coordinator.start()
        await asyncio.to_thread(claim_started.wait, 5)
        closing = asyncio.create_task(coordinator.close())
        await asyncio.sleep(0.05)
        # Returning now would leave the claim thread using the database.
        assert not closing.done()
        release_claim.set()
        await asyncio.wait_for(closing, timeout=5)

    asyncio.run(scenario())
    assert updates == [
        ("run-claimed-at-shutdown", {"status": "queued", "started_at": None})
    ]

###############################################################################
def test_nist_search_joins_indexes_by_name_synonym_and_experiment() -> None:
    class _Repository:
        @staticmethod
        def list_nist_experiment_ids() -> set[str]:
            return {"iso-1"}

    service = NISTDataService(None, _Repository(), None)  # type: ignore[arg-type]
    service._search_cache = {
        "at": 10**12,
        "experiments": [
            {
                "filename": "ISO-1",
                "DOI": "10.1/abc",
                "adsorbent": {"hashkey": "H1"},
                "adsorbates": [{"InChIKey": "K-CO2"}],
                "temperature": 298,
            },
            {
                "filename": "iso-2",
                "DOI": "10.1/xyz",
                "adsorbent": {"hashkey": "H2"},
                "adsorbates": [{"InChIKey": "K-CH4"}],
                "temperature": 350,
            },
        ],
        "guests": {
            "K-CO2": ("Carbon Dioxide", ["carbon dioxide", "co2"]),
            "K-CH4": ("Methane", ["methane"]),
        },
        "hosts": {"H1": ("ZIF-8", ["zif-8"]), "H2": ("MOF-5", ["mof-5"])},
    }

    async def search(**filters: Any) -> dict[str, Any]:
        base = {
            "adsorbate": None,
            "adsorbent": None,
            "experiment": None,
            "temperature_min": None,
            "temperature_max": None,
            "limit": 25,
            "offset": 0,
        }
        return await service.search_experiments(**{**base, **filters})

    async def scenario() -> None:
        by_synonym = await search(adsorbate="CO2")
        assert [item["filename"] for item in by_synonym["items"]] == ["ISO-1"]
        assert by_synonym["items"][0]["adsorbate_names"] == ["Carbon Dioxide"]
        assert by_synonym["items"][0]["adsorbent_name"] == "ZIF-8"
        assert by_synonym["items"][0]["local"] is True

        by_material = await search(adsorbent="mof-5")
        assert [item["filename"] for item in by_material["items"]] == ["iso-2"]
        assert by_material["items"][0]["local"] is False

        by_doi = await search(experiment="10.1/", temperature_min=300)
        assert by_doi["total"] == 1
        assert by_doi["items"][0]["filename"] == "iso-2"

        partial = await search(adsorbent="zif")
        assert [item["filename"] for item in partial["items"]] == ["ISO-1"]

        with pytest.raises(ValueError):
            await search()

    asyncio.run(scenario())

###############################################################################
def test_targeted_nist_request_replaces_fraction_with_identifiers() -> None:
    request = PublicDataSyncCreateRequest.model_validate(
        {
            "source_key": "nist",
            "operation": "nist_category",
            "category": "experiments",
            "fraction": 0.5,
            "identifiers": ["iso-1", "iso-2"],
        }
    )
    assert request.parameters() == {
        "category": "experiments",
        "identifiers": ["iso-1", "iso-2"],
    }
    service = NISTDataService(None, None, None)  # type: ignore[arg-type]
    index = __import__("pandas").DataFrame({"filename": ["ISO-1", "iso-2", "iso-3"]})
    assert service.select_identifiers(index, "filename", 1.0, ["iso-1", "gone"]) == (
        ["ISO-1"],
        3,
        ["gone"],
    )


###############################################################################
def _scripted_nist_service(
    ids: list[str], stored: set[str], failing: set[str]
) -> tuple[NISTDataService, list[list[str]]]:
    """NIST service whose index, storage and HTTP outcomes are scripted."""
    pd = __import__("pandas")
    service = NISTDataService(None, None, None)  # type: ignore[arg-type]
    api_client = type(
        "Api", (), {"request_statistics": lambda self: {}}
    )()
    service._api_client = lambda: api_client  # type: ignore[method-assign,return-value]
    service._request_timeout = lambda **_: None  # type: ignore[method-assign,return-value]
    fetched: list[list[str]] = []

    async def load_index(category: str, client: Any, api_client: Any = None) -> Any:
        return pd.DataFrame({"filename": ids}), "filename"

    async def details(category: str) -> Any:
        return set(stored), "https://example.invalid/{}"

    async def fetch(*, batch: list[str], on_failure: Any, **_: Any) -> Any:
        fetched.append(list(batch))
        bad = [item for item in batch if item in failing]
        for item in bad:
            await on_failure(item, "failed")
        good = [(item, {}) for item in batch if item not in failing]
        return good, bad

    async def persist(*, successful: list[Any], **_: Any) -> Any:
        stored.update(item.lower() for item, _payload in successful)
        return len(successful), 0

    service.load_index_for_category = load_index  # type: ignore[method-assign]
    service._category_sync_details = details  # type: ignore[method-assign]
    service._fetch_sync_batch = fetch  # type: ignore[method-assign]
    service._persist_sync_batch = persist  # type: ignore[method-assign]
    service.repository = type(
        "Repo", (), {"count_local_records_by_category": lambda self: {}}
    )()
    return service, fetched


async def _run_scripted_sync(
    service: NISTDataService, events: list[dict[str, Any]], **overrides: Any
) -> dict[str, Any]:
    async def progress(payload: dict[str, Any]) -> None:
        events.append(payload)

    async def never() -> bool:
        return False

    async def ignore(*_args: Any) -> None:
        return None

    arguments: dict[str, Any] = dict(
        category="experiments",
        fraction=1.0,
        batch_size=2,
        max_records=None,
        on_progress=progress,
        should_cancel=never,
        on_failure=ignore,
        on_skip=ignore,
    )
    return await service.sync_category_records(**(arguments | overrides))


def test_nist_checkpoint_never_advances_past_a_failed_record() -> None:
    ids = ["a", "b", "c", "d"]
    stored: set[str] = set()
    service, fetched = _scripted_nist_service(ids, stored, failing={"b"})
    events: list[dict[str, Any]] = []
    result = asyncio.run(_run_scripted_sync(service, events))
    assert result["failed_count"] == 1
    assert events[-1]["last_external_id"] == "a"

    # Resuming from that checkpoint replays the failed record only.
    assert stored == {"a", "c", "d"}
    service, fetched = _scripted_nist_service(ids, stored, failing=set())
    asyncio.run(_run_scripted_sync(service, [], resume_after_external_id="a"))
    assert fetched == [["b"]]
    assert stored == {"a", "b", "c", "d"}


def test_nist_fractional_requests_progress_through_missing_records() -> None:
    ids = [f"id{index}" for index in range(10)]
    stored = {"id0", "id1"}
    service, fetched = _scripted_nist_service(ids, stored, failing=set())
    result = asyncio.run(_run_scripted_sync(service, [], fraction=0.2, batch_size=10))
    assert fetched == [["id2", "id3"]]
    assert result["imported_count"] == 2

    fetched.clear()
    asyncio.run(_run_scripted_sync(service, [], fraction=0.2, batch_size=10))
    assert fetched == [["id4", "id5"]]


def test_nist_targeted_unknown_identifiers_are_reported_as_failures() -> None:
    service, fetched = _scripted_nist_service(["a", "b"], set(), failing=set())
    reported: list[str] = []

    async def on_failure(identifier: str, _reason: str) -> None:
        reported.append(identifier)

    result = asyncio.run(
        _run_scripted_sync(service, [], identifiers=["a", "ghost"], on_failure=on_failure)
    )
    assert reported == ["ghost"]
    assert result["failed_count"] == 1
    assert fetched == [["a"]]


def test_nist_builder_keeps_rows_missing_optional_metadata_and_invents_nothing() -> None:
    pd = __import__("pandas")
    from server.services.data.nistads import NISTDatasetBuilder

    def isotherm(name: str, guests: list[dict[str, Any]], points: list[dict[str, Any]]):
        return {
            "filename": name,
            "adsorbent": {"hashkey": "H", "name": "Zeolite"},
            "adsorbates": guests,
            "isotherm_data": points,
            "temperature": 298,
            "pressureUnits": "bar",
            "adsorptionUnits": "mmol/g",
            "adsorbent_extra": None,
        }

    single = isotherm(
        "S1",
        [{"name": "CO2", "InChIKey": "K1"}],
        [
            {"pressure": 1.0, "total_adsorption": 2.0},
            {"pressure": 2.0, "total_adsorption": 3.0},
        ],
    )
    mixture = isotherm(
        "M1",
        [{"name": "CO2", "InChIKey": "K1"}, {"name": "N2", "InChIKey": "K2"}],
        [
            {
                "pressure": 2.0,
                "species_data": [
                    {"composition": 0.5, "adsorption": 1.0},
                    {"composition": 0.5, "adsorption": 2.0},
                ],
            },
            {"pressure": 4.0, "species_data": [{"composition": 1.0}]},
        ],
    )
    builder = NISTDatasetBuilder()
    singles, _ = builder.build_datasets(pd.DataFrame([single]))
    assert len(singles) == 2
    _, binaries = builder.build_datasets(pd.DataFrame([mixture]))
    # The point without second-species data is rejected, not filled with 1.0.
    assert len(binaries) == 1
    assert binaries.loc[0, "compound_2_adsorption"] == 2.0

###############################################################################
def test_cod_bulk_imports_only_exact_matches_and_remembers_the_rest(
    tmp_path: Path,
) -> None:
    from server.repositories.schemas.models import Adsorbent
    from server.services.providers.cod import CODProvider

    async def scenario() -> None:
        manager = _database(tmp_path)
        repository = PublicDataRepository(manager)
        repository.ensure_sources(PUBLIC_SOURCE_DEFINITIONS)
        with manager.transaction() as session:
            for name in (
                "Alpha-MOF",
                "Beta-MOF",
                "Gamma-MOF",
                "Delta-MOF",
                "Epsilon-MOF (synthetic)",
            ):
                session.add(Adsorbent(key=name.lower(), name=name))
        provider = CODProvider(
            request_timeout_seconds=5.0, retry_attempts=1, max_interactive_results=250
        )
        catalogue = {
            "Alpha-MOF": [
                {"cod_id": "1000001", "raw_metadata": {"chemname": "alpha-mof"}}
            ],
            "Beta-MOF": [
                {
                    "cod_id": "1000002",
                    "has_coordinates": False,
                    "raw_metadata": {"mineral": "Beta-MOF"},
                },
                {
                    "cod_id": "1000006",
                    "has_coordinates": True,
                    "raw_metadata": {"chemname": "Beta-MOF"},
                },
                {
                    "cod_id": "1000003",
                    "has_coordinates": True,
                    "raw_metadata": {"chemname": "Beta-MOF"},
                },
            ],
            "Gamma-MOF": [],
            "Delta-MOF": [
                {"cod_id": "1000004", "raw_metadata": {"title": "Delta-MOF framework"}}
            ],
            "Epsilon-MOF": [
                {"cod_id": "1000005", "raw_metadata": {"commonname": "Epsilon-MOF"}}
            ],
        }

        async def search(**kwargs: Any) -> list[dict[str, Any]]:
            return catalogue[kwargs["text"]]

        provider.search = search  # type: ignore[method-assign]
        public_data = PublicDataService(repository=repository, providers=[provider])
        linked: list[tuple[str, int | None]] = []

        async def import_cod(
            cod_id: str,
            adsorbent_id: int | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> Any:
            # The bulk run passes the search row so the import skips a lookup.
            assert metadata is not None and str(metadata["cod_id"]) == cod_id
            linked.append((cod_id, adsorbent_id))
            return len(linked), True

        public_data.import_cod = import_cod  # type: ignore[method-assign]
        coordinator = PublicDataSyncCoordinator(
            repository=repository,
            public_data=public_data,
            nist=_Nist(),  # type: ignore[arg-type]
            config=PublicDataConfig(),
        )
        await coordinator.start()
        try:
            queued = await coordinator.enqueue(
                source_key="cod",
                operation="cod_import",
                parameters={"fraction": 1.0},
            )
            run = await _wait_for_terminal(coordinator, queued["id"])
            failures, _ = await coordinator.failures(run["id"], limit=10, offset=0)

            assert run["status"] == "completed"
            assert (run["processed_count"], run["imported_count"]) == (5, 3)
            assert run["skipped_count"] == 2
            assert linked == [("1000001", 1), ("1000003", 2), ("1000005", 5)]
            assert any("Ambiguous" in item["reason"] for item in failures)
            assert any("No COD structure" in item["reason"] for item in failures)
            # Settled outcomes are not retried; only the unlinked matches remain.
            remaining = repository.cod_bulk_candidates(10)
            assert [item["name"] for item in remaining] == [
                "Alpha-MOF",
                "Beta-MOF",
                "Epsilon-MOF (synthetic)",
            ]
        finally:
            await coordinator.close()
            await public_data.close()
            manager.dispose()

    asyncio.run(scenario())

###############################################################################
def test_conformer_download_is_cached_and_bulk_run_skips_cached_species(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    from server.repositories.materials import MaterialRepository

    async def scenario() -> None:
        manager = _database(tmp_path)
        repository = PublicDataRepository(manager)
        repository.ensure_sources(PUBLIC_SOURCE_DEFINITIONS)
        adsorbate_id = repository.upsert_pubchem_compound(
            {
                "cid": "297",
                "name": "Methane",
                "formula": "CH4",
                "inchi_key": "VNWKTOKETHGBQD-UHFFFAOYSA-N",
                "conformer_3d_url": "https://pubchem.ncbi.nlm.nih.gov/conformer",
                "raw_metadata": {},
            }
        )
        provider = PubChemProvider(parallel_requests=1)
        downloads: list[str] = []

        async def fake_conformer(cid: str) -> str:
            downloads.append(cid)
            return "methane\n  3D\n\nM  END"

        monkeypatch.setattr(provider, "fetch_conformer", fake_conformer)
        public_data = PublicDataService(repository=repository, providers=[provider])
        coordinator = PublicDataSyncCoordinator(
            repository=repository,
            public_data=public_data,
            nist=_Nist(),
            config=PublicDataConfig(),
        )
        materials = MaterialRepository(manager)
        assert materials.get_species(adsorbate_id)["conformer_3d_cached"] is False
        await coordinator.start()
        try:
            queued = await coordinator.enqueue(
                source_key="pubchem",
                operation="pubchem_enrich",
                parameters={"fraction": 1.0, "conformers": True},
            )
            run = await _wait_for_terminal(coordinator, queued["id"])
            assert run["status"] == "completed"
            assert run["imported_count"] == 1
            assert downloads == ["297"]
            assert materials.get_species(adsorbate_id)["conformer_3d_cached"] is True
            assert repository.conformer_candidates(10) == []
            sdf, downloaded = await public_data.ensure_conformer(adsorbate_id)
            assert (sdf, downloaded) == ("methane\n  3D\n\nM  END", False)
            assert downloads == ["297"]
        finally:
            await coordinator.close()
            await public_data.close()
            manager.dispose()

    asyncio.run(scenario())

###############################################################################
def test_conformers_flag_applies_only_to_pubchem_enrichment() -> None:
    with pytest.raises(ValueError):
        PublicDataSyncCreateRequest(
            source_key="cod", operation="cod_import", fraction=0.5, conformers=True
        )
    request = PublicDataSyncCreateRequest(
        source_key="pubchem", operation="pubchem_enrich", fraction=0.5, conformers=True
    )
    assert request.parameters() == {"fraction": 0.5, "conformers": True}


###############################################################################
def test_fraction_count_applies_cap_only_when_given() -> None:
    assert PublicDataRepository._fraction_count(9328, 0.25, None) == 2332
    assert PublicDataRepository._fraction_count(9328, 0.25, 200) == 200
