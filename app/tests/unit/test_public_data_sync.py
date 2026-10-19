# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from server.configurations.settings import DatabaseConfig, PublicDataConfig, load_config
from server.domain.public_data import PublicDataSyncCreateRequest
from server.repositories.database.manager import DatabaseManager
from server.repositories.public_data import PublicDataRepository
from server.repositories.schemas import Base
from server.services.data.public_data import PublicDataService
from server.services.data.public_data_sync import PublicDataSyncCoordinator
from server.services.data.nist_service import NISTDataService
from server.services.data.nistads import NISTApiClient
from server.services.providers import pubchem as pubchem_module
from server.services.providers.pubchem import PubChemProvider, PubChemRequestBudget
from server.services.providers.public_data import (
    ProviderCapability,
    ProviderUnavailableError,
    ProviderHealth,
    PublicDataProvider,
)


def _database(path: Path) -> DatabaseManager:
    manager = DatabaseManager(
        DatabaseConfig(
            embedded_database=True,
            connect_timeout=5,
            insert_batch_size=100,
            sqlite_path=str(path / "public-data-sync.db"),
        )
    )
    Base.metadata.create_all(manager.engine)
    return manager


class _Provider(PublicDataProvider):
    key = "nist"
    name = "NIST test provider"
    description = "Deterministic provider for coordinator tests."
    homepage_url = "https://adsorption.nist.gov/"
    license_name = None
    license_url = None
    terms_url = None
    capabilities = (ProviderCapability.ADSORPTION,)

    async def health(self) -> ProviderHealth:
        return ProviderHealth(
            status="available", detail=None, checked_at=datetime.now(timezone.utc)
        )


class _Nist:
    async def sync_category_records(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        raise NotImplementedError


async def _wait_for_terminal(
    coordinator: PublicDataSyncCoordinator, run_id: str
) -> dict[str, Any]:
    for _ in range(300):
        run = await coordinator.get(run_id)
        if run["status"] not in {"queued", "running"}:
            return run
        await asyncio.sleep(0.01)
    raise AssertionError(f"Sync run {run_id} did not reach a terminal state.")


def test_coordinator_persists_checkpoints_partial_failures_and_skip_reasons(
    tmp_path: Path,
) -> None:
    class PartialNist:
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
        repository.ensure_sources()
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


def test_coordinator_cancels_between_nist_batches(tmp_path: Path) -> None:
    async def scenario() -> None:
        started = asyncio.Event()

        class BlockingNist:
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
        repository.ensure_sources()
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


def test_coordinator_reuses_local_pubchem_identity_without_a_provider_call(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    async def scenario() -> None:
        manager = _database(tmp_path)
        repository = PublicDataRepository(manager)
        repository.ensure_sources()
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


def test_cod_batch_import_is_bounded_and_keeps_record_failures(tmp_path: Path) -> None:
    async def scenario() -> None:
        manager = _database(tmp_path)
        repository = PublicDataRepository(manager)
        repository.ensure_sources()
        public_data = PublicDataService(repository=repository, providers=[_Provider()])
        imported: list[str] = []

        async def import_cod(request: Any) -> Any:
            if request.cod_id == "9012346":
                raise ProviderUnavailableError("COD CIF request timed out.")
            imported.append(request.cod_id)
            return type("Structure", (), {"id": len(imported)})()

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
                parameters=request.model_dump(
                    exclude={"source_key", "operation"}, exclude_none=True
                ),
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


def test_concurrent_sync_run_creation_serializes_sqlite_writes(tmp_path: Path) -> None:
    manager = _database(tmp_path)
    try:
        repository = PublicDataRepository(manager)
        repository.ensure_sources()

        async def create_runs() -> list[dict[str, Any]]:
            return await asyncio.gather(
                *(
                    asyncio.to_thread(
                        repository.create_sync_run,
                        source_key="nist",
                        operation="health_check",
                        parameters={},
                    )
                    for _ in range(12)
                )
            )

        runs = asyncio.run(create_runs())

        assert len(runs) == 12
        assert len({run["id"] for run in runs}) == 12
        assert all(run["status"] == "queued" for run in runs)
    finally:
        manager.dispose()


def test_nist_restart_resumes_after_checkpoint_without_recounting_prior_batches(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manager = _database(tmp_path)
        repository = PublicDataRepository(manager)
        repository.ensure_sources()
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

        class ResumingNist:
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
                "max_records": 200,
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


def test_cod_restart_continues_after_checkpoint_without_reimporting_saved_ids(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manager = _database(tmp_path)
        repository = PublicDataRepository(manager)
        repository.ensure_sources()
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

        async def import_cod(request: Any) -> Any:
            imported.append(request.cod_id)
            structure_id = 100 + len(imported)
            local_ids[request.cod_id] = structure_id
            return type("Structure", (), {"id": structure_id})()

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


def test_restart_policy_preserves_runs_and_exposes_resume_choice(tmp_path: Path) -> None:
    manager = _database(tmp_path)
    try:
        repository = PublicDataRepository(manager)
        repository.ensure_sources()
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


def test_pubchem_request_budget_is_shared_across_provider_instances(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    clock = [0.0]
    delays: list[float] = []
    budget = PubChemRequestBudget(minimum_interval_seconds=0.22)

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)
        clock[0] += delay

    monkeypatch.setattr(pubchem_module, "monotonic", lambda: clock[0])
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    first = PubChemProvider(parallel_requests=1, request_budget=budget)
    second = PubChemProvider(parallel_requests=1, request_budget=budget)

    async def scenario() -> None:
        await first._before_attempt()
        await second._before_attempt()

    asyncio.run(scenario())
    assert delays == [0.22]


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


def test_nist_streaming_client_retries_with_retry_after_and_records_statistics(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    url = "https://adsorption.nist.gov/isodb/api/gases/test.json"
    responses = [
        httpx.Response(429, headers={"Retry-After": "1.5"}),
        httpx.Response(200, json={"name": "methane"}),
    ]
    delays: list[float] = []

    class _Stream:
        def __init__(self, response: Any) -> None:
            self.response = response

        async def __aenter__(self) -> Any:
            return self.response

        async def __aexit__(self, *_args: Any) -> None:
            return None

    class _Client:
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


def test_nist_streaming_client_rejects_oversized_response() -> None:
    class _Stream:
        async def __aenter__(self) -> httpx.Response:
            return httpx.Response(
                200,
                text="x" * 2048,
                request=httpx.Request(
                    "GET", "https://adsorption.nist.gov/isodb/api/gases/test.json"
                ),
            )

        async def __aexit__(self, *_args: Any) -> None:
            return None

    class _Client:
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


def test_nist_health_probe_requests_only_a_single_byte_range(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[tuple[str, str, dict[str, str]]] = []

    class _Stream:
        async def __aenter__(self) -> httpx.Response:
            return httpx.Response(
                206,
                headers={"Content-Range": "bytes 0-0/2000000"},
                request=httpx.Request(
                    "GET", "https://adsorption.nist.gov/isodb/api/isotherms.json"
                ),
            )

        async def __aexit__(self, *_args: Any) -> None:
            return None

    class _Client:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def stream(
            self, method: str, url: str, *, headers: dict[str, str]
        ) -> _Stream:
            calls.append((method, url, headers))
            return _Stream()

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    service = NISTDataService(
        config=load_config(Path("data/adsmod.json")),
        job_manager=object(),  # type: ignore[arg-type]
        repository=object(),  # type: ignore[arg-type]
        mapper=object(),  # type: ignore[arg-type]
    )
    result = asyncio.run(service.ping_server("experiments"))

    assert result["server_ok"] is True
    assert calls == [
        (
            "GET",
            "https://adsorption.nist.gov/isodb/api/isotherms.json",
            {"Range": "bytes=0-0"},
        )
    ]
