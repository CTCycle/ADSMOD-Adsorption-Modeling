# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
import math
import threading
import uuid
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Literal
from urllib.parse import quote

import httpx
import pandas as pd

from server.configurations.settings import AdsmodConfig
from server.common.utils.logger import logger
from server.domain.nist import (
    NISTCategory,
    NISTCategoryFetchRequest,
    NISTFetchRequest,
    NISTPropertiesRequest,
)
from server.services.data.nistads import (
    NISTApiClient,
    NISTDatasetBuilder,
)
from server.services.data.nist_mapper import NISTCanonicalMapper
from server.repositories.nist import NISTRepository
from server.services.data.pubchem_client import PubChemClient
from server.services.providers.pubchem import PubChemRequestBudget
from server.domain.jobs import (
    JobCancelResponse,
    JobListResponse,
    JobStartResponse,
    JobStatusResponse,
)
from server.services.job_responses import JobResponseFactory
from server.services.jobs import JobManager

###############################################################################
class NISTDataService:
    JOB_TYPE_FETCH = "nist_fetch"
    JOB_TYPE_PROPERTIES = "nist_properties"
    CATEGORY_INDEX_SUFFIX = "index"
    CATEGORY_FETCH_SUFFIX = "fetch"
    CATEGORY_ENRICH_SUFFIX = "enrich"

    # -------------------------------------------------------------------------
    def __init__(
        self,
        config: AdsmodConfig,
        job_manager: JobManager,
        repository: NISTRepository,
        mapper: NISTCanonicalMapper,
        pubchem_request_budget: PubChemRequestBudget | None = None,
    ) -> None:
        self.config = config
        self.job_manager = job_manager
        self.repository = repository
        self.mapper = mapper
        self.pubchem_request_budget = pubchem_request_budget or PubChemRequestBudget()
        self.builder = NISTDatasetBuilder()
        self.state_lock = threading.Lock()
        self.category_state: dict[str, dict[str, Any]] = {
            "experiments": {
                "available_count": 0,
                "last_update": None,
                "server_ok": None,
                "server_checked_at": None,
            },
            "guest": {
                "available_count": 0,
                "last_update": None,
                "server_ok": None,
                "server_checked_at": None,
            },
            "host": {
                "available_count": 0,
                "last_update": None,
                "server_ok": None,
                "server_checked_at": None,
            },
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    # -------------------------------------------------------------------------
    def _api_client(self) -> NISTApiClient:
        public_data = self.config.application.public_data
        return NISTApiClient(
            self.config.application.nist.parallel_tasks,
            retry_attempts=public_data.retry_attempts,
            retry_backoff_max_seconds=public_data.retry_backoff_max_seconds,
            max_response_bytes=public_data.sync_response_size_limit_bytes,
        )

    # -------------------------------------------------------------------------
    def _request_timeout(self, *, health: bool = False) -> httpx.Timeout:
        seconds = self.config.application.public_data.request_timeout_seconds
        return httpx.Timeout(min(10.0, seconds) if health else seconds)

    # -------------------------------------------------------------------------
    @staticmethod
    def category_supports_enrichment(category: str) -> bool:
        return category in ("guest", "host")

    # -------------------------------------------------------------------------
    @staticmethod
    def clamp_fraction(fraction: float) -> float:
        return min(1.0, max(0.001, float(fraction)))

    # -------------------------------------------------------------------------
    def update_category_state(self, category: str, **patch: Any) -> None:
        with self.state_lock:
            state = self.category_state.setdefault(category, {})
            state.update(patch)

    # -------------------------------------------------------------------------
    def get_category_state(self, category: str) -> dict[str, Any]:
        with self.state_lock:
            return dict(self.category_state.get(category, {}))

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_identifier(value: object) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return ""
        return text.lower()

    # -------------------------------------------------------------------------
    @staticmethod
    def build_category_job_type(category: NISTCategory, suffix: str) -> str:
        return f"nist_{category}_{suffix}"

    # -------------------------------------------------------------------------
    @staticmethod
    def run_awaitable_sync(awaitable: Any) -> dict[str, Any]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(awaitable)
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    # -------------------------------------------------------------------------
    def start_background_job(
        self,
        job_type: str,
        runner: Any,
        args: tuple[Any, ...] = (),
        message: str = "Job started.",
    ) -> JobStartResponse:
        if self.job_manager.is_job_running(job_type):
            raise ValueError(f"A {job_type} job is already running.")

        job_id = str(uuid.uuid4())[:8]
        self.job_manager.start_job(
            job_type=job_type,
            runner=runner,
            args=args,
            job_id=job_id,
        )
        return JobResponseFactory.start(
            job_id=job_id,
            job_type=job_type,
            message=message,
            poll_interval=self.config.application.jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def _run_fetch_sync(
        self,
        experiments_fraction: float,
        guest_fraction: float,
        host_fraction: float,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        return self.run_awaitable_sync(
            self.fetch_and_store(
                experiments_fraction=experiments_fraction,
                guest_fraction=guest_fraction,
                host_fraction=host_fraction,
                job_id=job_id,
            )
        )

    # -------------------------------------------------------------------------
    def _run_properties_sync(
        self, target: str, job_id: str | None = None
    ) -> dict[str, Any]:
        return self.run_awaitable_sync(
            self.enrich_properties(target=target, job_id=job_id)
        )

    # -------------------------------------------------------------------------
    def _run_category_index_sync(
        self, category: NISTCategory, job_id: str | None = None
    ) -> dict[str, Any]:
        if category == "experiments":
            return self.run_awaitable_sync(self.fetch_experiments_index(job_id=job_id))
        if category == "guest":
            return self.run_awaitable_sync(self.fetch_guest_index(job_id=job_id))
        return self.run_awaitable_sync(self.fetch_host_index(job_id=job_id))

    # -------------------------------------------------------------------------
    def _run_category_fetch_sync(
        self, category: NISTCategory, fraction: float, job_id: str | None = None
    ) -> dict[str, Any]:
        if category == "experiments":
            return self.run_awaitable_sync(
                self.fetch_experiments_records(fraction=fraction, job_id=job_id)
            )
        if category == "guest":
            return self.run_awaitable_sync(
                self.fetch_guest_records(fraction=fraction, job_id=job_id)
            )
        return self.run_awaitable_sync(
            self.fetch_host_records(fraction=fraction, job_id=job_id)
        )

    # -------------------------------------------------------------------------
    def _run_category_enrich_sync(
        self, category: NISTCategory, job_id: str | None = None
    ) -> dict[str, Any]:
        if category == "guest":
            return self.run_awaitable_sync(self.enrich_guest_properties(job_id=job_id))
        if category == "host":
            return self.run_awaitable_sync(self.enrich_host_properties(job_id=job_id))
        raise ValueError("Enrichment is not supported for experiments.")

    # -------------------------------------------------------------------------
    def start_fetch_job(self, request: NISTFetchRequest) -> JobStartResponse:
        return self.start_background_job(
            job_type=self.JOB_TYPE_FETCH,
            runner=self._run_fetch_sync,
            args=(
                request.experiments_fraction,
                request.guest_fraction,
                request.host_fraction,
            ),
            message="NIST fetch job started.",
        )

    # -------------------------------------------------------------------------
    def start_properties_job(self, request: NISTPropertiesRequest) -> JobStartResponse:
        return self.start_background_job(
            job_type=self.JOB_TYPE_PROPERTIES,
            runner=self._run_properties_sync,
            args=(request.target,),
            message=f"NIST properties enrichment job started for {request.target}.",
        )

    # -------------------------------------------------------------------------
    def start_category_index_job(self, category: NISTCategory) -> JobStartResponse:
        return self.start_background_job(
            job_type=self.build_category_job_type(category, self.CATEGORY_INDEX_SUFFIX),
            runner=self._run_category_index_sync,
            args=(category,),
            message=f"NIST {category} index job started.",
        )

    # -------------------------------------------------------------------------
    def start_category_fetch_job(
        self, category: NISTCategory, request: NISTCategoryFetchRequest
    ) -> JobStartResponse:
        return self.start_background_job(
            job_type=self.build_category_job_type(category, self.CATEGORY_FETCH_SUFFIX),
            runner=self._run_category_fetch_sync,
            args=(category, request.fraction),
            message=f"NIST {category} fetch job started.",
        )

    # -------------------------------------------------------------------------
    def start_category_enrich_job(self, category: NISTCategory) -> JobStartResponse:
        if category == "experiments":
            raise ValueError("Enrichment is not supported for experiments.")

        return self.start_background_job(
            job_type=self.build_category_job_type(
                category, self.CATEGORY_ENRICH_SUFFIX
            ),
            runner=self._run_category_enrich_sync,
            args=(category,),
            message=f"NIST {category} enrichment job started.",
        )

    # -------------------------------------------------------------------------
    def get_job_status(self, job_id: str) -> JobStatusResponse:
        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise LookupError(f"Job {job_id} not found.")
        return JobResponseFactory.status(
            job_status=job_status,
            poll_interval=self.config.application.jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def list_jobs(self) -> JobListResponse:
        all_jobs = [
            job
            for job in self.job_manager.list_jobs()
            if str(job.get("job_type", "")).startswith("nist_")
        ]
        return JobResponseFactory.list(
            job_statuses=all_jobs,
            poll_interval=self.config.application.jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def cancel_job(self, job_id: str) -> JobCancelResponse:
        success = self.job_manager.cancel_job(job_id)
        if not success:
            raise ValueError(
                f"Job {job_id} cannot be cancelled (not found or already completed)."
            )
        return JobResponseFactory.cancelled(job_id)

    # -------------------------------------------------------------------------
    def ordered_identifiers(
        self, index: pd.DataFrame, identifier_column: str, fraction: float
    ) -> tuple[list[str], int]:
        if identifier_column not in index.columns:
            raise ValueError(f"NIST index missing {identifier_column} column.")

        ordered: list[str] = []
        seen: set[str] = set()
        for raw_value in index[identifier_column].tolist():
            text = str(raw_value).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(text)

        total = len(ordered)
        if total == 0:
            return [], 0

        requested = int(math.ceil(self.clamp_fraction(fraction) * total))
        requested = min(total, max(0, requested))
        return ordered[:requested], total

    # -------------------------------------------------------------------------
    async def load_index_for_category(
        self,
        category: Literal["experiments", "guest", "host"],
        client: httpx.AsyncClient,
    ) -> tuple[pd.DataFrame, str]:
        api_client = self._api_client()
        if category == "experiments":
            return await api_client.fetch_experiments_index(
                client
            ), api_client.exp_identifier
        if category == "guest":
            return await api_client.fetch_guest_index(
                client
            ), api_client.guest_identifier
        return await api_client.fetch_host_index(client), api_client.host_identifier

    # -------------------------------------------------------------------------
    async def fetch_and_store(
        self,
        experiments_fraction: float,
        guest_fraction: float,
        host_fraction: float,
        job_id: str | None = None,
    ) -> dict[str, int]:
        logger.info(
            "NIST fetch starting (experiments_fraction=%s, guest_fraction=%s, host_fraction=%s)",
            experiments_fraction,
            guest_fraction,
            host_fraction,
        )
        if job_id:
            self.job_manager.update_progress(job_id, 0.0)

        experiments_result = await self.fetch_experiments_records(
            experiments_fraction, job_id=None
        )
        if job_id:
            self.job_manager.update_progress(job_id, 35.0)

        guest_result = await self.fetch_guest_records(guest_fraction, job_id=None)
        if job_id:
            self.job_manager.update_progress(job_id, 70.0)

        host_result = await self.fetch_host_records(host_fraction, job_id=None)
        if job_id:
            self.job_manager.update_progress(job_id, 100.0)

        return {
            "experiments_count": int(experiments_result.get("fetched_count", 0)),
            "single_component_rows": int(
                experiments_result.get("single_component_rows", 0)
            ),
            "binary_mixture_rows": int(
                experiments_result.get("binary_mixture_rows", 0)
            ),
            "skipped_experiment_count": int(
                experiments_result.get("skipped_count", 0)
            ),
            "guest_rows": int(guest_result.get("fetched_count", 0)),
            "host_rows": int(host_result.get("fetched_count", 0)),
        }

    # -------------------------------------------------------------------------
    async def ping_server(
        self, category: Literal["experiments", "guest", "host"]
    ) -> dict[str, Any]:
        api_client = self._api_client()
        url_by_category = {
            "experiments": api_client.url_isotherms,
            "guest": api_client.url_guest_index,
            "host": api_client.url_host_index,
        }
        url = url_by_category[category]

        server_ok = False
        rate_limited = False
        timeout = self._request_timeout(health=True)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                # Stream headers only. A normal GET can download a complete,
                # large JSON index merely to establish reachability.
                async with client.stream(
                    "GET", url, headers={"Range": "bytes=0-0"}
                ) as response:
                    rate_limited = response.status_code == 429
                    response.raise_for_status()
                    server_ok = response.status_code in (200, 206)
            except httpx.HTTPError:
                server_ok = False

        checked_at = self.now_iso()
        self.update_category_state(
            category,
            server_ok=server_ok,
            server_checked_at=checked_at,
        )
        return {
            "category": category,
            "server_ok": server_ok,
            "checked_at": checked_at,
            "request_stats": {
                "requests": 1,
                "retries": 0,
                "rate_limits": int(rate_limited),
                "failures": int(not server_ok),
            },
        }

    # -------------------------------------------------------------------------
    async def fetch_category_index(
        self,
        category: Literal["experiments", "guest", "host"],
        job_id: str | None = None,
    ) -> dict[str, Any]:
        if job_id:
            self.job_manager.update_progress(job_id, 10.0)

        timeout = self._request_timeout()
        async with httpx.AsyncClient(timeout=timeout) as client:
            index, _ = await self.load_index_for_category(category, client)

        available_count = int(len(index))
        if job_id:
            self.job_manager.update_progress(job_id, 100.0)

        self.update_category_state(
            category,
            available_count=available_count,
            last_update=self.now_iso(),
        )
        return {
            "category": category,
            "available_count": available_count,
        }

    # -------------------------------------------------------------------------
    async def fetch_category_records(
        self,
        category: Literal["experiments", "guest", "host"],
        fraction: float,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        normalized_fraction = self.clamp_fraction(fraction)
        start_time = monotonic()
        api_client = self._api_client()

        if job_id:
            self.job_manager.update_progress(job_id, 5.0)

        timeout = self._request_timeout()
        async with httpx.AsyncClient(timeout=timeout) as client:
            index, identifier_column = await self.load_index_for_category(
                category, client
            )
            requested_identifiers, available_count = self.ordered_identifiers(
                index=index,
                identifier_column=identifier_column,
                fraction=normalized_fraction,
            )

            if category == "experiments":
                existing_ids = await asyncio.to_thread(
                    self.repository.list_nist_experiment_ids
                )
            elif category == "guest":
                existing_ids = await asyncio.to_thread(
                    self.repository.list_adsorbate_inchi_keys
                )
            else:
                existing_ids = await asyncio.to_thread(
                    self.repository.list_adsorbent_hash_keys
                )

            if job_id:
                self.job_manager.update_progress(job_id, 25.0)

            identifiers_to_fetch = [
                identifier
                for identifier in requested_identifiers
                if self.normalize_identifier(identifier) not in existing_ids
            ]

            fetched_count = 0
            single_component_rows = 0
            binary_mixture_rows = 0
            skipped_count = 0

            if category == "experiments":
                experiments_data = (
                    await api_client.fetch_experiments_data_by_identifiers(
                        client, identifiers_to_fetch
                    )
                )
                fetched_count = int(len(experiments_data))
                if job_id:
                    self.job_manager.update_progress(job_id, 65.0)

                single_component, binary_mixture = self.builder.build_datasets(
                    experiments_data
                )
                single_component_rows = int(len(single_component))
                binary_mixture_rows = int(len(binary_mixture))
                if job_id:
                    self.job_manager.update_progress(job_id, 80.0)

                experiments = self.mapper.experiment_records(
                    single_component, binary_mixture
                )
                skipped_count = max(0, fetched_count - len(experiments))
                if skipped_count:
                    logger.warning(
                        "NIST experiment mapping skipped %d of %d fetched records",
                        skipped_count,
                        fetched_count,
                    )
                await asyncio.to_thread(
                    self.repository.save_experiments,
                    experiments,
                    False,
                )
            elif category == "guest":
                guest_data = await api_client.fetch_material_dataset_by_identifiers(
                    client=client,
                    identifiers=identifiers_to_fetch,
                    identifier_column=api_client.guest_identifier,
                    url_template="https://adsorption.nist.gov/isodb/api/gas/{identifier}.json",
                    extra_columns=api_client.extra_guest_columns,
                    drop_columns=["synonyms"],
                )
                fetched_count = int(len(guest_data))
                if job_id:
                    self.job_manager.update_progress(job_id, 70.0)
                guest_records = self.mapper.material_records(guest_data, "adsorbate")
                await asyncio.to_thread(
                    self.repository.save_materials,
                    guest_records,
                    [],
                )
            else:
                host_data = await api_client.fetch_material_dataset_by_identifiers(
                    client=client,
                    identifiers=identifiers_to_fetch,
                    identifier_column=api_client.host_identifier,
                    url_template="https://adsorption.nist.gov/matdb/api/material/{identifier}.json",
                    extra_columns=api_client.extra_host_columns,
                    drop_columns=["External_Resources", "synonyms"],
                )
                fetched_count = int(len(host_data))
                if job_id:
                    self.job_manager.update_progress(job_id, 70.0)
                host_records = self.mapper.material_records(host_data, "adsorbent")
                await asyncio.to_thread(
                    self.repository.save_materials,
                    [],
                    host_records,
                )

        local_counts = await asyncio.to_thread(
            self.repository.count_local_records_by_category
        )
        local_count = int(local_counts.get(category, 0))

        self.update_category_state(
            category,
            available_count=int(available_count),
            last_update=self.now_iso(),
        )
        if job_id:
            self.job_manager.update_progress(job_id, 100.0)

        elapsed = monotonic() - start_time
        logger.info(
            "NIST category fetch completed (category=%s, available=%d, requested=%d, fetched=%d, local=%d, elapsed_s=%.2f)",
            category,
            int(available_count),
            len(requested_identifiers),
            fetched_count,
            local_count,
            elapsed,
        )

        return {
            "category": category,
            "available_count": int(available_count),
            "requested_count": int(len(requested_identifiers)),
            "fetched_count": fetched_count,
            "local_count": local_count,
            "skipped_count": skipped_count,
            "single_component_rows": single_component_rows,
            "binary_mixture_rows": binary_mixture_rows,
        }

    # -------------------------------------------------------------------------
    async def sync_category_records(
        self,
        *,
        category: Literal["experiments", "guest", "host"],
        fraction: float,
        batch_size: int,
        max_records: int,
        on_progress: Any,
        should_cancel: Any,
        on_failure: Any,
        on_skip: Any,
        processed_before: int = 0,
        resume_after_external_id: str | None = None,
    ) -> dict[str, Any]:
        """Fetch one NIST collection in idempotent, checkpointed batches."""
        api_client = self._api_client()
        timeout = self._request_timeout()
        failed_count = imported_count = skipped_count = processed_count = 0
        last_external_id: str | None = None
        async with httpx.AsyncClient(timeout=timeout) as client:
            index, identifier_column = await self.load_index_for_category(
                category, client
            )
            identifiers, available_count = self.ordered_identifiers(
                index=index,
                identifier_column=identifier_column,
                fraction=self.clamp_fraction(fraction),
            )
            if resume_after_external_id and resume_after_external_id not in identifiers:
                raise ValueError(
                    "The saved NIST checkpoint is not present in the current index; "
                    "start a new acquisition to avoid skipping records."
                )
            existing, base_url = await self._category_sync_details(category)
            checkpointed_identifiers = identifiers
            if resume_after_external_id:
                checkpoint_index = identifiers.index(resume_after_external_id)
                checkpointed_identifiers = identifiers[checkpoint_index + 1 :]
            candidates = self._unimported_identifiers(
                checkpointed_identifiers, existing
            )
            skipped_count = len(checkpointed_identifiers) - len(candidates)
            record_cap = max(0, max_records)
            remaining_cap = max(0, record_cap - max(0, processed_before))
            remaining = candidates[:remaining_cap]
            batch_size = max(1, batch_size)
            for offset in range(0, len(remaining), batch_size):
                if await should_cancel():
                    break
                batch = remaining[offset : offset + batch_size]
                successful, batch_failures = await self._fetch_sync_batch(
                    api_client=api_client,
                    client=client,
                    base_url=base_url,
                    batch=batch,
                    on_failure=on_failure,
                )
                mapped_count, batch_skips = await self._persist_sync_batch(
                    api_client=api_client,
                    category=category,
                    successful=successful,
                    on_skip=on_skip,
                )
                failed_count += batch_failures
                skipped_count += batch_skips
                processed_count += len(batch)
                imported_count += mapped_count
                last_external_id = batch[-1]
                await on_progress(
                    {
                        "progress": (
                            100.0 * processed_count / len(remaining)
                            if remaining
                            else 100.0
                        ),
                        "processed_count": processed_count,
                        "imported_count": imported_count,
                        "skipped_count": skipped_count,
                        "failed_count": failed_count,
                        "category": category,
                        "available_count": int(available_count),
                        "requested_count": len(identifiers),
                        "record_cap": record_cap,
                        "pending_count": max(0, len(candidates) - processed_count),
                        "last_external_id": last_external_id,
                    }
                )

            if not remaining:
                await on_progress(
                    {
                        "progress": 100.0,
                        "processed_count": 0,
                        "imported_count": 0,
                        "skipped_count": skipped_count,
                        "failed_count": failed_count,
                        "category": category,
                        "available_count": int(available_count),
                        "requested_count": len(identifiers),
                        "record_cap": record_cap,
                        "pending_count": max(0, len(candidates) - processed_count),
                        "last_external_id": last_external_id or resume_after_external_id,
                    }
                )

        local_counts = await asyncio.to_thread(
            self.repository.count_local_records_by_category
        )
        self.update_category_state(
            category,
            available_count=int(available_count),
            last_update=self.now_iso(),
        )
        return {
            "available_count": int(available_count),
            "requested_count": len(identifiers),
            "processed_count": processed_count,
            "imported_count": imported_count,
            "skipped_count": skipped_count,
            "failed_count": failed_count,
            "local_count": int(local_counts.get(category, 0)),
            "pending_count": max(0, len(candidates) - processed_count),
            "request_stats": api_client.request_statistics(),
        }

    # -------------------------------------------------------------------------
    async def _category_sync_details(
        self, category: Literal["experiments", "guest", "host"]
    ) -> tuple[set[str], str]:
        if category == "experiments":
            existing = await asyncio.to_thread(
                self.repository.list_nist_experiment_ids
            )
            base_url = "https://adsorption.nist.gov/isodb/api/isotherm/{}.json"
        elif category == "guest":
            existing = await asyncio.to_thread(
                self.repository.list_adsorbate_inchi_keys
            )
            base_url = "https://adsorption.nist.gov/isodb/api/gas/{}.json"
        else:
            existing = await asyncio.to_thread(
                self.repository.list_adsorbent_hash_keys
            )
            base_url = "https://adsorption.nist.gov/matdb/api/material/{}.json"
        return existing, base_url

    # -------------------------------------------------------------------------
    def _unimported_identifiers(
        self, identifiers: list[str], existing: set[str]
    ) -> list[str]:
        return [
            str(identifier)
            for identifier in identifiers
            if self.normalize_identifier(identifier) not in existing
        ]

    # -------------------------------------------------------------------------
    async def _fetch_sync_batch(
        self,
        *,
        api_client: NISTApiClient,
        client: httpx.AsyncClient,
        base_url: str,
        batch: list[str],
        on_failure: Any,
    ) -> tuple[list[tuple[str, dict[str, Any]]], int]:
        responses = await asyncio.gather(
            *(
                api_client.fetch_json(client, base_url.format(quote(identifier, safe="")))
                for identifier in batch
            )
        )
        successful = []
        failures = 0
        for identifier, payload in zip(batch, responses, strict=True):
            if isinstance(payload, dict):
                successful.append((identifier, payload))
            else:
                failures += 1
                await on_failure(
                    identifier,
                    "NIST record request failed or returned malformed data.",
                )
        return successful, failures

    # -------------------------------------------------------------------------
    async def _persist_sync_batch(
        self,
        *,
        api_client: NISTApiClient,
        category: Literal["experiments", "guest", "host"],
        successful: list[tuple[str, dict[str, Any]]],
        on_skip: Any,
    ) -> tuple[int, int]:
        if not successful:
            return 0, 0
        if category == "experiments":
            mapped_records, reasons = await self._map_experiment_batch(successful)
            identity_key = "external_key"
        else:
            mapped_records, reasons = self._map_material_batch(
                api_client, category, successful
            )
            identity_key = "inchi_key" if category == "guest" else "external_identifier"
        mapped_ids = {
            self.normalize_identifier(record.get(identity_key))
            for record in mapped_records
            if record.get(identity_key)
        }
        skipped = 0
        for identifier, _payload in successful:
            if self.normalize_identifier(identifier) in mapped_ids:
                continue
            skipped += 1
            await on_skip(
                identifier,
                reasons.get(identifier)
                or "NIST record was excluded by current schema or identity support.",
            )
        return len(mapped_records), skipped

    # -------------------------------------------------------------------------
    async def _map_experiment_batch(
        self, successful: list[tuple[str, dict[str, Any]]]
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        fetched = pd.DataFrame([payload for _identifier, payload in successful])
        single_component, binary_mixture = self.builder.build_datasets(fetched)
        records, reasons = self.mapper.experiment_records_with_skips(
            single_component, binary_mixture
        )
        if records:
            await asyncio.to_thread(self.repository.save_experiments, records, False)
        return records, reasons

    # -------------------------------------------------------------------------
    def _map_material_batch(
        self,
        api_client: NISTApiClient,
        category: Literal["guest", "host"],
        successful: list[tuple[str, dict[str, Any]]],
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        if category == "guest":
            identifier_column = api_client.guest_identifier
            extra_columns = api_client.extra_guest_columns
            drop_columns = ["synonyms"]
            record_type = "adsorbate"
        else:
            identifier_column = api_client.host_identifier
            extra_columns = api_client.extra_host_columns
            drop_columns = ["External_Resources", "synonyms"]
            record_type = "adsorbent"
        frame = api_client.prepare_materials_frame(
            [payload for _identifier, payload in successful],
            identifier_column=identifier_column,
            extra_columns=extra_columns,
            drop_columns=drop_columns,
        )
        records = self.mapper.material_records(frame, record_type)
        if category == "guest":
            self.repository.save_materials(records, [])
        else:
            self.repository.save_materials([], records)
        return records, {}

    # -------------------------------------------------------------------------
    async def ping_experiments_server(self) -> dict[str, Any]:
        return await self.ping_server("experiments")

    # -------------------------------------------------------------------------
    async def ping_guest_server(self) -> dict[str, Any]:
        return await self.ping_server("guest")

    # -------------------------------------------------------------------------
    async def ping_host_server(self) -> dict[str, Any]:
        return await self.ping_server("host")

    # -------------------------------------------------------------------------
    async def fetch_experiments_index(
        self, job_id: str | None = None
    ) -> dict[str, Any]:
        return await self.fetch_category_index("experiments", job_id=job_id)

    # -------------------------------------------------------------------------
    async def fetch_guest_index(self, job_id: str | None = None) -> dict[str, Any]:
        return await self.fetch_category_index("guest", job_id=job_id)

    # -------------------------------------------------------------------------
    async def fetch_host_index(self, job_id: str | None = None) -> dict[str, Any]:
        return await self.fetch_category_index("host", job_id=job_id)

    # -------------------------------------------------------------------------
    async def fetch_experiments_records(
        self, fraction: float, job_id: str | None = None
    ) -> dict[str, Any]:
        return await self.fetch_category_records("experiments", fraction, job_id=job_id)

    # -------------------------------------------------------------------------
    async def fetch_guest_records(
        self, fraction: float, job_id: str | None = None
    ) -> dict[str, Any]:
        return await self.fetch_category_records("guest", fraction, job_id=job_id)

    # -------------------------------------------------------------------------
    async def fetch_host_records(
        self, fraction: float, job_id: str | None = None
    ) -> dict[str, Any]:
        return await self.fetch_category_records("host", fraction, job_id=job_id)

    # -------------------------------------------------------------------------
    async def enrich_properties(
        self, target: str, job_id: str | None = None
    ) -> dict[str, int | str]:
        logger.info("NIST properties enrichment starting (target=%s)", target)
        adsorption_data, guest_data, host_data = await asyncio.to_thread(
            self.repository.load_adsorption_datasets
        )

        if target == "guest":
            category = "guest"
            data = guest_data.copy()
            adsorbate_series = adsorption_data.get("adsorbate", pd.Series(dtype=str))
        elif target == "host":
            category = "host"
            data = host_data.copy()
            adsorbent_series = adsorption_data.get("adsorbent", pd.Series(dtype=str))
        else:
            raise ValueError("Target must be 'guest' or 'host'.")

        reference_data = await asyncio.to_thread(
            self.repository.load_nist_reference_materials, category
        )
        data = pd.concat([data, reference_data], ignore_index=True)
        if not data.empty:
            data["_normalized_name"] = (
                data["name"].astype("string").str.strip().str.casefold()
            )
            data = data.drop_duplicates(subset=["_normalized_name"], keep="last")
            data = data.drop(columns=["_normalized_name"])
        name_series = pd.concat(
            [
                adsorbate_series if category == "guest" else adsorbent_series,
                data.get("name", pd.Series(dtype=str)),
            ]
        )

        weight_col = "molecular_weight"
        formula_col = "molecular_formula"
        smile_col = "smile_code"

        names = (
            name_series.dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .loc[lambda series: series != ""]
            .unique()
            .tolist()
        )
        local_counts = await asyncio.to_thread(
            self.repository.count_local_records_by_category
        )

        if not names or data.empty:
            logger.info(
                "NIST properties enrichment skipped (target=%s, names=%d, data_empty=%s)",
                target,
                len(names),
                data.empty,
            )
            return {
                "category": category,
                "names_requested": len(names),
                "names_matched": 0,
                "rows_updated": 0,
                "local_count": int(local_counts.get(category, 0)),
            }

        if job_id:
            self.job_manager.update_progress(job_id, 10.0)

        for column in (weight_col, formula_col, smile_col):
            if column not in data.columns:
                data[column] = pd.NA

        public_data_config = self.config.application.public_data
        pubchem = PubChemClient(
            self.config.application.nist.pubchem_parallel_tasks,
            request_budget=self.pubchem_request_budget,
            request_timeout_seconds=public_data_config.request_timeout_seconds,
            retry_attempts=public_data_config.retry_attempts,
            retry_backoff_max_seconds=public_data_config.retry_backoff_max_seconds,
            max_response_bytes=public_data_config.sync_response_size_limit_bytes,
        )
        try:
            properties = await pubchem.fetch_properties_for_names(names)
        finally:
            await pubchem.close()

        if job_id:
            self.job_manager.update_progress(job_id, 50.0)

        properties_frame = pd.DataFrame(properties)
        if properties_frame.empty:
            logger.info(
                "NIST properties enrichment returned no results (target=%s, names=%d)",
                target,
                len(names),
            )
            return {
                "category": category,
                "names_requested": len(names),
                "names_matched": 0,
                "rows_updated": 0,
                "local_count": int(local_counts.get(category, 0)),
            }

        data["name"] = data["name"].astype("string").str.strip().str.lower()
        properties_frame["name"] = (
            properties_frame["name"].astype("string").str.strip().str.lower()
        )

        before_count = 0
        if weight_col in data.columns:
            before_count = int(data[weight_col].notna().sum())

        properties_frame = properties_frame.rename(
            columns={
                "molecular_weight": weight_col,
                "molecular_formula": formula_col,
                "smile": smile_col,
            }
        )
        if weight_col in properties_frame.columns:
            properties_frame[weight_col] = pd.to_numeric(
                properties_frame[weight_col], errors="coerce"
            )
        if weight_col in data.columns:
            data[weight_col] = pd.to_numeric(data[weight_col], errors="coerce")
        for col in (formula_col, smile_col):
            if col in properties_frame.columns:
                properties_frame[col] = properties_frame[col].astype("string")
            if col in data.columns:
                data[col] = data[col].astype("string")
        data_indexed = data.set_index("name")
        properties_indexed = properties_frame.set_index("name")
        data_indexed.update(properties_indexed)
        updated = data_indexed.reset_index()

        after_count = 0
        if weight_col in updated.columns:
            after_count = int(updated[weight_col].notna().sum())
        rows_updated = max(after_count - before_count, 0)

        if job_id:
            self.job_manager.update_progress(job_id, 80.0)

        if target == "guest":
            guest_records = self.mapper.material_records(updated, "adsorbate")
            await asyncio.to_thread(self.repository.save_materials, guest_records, [])
        else:
            host_records = self.mapper.material_records(updated, "adsorbent")
            await asyncio.to_thread(self.repository.save_materials, [], host_records)

        matched = int(
            properties_frame[[weight_col, formula_col, smile_col]]
            .notna()
            .any(axis=1)
            .sum()
        )
        logger.info(
            "NIST properties enrichment completed (target=%s, names_requested=%d, names_matched=%d, rows_updated=%d)",
            target,
            len(names),
            matched,
            rows_updated,
        )
        local_counts = await asyncio.to_thread(
            self.repository.count_local_records_by_category
        )
        self.update_category_state(category, last_update=self.now_iso())
        if job_id:
            self.job_manager.update_progress(job_id, 100.0)

        return {
            "category": category,
            "names_requested": len(names),
            "names_matched": matched,
            "rows_updated": rows_updated,
            "local_count": int(local_counts.get(category, 0)),
        }

    # -------------------------------------------------------------------------
    async def enrich_category_records(
        self,
        category: Literal["guest", "host"],
        job_id: str | None = None,
    ) -> dict[str, Any]:
        return await self.enrich_properties(target=category, job_id=job_id)

    # -------------------------------------------------------------------------
    async def enrich_guest_properties(
        self, job_id: str | None = None
    ) -> dict[str, Any]:
        return await self.enrich_category_records("guest", job_id=job_id)

    # -------------------------------------------------------------------------
    async def enrich_host_properties(self, job_id: str | None = None) -> dict[str, Any]:
        return await self.enrich_category_records("host", job_id=job_id)

    # -------------------------------------------------------------------------
    async def get_category_status(self) -> list[dict[str, Any]]:
        local_counts = await asyncio.to_thread(
            self.repository.count_local_records_by_category
        )
        status_entries: list[dict[str, Any]] = []
        for category in ("experiments", "guest", "host"):
            state = self.get_category_state(category)
            status_entries.append(
                {
                    "category": category,
                    "local_count": int(local_counts.get(category, 0)),
                    "available_count": int(state.get("available_count") or 0),
                    "last_update": state.get("last_update"),
                    "server_ok": state.get("server_ok"),
                    "server_checked_at": state.get("server_checked_at"),
                    "supports_enrichment": self.category_supports_enrichment(category),
                }
            )
        return status_entries

    # -------------------------------------------------------------------------
    async def get_status(self) -> dict[str, int | bool]:
        counts = await asyncio.to_thread(self.repository.count_nist_rows)
        data_available = any(value > 0 for value in counts.values())
        return {
            "data_available": data_available,
            **counts,
        }
