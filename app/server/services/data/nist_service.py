# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
import math
from typing import Any, Literal
from urllib.parse import quote

import httpx
import pandas as pd

from server.configurations.settings import AdsmodConfig
from server.services.data.nistads import (
    NISTApiClient,
    NISTDatasetBuilder,
)
from server.services.data.nist_mapper import NISTCanonicalMapper
from server.repositories.nist import NISTRepository

NISTCategory = Literal["experiments", "guest", "host"]

###############################################################################
class NISTDataService:
    """Remote NIST/ARPA-E access used by durable public-data acquisition runs."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        config: AdsmodConfig,
        repository: NISTRepository,
        mapper: NISTCanonicalMapper,
    ) -> None:
        self.config = config
        self.repository = repository
        self.mapper = mapper
        self.builder = NISTDatasetBuilder()

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
    def clamp_fraction(fraction: float) -> float:
        return min(1.0, max(0.001, float(fraction)))

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
        category: NISTCategory,
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
    async def ping_server(self, category: NISTCategory) -> dict[str, Any]:
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

        return {
            "category": category,
            "server_ok": server_ok,
            "request_stats": {
                "requests": 1,
                "retries": 0,
                "rate_limits": int(rate_limited),
                "failures": int(not server_ok),
            },
        }

    # -------------------------------------------------------------------------
    async def index_category(self, category: NISTCategory) -> dict[str, Any]:
        """Count the records NIST currently publishes for one collection."""
        api_client = self._api_client()
        async with httpx.AsyncClient(timeout=self._request_timeout()) as client:
            index, identifier_column = await self.load_index_for_category(
                category, client
            )
        _identifiers, available_count = self.ordered_identifiers(
            index=index, identifier_column=identifier_column, fraction=1.0
        )
        local_counts = await asyncio.to_thread(
            self.repository.count_local_records_by_category
        )
        return {
            "category": category,
            "available_count": int(available_count),
            "local_count": int(local_counts.get(category, 0)),
            "request_stats": api_client.request_statistics(),
        }

    # -------------------------------------------------------------------------
    async def local_counts(self) -> dict[str, int]:
        counts = await asyncio.to_thread(
            self.repository.count_local_records_by_category
        )
        return {key: int(value) for key, value in counts.items()}

    # -------------------------------------------------------------------------
    async def sync_category_records(
        self,
        *,
        category: NISTCategory,
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
        self, category: NISTCategory
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
        category: NISTCategory,
        successful: list[tuple[str, dict[str, Any]]],
        on_skip: Any,
    ) -> tuple[int, int]:
        if not successful:
            return 0, 0
        if category == "experiments":
            mapped_records, reasons = await self._map_experiment_batch(successful)
            identity_key = "external_key"
        else:
            mapped_records, reasons = await asyncio.to_thread(
                self._map_material_batch, api_client, category, successful
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
        self.mapper.attach_source_identities(
            records, [payload for _identifier, payload in successful]
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


__all__ = ["NISTCategory", "NISTDataService"]
