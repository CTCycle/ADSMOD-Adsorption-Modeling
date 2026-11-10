# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
import math
import time
from typing import Any, Literal
from urllib.parse import quote

import httpx
import pandas as pd

from server.common.utils.logger import logger
from server.configurations.settings import AdsmodConfig
from server.services.data.nistads import (
    NISTApiClient,
    NISTDatasetBuilder,
)
from server.services.data.nist_mapper import NISTCanonicalMapper
from server.repositories.nist import NISTRepository

NISTCategory = Literal["experiments", "guest", "host"]
SEARCH_CACHE_SECONDS = 600.0

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
        self._search_lock = asyncio.Lock()
        self._search_cache: dict[str, Any] | None = None
        self._index_lock = asyncio.Lock()
        self._index_cache: dict[str, tuple[float, pd.DataFrame, str]] = {}

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
    def select_identifiers(
        self,
        index: pd.DataFrame,
        identifier_column: str,
        fraction: float,
        requested: list[str] | None,
    ) -> tuple[list[str], int, list[str]]:
        """Return the full ordered index, or the requested records that the
        current index lists (in requested order) plus the unknown requests.

        Fractional sizing is applied later against records still missing
        locally, so repeated requests progress through the catalogue.
        """
        indexed, total = self.ordered_identifiers(index, identifier_column, 1.0)
        if not requested:
            return indexed, total, []
        known = {item.lower(): item for item in indexed}
        selected = [known[item.lower()] for item in requested if item.lower() in known]
        unknown = [item for item in requested if item.lower() not in known]
        return selected, total, unknown

    # -------------------------------------------------------------------------
    async def load_index_for_category(
        self,
        category: NISTCategory,
        client: httpx.AsyncClient,
        api_client: NISTApiClient | None = None,
    ) -> tuple[pd.DataFrame, str]:
        """Return a category index, reusing a snapshot younger than the TTL."""
        api_client = api_client or self._api_client()
        async with self._index_lock:
            cached = self._index_cache.get(category)
            if cached and time.monotonic() - cached[0] < SEARCH_CACHE_SECONDS:
                return cached[1], cached[2]
            if category == "experiments":
                index = await api_client.fetch_experiments_index(client)
                column = api_client.exp_identifier
            elif category == "guest":
                index = await api_client.fetch_guest_index(client)
                column = api_client.guest_identifier
            else:
                index = await api_client.fetch_host_index(client)
                column = api_client.host_identifier
            self._index_cache[category] = (time.monotonic(), index, column)
            return index, column

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
                category, client, api_client
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
    async def _search_indexes(self) -> dict[str, Any]:
        """Return the three NIST indexes, cached briefly across searches.

        NIST exposes no server-side search: query parameters are ignored by
        the index endpoints, so searches join the downloaded indexes locally.
        """
        async with self._search_lock:
            cached = self._search_cache
            if cached and time.monotonic() - cached["at"] < SEARCH_CACHE_SECONDS:
                return cached
            async with httpx.AsyncClient(timeout=self._request_timeout()) as client:
                experiments, _ = await self.load_index_for_category(
                    "experiments", client
                )
                guests, guest_column = await self.load_index_for_category(
                    "guest", client
                )
                hosts, host_column = await self.load_index_for_category(
                    "host", client
                )
            self._search_cache = {
                "at": time.monotonic(),
                "experiments": experiments.to_dict("records"),
                "guests": self._name_lookup(guests, guest_column),
                "hosts": self._name_lookup(hosts, host_column),
            }
            return self._search_cache

    # -------------------------------------------------------------------------
    @staticmethod
    def _name_lookup(
        frame: pd.DataFrame, identifier_column: str
    ) -> dict[str, tuple[str, list[str]]]:
        lookup: dict[str, tuple[str, list[str]]] = {}
        for row in frame.to_dict("records"):
            identifier = str(row.get(identifier_column) or "").strip()
            if not identifier:
                continue
            name = str(row.get("name") or "").strip()
            synonyms = [
                str(item).strip().casefold()
                for item in (row.get("synonyms") or [])
                if str(item).strip()
            ]
            lookup[identifier] = (name, [name.casefold(), *synonyms])
        return lookup

    # -------------------------------------------------------------------------
    @staticmethod
    def _matching_identifiers(
        lookup: dict[str, tuple[str, list[str]]], text: str | None
    ) -> set[str] | None:
        """Identifiers named `text`: exact name/synonym/key matches when there
        are any, otherwise names that contain it. Synonyms are matched exactly
        only, since partial synonym hits (e.g. 'methane' in 'dimethyl ether
        synonyms') bury the intended compound."""
        needle = (text or "").strip().casefold()
        if not needle:
            return None
        exact = {
            identifier
            for identifier, (_name, names) in lookup.items()
            if identifier.casefold() == needle or needle in names
        }
        if exact:
            return exact
        return {
            identifier
            for identifier, (_name, names) in lookup.items()
            if needle in names[0]
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def _isotherm_matches(
        row: dict[str, Any],
        *,
        adsorbate_keys: set[str] | None,
        adsorbent_keys: set[str] | None,
        experiment: str,
        temperature_range: tuple[float | None, float | None],
    ) -> bool:
        adsorbent = (row.get("adsorbent") or {}).get("hashkey")
        inchi_keys = {item.get("InChIKey") for item in row.get("adsorbates") or []}
        if adsorbent_keys is not None and adsorbent not in adsorbent_keys:
            return False
        if adsorbate_keys is not None and not (inchi_keys & adsorbate_keys):
            return False
        if experiment and experiment not in (
            f"{row.get('filename') or ''} {row.get('DOI') or ''}".casefold()
        ):
            return False
        low, high = temperature_range
        temperature = row.get("temperature")
        if low is None and high is None:
            return True
        if not isinstance(temperature, (int, float)):
            return False
        return (low is None or temperature >= low) and (
            high is None or temperature <= high
        )

    # -------------------------------------------------------------------------
    async def search_experiments(
        self,
        *,
        adsorbate: str | None,
        adsorbent: str | None,
        experiment: str | None,
        temperature_min: float | None,
        temperature_max: float | None,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        """Find NIST isotherms by adsorbate, adsorbent or experiment text."""
        if not any((adsorbate, adsorbent, experiment)):
            raise ValueError(
                "Provide an adsorbate, adsorbent or experiment (filename or DOI)."
            )
        indexes = await self._search_indexes()
        guests, hosts = indexes["guests"], indexes["hosts"]
        adsorbate_keys = self._matching_identifiers(guests, adsorbate)
        adsorbent_keys = self._matching_identifiers(hosts, adsorbent)
        needle = (experiment or "").strip().casefold()
        matches = [
            row
            for row in indexes["experiments"]
            if self._isotherm_matches(
                row,
                adsorbate_keys=adsorbate_keys,
                adsorbent_keys=adsorbent_keys,
                experiment=needle,
                temperature_range=(temperature_min, temperature_max),
            )
        ]
        existing = await asyncio.to_thread(self.repository.list_nist_experiment_ids)
        page = matches[offset : offset + limit]
        return {
            "total": len(matches),
            "items": [
                self._search_item(row, guests, hosts, existing) for row in page
            ],
        }

    # -------------------------------------------------------------------------
    def _search_item(
        self,
        row: dict[str, Any],
        guests: dict[str, tuple[str, list[str]]],
        hosts: dict[str, tuple[str, list[str]]],
        existing: set[str],
    ) -> dict[str, Any]:
        filename = str(row.get("filename") or "")
        hashkey = (row.get("adsorbent") or {}).get("hashkey")
        keys = [
            str(item["InChIKey"])
            for item in row.get("adsorbates") or []
            if item.get("InChIKey")
        ]
        temperature = row.get("temperature")
        return {
            "filename": filename,
            "doi": row.get("DOI") or None,
            "adsorbent_name": hosts.get(hashkey, (None,))[0],
            "adsorbent_hashkey": hashkey,
            "adsorbate_names": [guests.get(key, (key,))[0] or key for key in keys],
            "adsorbate_inchi_keys": keys,
            "temperature_k": (
                float(temperature) if isinstance(temperature, (int, float)) else None
            ),
            "local": self.normalize_identifier(filename) in existing,
        }

    # -------------------------------------------------------------------------
    async def sync_targeted_dependencies(self, identifiers: list[str]) -> list[str]:
        """Store the compound and material records of targeted isotherms first.

        NIST isotherm payloads carry only names and identities; unit
        conversions (for example wt% needing a molar mass) rely on the full
        compound record, which bulk acquisition fetches as its own collection.
        Records already stored are skipped. Identifiers that could not be
        fetched are returned so the run can report them explicitly.
        """
        indexes = await self._search_indexes()
        wanted = {item.lower() for item in identifiers}
        guests: set[str] = set()
        hosts: set[str] = set()
        for row in indexes["experiments"]:
            if str(row.get("filename") or "").lower() not in wanted:
                continue
            if (row.get("adsorbent") or {}).get("hashkey"):
                hosts.add(str(row["adsorbent"]["hashkey"]))
            guests.update(
                str(item["InChIKey"])
                for item in row.get("adsorbates") or []
                if item.get("InChIKey")
            )

        failed: list[str] = []

        async def collect(identifier: str, _reason: str) -> None:
            failed.append(identifier)

        async def ignore(_identifier: str, _reason: str) -> None:
            return None

        api_client = self._api_client()
        async with httpx.AsyncClient(timeout=self._request_timeout()) as client:
            for category, keys in (("guest", guests), ("host", hosts)):
                existing, base_url = await self._category_sync_details(category)
                missing = self._unimported_identifiers(sorted(keys), existing)
                if not missing:
                    continue
                successful, _failures = await self._fetch_sync_batch(
                    api_client=api_client,
                    client=client,
                    base_url=base_url,
                    batch=missing,
                    on_failure=collect,
                )
                await self._persist_sync_batch(
                    api_client=api_client,
                    category=category,
                    successful=successful,
                    on_skip=ignore,
                )
        if failed:
            logger.warning(
                "NIST dependency records could not be fetched: %s", sorted(failed)
            )
        return sorted(failed)

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
        max_records: int | None,
        on_progress: Any,
        should_cancel: Any,
        on_failure: Any,
        on_skip: Any,
        processed_before: int = 0,
        resume_after_external_id: str | None = None,
        overwrite_local_edits: bool = False,
        identifiers: list[str] | None = None,
        excluded: frozenset[str] | set[str] = frozenset(),
    ) -> dict[str, Any]:
        """Fetch one NIST collection in idempotent, checkpointed batches.

        Existing experiments and `excluded` identifiers (records settled as
        unimportable by earlier runs) are skipped, so the fractional budget
        covers records that can still be imported. With `overwrite_local_edits`, locally
        edited NIST experiments are fetched again and restored from source.
        Explicit `identifiers` (targeted acquisition) replace the fractional
        selection but must still be present in the current NIST index.
        """
        api_client = self._api_client()
        timeout = self._request_timeout()
        failed_count = imported_count = skipped_count = processed_count = 0
        async with httpx.AsyncClient(timeout=timeout) as client:
            index, identifier_column = await self.load_index_for_category(
                category, client
            )
            targeted = bool(identifiers)
            identifiers, available_count, unknown = self.select_identifiers(
                index, identifier_column, fraction, identifiers
            )
            for identifier in unknown:
                failed_count += 1
                await on_failure(
                    identifier, "The identifier is not present in the current NIST index."
                )
            checkpoint_frozen = False
            last_external_id = resume_after_external_id
            if resume_after_external_id and resume_after_external_id not in identifiers:
                raise ValueError(
                    "The saved NIST checkpoint is not present in the current index; "
                    "start a new acquisition to avoid skipping records."
                )
            existing, base_url = await self._category_sync_details(category)
            restore = category == "experiments" and overwrite_local_edits
            if restore:
                existing = existing - await asyncio.to_thread(
                    self.repository.edited_experiment_ids
                )
            budget = (
                len(identifiers)
                if targeted
                else int(math.ceil(self.clamp_fraction(fraction) * available_count))
            )
            record_cap = None if max_records is None else max(0, max_records)
            eligible, remaining, skipped_count = self._budgeted_selection(
                identifiers,
                existing | excluded,
                resume_after=resume_after_external_id,
                budget=budget,
                record_cap=record_cap,
                processed_before=processed_before,
            )
            batch_size = max(1, batch_size)
            for offset in range(0, len(remaining), batch_size):
                if await should_cancel():
                    break
                batch = remaining[offset : offset + batch_size]
                successful, failed_ids = await self._fetch_sync_batch(
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
                    restore_edited=restore,
                )
                failed_count += len(failed_ids)
                skipped_count += batch_skips
                processed_count += len(batch)
                imported_count += mapped_count
                # The checkpoint never moves past a failed record, so a resume
                # replays it (records already stored are skipped on replay).
                if not checkpoint_frozen:
                    if failed_ids:
                        checkpoint_frozen = True
                        first_failed = batch.index(failed_ids[0])
                        if first_failed:
                            last_external_id = batch[first_failed - 1]
                    else:
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
                        "requested_count": budget,
                        "record_cap": record_cap,
                        "pending_count": max(0, len(eligible) - processed_count),
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
                        "requested_count": budget,
                        "record_cap": record_cap,
                        "pending_count": max(0, len(eligible) - processed_count),
                        "last_external_id": last_external_id,
                    }
                )

        local_counts = await asyncio.to_thread(
            self.repository.count_local_records_by_category
        )
        return {
            "available_count": int(available_count),
            "requested_count": budget,
            "processed_count": processed_count,
            "imported_count": imported_count,
            "skipped_count": skipped_count,
            "failed_count": failed_count,
            "local_count": int(local_counts.get(category, 0)),
            "pending_count": max(0, len(eligible) - processed_count),
            "request_stats": api_client.request_statistics(),
        }

    # -------------------------------------------------------------------------
    def _budgeted_selection(
        self,
        identifiers: list[str],
        settled: set[str],
        *,
        resume_after: str | None,
        budget: int,
        record_cap: int | None,
        processed_before: int,
    ) -> tuple[list[str], list[str], int]:
        """Pick the records a run fetches: (eligible, remaining, skipped count).

        A fractional request covers the next share of the catalogue that is
        not yet settled locally, so repeated requests make progress. Records
        scanned past but already settled count as skipped. A resume starts
        after the checkpointed identifier.
        """
        if resume_after:
            identifiers = identifiers[identifiers.index(resume_after) + 1 :]
        candidates = self._unimported_identifiers(identifiers, settled)
        done = max(0, processed_before)
        eligible = candidates[: max(0, budget - done)]
        remaining = (
            eligible
            if record_cap is None
            else candidates[: max(0, min(budget, record_cap) - done)]
        )
        scanned = (
            identifiers[: identifiers.index(remaining[-1]) + 1]
            if remaining
            else identifiers
        )
        return eligible, remaining, len(scanned) - len(remaining)

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
    ) -> tuple[list[tuple[str, dict[str, Any]]], list[str]]:
        responses = await asyncio.gather(
            *(
                api_client.fetch_json(client, base_url.format(quote(identifier, safe="")))
                for identifier in batch
            )
        )
        successful = []
        failures: list[str] = []
        for identifier, payload in zip(batch, responses, strict=True):
            if isinstance(payload, dict):
                successful.append((identifier, payload))
            else:
                failures.append(identifier)
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
        restore_edited: bool = False,
    ) -> tuple[int, int]:
        if not successful:
            return 0, 0
        if category == "experiments":
            mapped_records, reasons = await self._map_experiment_batch(
                successful, restore_edited
            )
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
                or "NIST record was excluded: no complete measurement rows or unsupported schema/identity.",
            )
        return len(mapped_records), skipped

    # -------------------------------------------------------------------------
    async def _map_experiment_batch(
        self, successful: list[tuple[str, dict[str, Any]]], restore_edited: bool = False
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
            await asyncio.to_thread(
                self.repository.save_experiments, records, restore_edited
            )
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
        self.mapper.attach_synonyms(
            records,
            [payload for _identifier, payload in successful],
            identifier_column,
            "nist_identifier" if category == "guest" else "external_identifier",
        )
        if category == "guest":
            self.repository.save_materials(records, [])
        else:
            self.repository.save_materials([], records)
        return records, {}


__all__ = ["NISTCategory", "NISTDataService"]
