from __future__ import annotations

import asyncio
import math
import threading
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Literal

import httpx
import pandas as pd

from ADSMOD.server.configurations import get_server_settings
from ADSMOD.server.common.utils.logger import logger
from ADSMOD.server.repositories.queries.nist import NISTDataSerializer
from ADSMOD.server.services.data.nistads import (
    NISTApiClient,
    NISTDatasetBuilder,
    PubChemClient,
)
from ADSMOD.server.services.jobs import job_manager


class NISTDataService:
    def __init__(self) -> None:
        self.serializer = NISTDataSerializer()
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
        api_client = NISTApiClient(get_server_settings().nist.parallel_tasks)
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
            job_manager.update_progress(job_id, 0.0)

        experiments_result = await self.fetch_experiments_records(
            experiments_fraction, job_id=None
        )
        if job_id:
            job_manager.update_progress(job_id, 35.0)

        guest_result = await self.fetch_guest_records(guest_fraction, job_id=None)
        if job_id:
            job_manager.update_progress(job_id, 70.0)

        host_result = await self.fetch_host_records(host_fraction, job_id=None)
        if job_id:
            job_manager.update_progress(job_id, 100.0)

        return {
            "experiments_count": int(experiments_result.get("fetched_count", 0)),
            "single_component_rows": int(
                experiments_result.get("single_component_rows", 0)
            ),
            "binary_mixture_rows": int(
                experiments_result.get("binary_mixture_rows", 0)
            ),
            "guest_rows": int(guest_result.get("fetched_count", 0)),
            "host_rows": int(host_result.get("fetched_count", 0)),
        }

    # -------------------------------------------------------------------------
    async def ping_server(
        self, category: Literal["experiments", "guest", "host"]
    ) -> dict[str, Any]:
        api_client = NISTApiClient(get_server_settings().nist.parallel_tasks)
        url_by_category = {
            "experiments": api_client.url_isotherms,
            "guest": api_client.url_guest_index,
            "host": api_client.url_host_index,
        }
        url = url_by_category[category]

        server_ok = False
        timeout = httpx.Timeout(10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                server_ok = True
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
        }

    # -------------------------------------------------------------------------
    async def fetch_category_index(
        self,
        category: Literal["experiments", "guest", "host"],
        job_id: str | None = None,
    ) -> dict[str, Any]:
        if job_id:
            job_manager.update_progress(job_id, 10.0)

        timeout = httpx.Timeout(30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            index, _ = await self.load_index_for_category(category, client)

        available_count = int(len(index))
        if job_id:
            job_manager.update_progress(job_id, 100.0)

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
        api_client = NISTApiClient(get_server_settings().nist.parallel_tasks)

        if job_id:
            job_manager.update_progress(job_id, 5.0)

        timeout = httpx.Timeout(30.0)
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
                    self.serializer.list_nist_experiment_ids
                )
            elif category == "guest":
                existing_ids = await asyncio.to_thread(
                    self.serializer.list_adsorbate_inchi_keys
                )
            else:
                existing_ids = await asyncio.to_thread(
                    self.serializer.list_adsorbent_hash_keys
                )

            if job_id:
                job_manager.update_progress(job_id, 25.0)

            identifiers_to_fetch = [
                identifier
                for identifier in requested_identifiers
                if self.normalize_identifier(identifier) not in existing_ids
            ]

            fetched_count = 0
            single_component_rows = 0
            binary_mixture_rows = 0

            if category == "experiments":
                experiments_data = (
                    await api_client.fetch_experiments_data_by_identifiers(
                        client, identifiers_to_fetch
                    )
                )
                fetched_count = int(len(experiments_data))
                if job_id:
                    job_manager.update_progress(job_id, 65.0)

                single_component, binary_mixture = self.builder.build_datasets(
                    experiments_data
                )
                single_component_rows = int(len(single_component))
                binary_mixture_rows = int(len(binary_mixture))
                if job_id:
                    job_manager.update_progress(job_id, 80.0)

                await asyncio.to_thread(
                    self.serializer.save_adsorption_datasets,
                    single_component,
                    binary_mixture,
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
                    job_manager.update_progress(job_id, 70.0)
                await asyncio.to_thread(
                    self.serializer.save_materials_datasets,
                    guest_data,
                    None,
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
                    job_manager.update_progress(job_id, 70.0)
                await asyncio.to_thread(
                    self.serializer.save_materials_datasets,
                    None,
                    host_data,
                )

        local_counts = await asyncio.to_thread(
            self.serializer.count_local_records_by_category
        )
        local_count = int(local_counts.get(category, 0))

        self.update_category_state(
            category,
            available_count=int(available_count),
            last_update=self.now_iso(),
        )
        if job_id:
            job_manager.update_progress(job_id, 100.0)

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
            "single_component_rows": single_component_rows,
            "binary_mixture_rows": binary_mixture_rows,
        }

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
            self.serializer.load_adsorption_datasets
        )

        if target == "guest":
            category = "guest"
            data = guest_data.copy()
            adsorbate_series = adsorption_data.get("adsorbate", pd.Series(dtype=str))
            name_series = pd.concat(
                [
                    adsorbate_series,
                    data.get("name", pd.Series(dtype=str)),
                ]
            )
        elif target == "host":
            category = "host"
            data = host_data.copy()
            adsorbent_series = adsorption_data.get("adsorbent", pd.Series(dtype=str))
            name_series = pd.concat(
                [
                    adsorbent_series,
                    data.get("name", pd.Series(dtype=str)),
                ]
            )
        else:
            raise ValueError("Target must be 'guest' or 'host'.")

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
            self.serializer.count_local_records_by_category
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
            job_manager.update_progress(job_id, 10.0)

        for column in (weight_col, formula_col, smile_col):
            if column not in data.columns:
                data[column] = pd.NA

        pubchem = PubChemClient(get_server_settings().nist.pubchem_parallel_tasks)
        properties = await pubchem.fetch_properties_for_names(names)

        if job_id:
            job_manager.update_progress(job_id, 50.0)

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
            job_manager.update_progress(job_id, 80.0)

        if target == "guest":
            await asyncio.to_thread(
                self.serializer.save_materials_datasets, updated, None
            )
        else:
            await asyncio.to_thread(
                self.serializer.save_materials_datasets, None, updated
            )

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
            self.serializer.count_local_records_by_category
        )
        self.update_category_state(category, last_update=self.now_iso())
        if job_id:
            job_manager.update_progress(job_id, 100.0)

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
            self.serializer.count_local_records_by_category
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
        counts = await asyncio.to_thread(self.serializer.count_nist_rows)
        data_available = any(value > 0 for value in counts.values())
        return {
            "data_available": data_available,
            **counts,
        }

