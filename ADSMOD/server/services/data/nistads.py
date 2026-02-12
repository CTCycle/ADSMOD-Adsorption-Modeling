from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Literal

import httpx
import pandas as pd
import pubchempy as pcp

from ADSMOD.server.configurations import server_settings
from ADSMOD.server.common.utils.encoding import (
    decode_json_response_bytes,
    sanitize_dataframe_strings,
)
from ADSMOD.server.common.utils.logger import logger
from ADSMOD.server.repositories.queries.nist import NISTDataSerializer
from ADSMOD.server.services.jobs import job_manager


###############################################################################
class NISTDatasetBuilder:
    def __init__(self) -> None:
        self.raw_drop_cols = [
            "DOI",
            "category",
            "tabular_data",
            "digitizer",
            "isotherm_type",
            "articleSource",
            "concentrationUnits",
            "compositionType",
        ]
        self.single_explode_cols = ["pressure", "adsorbed_amount"]
        self.single_drop_cols = [
            "date",
            "adsorbent",
            "adsorbates",
            "num_guests",
            "isotherm_data",
            "adsorbent_ID",
            "adsorbates_ID",
        ]
        self.binary_explode_cols = [
            "compound_1_pressure",
            "compound_2_pressure",
            "compound_1_adsorption",
            "compound_2_adsorption",
            "compound_1_composition",
            "compound_2_composition",
        ]
        self.binary_drop_cols = [
            "date",
            "adsorbent",
            "adsorbates",
            "num_guests",
            "isotherm_data",
            "adsorbent_ID",
            "adsorbates_ID",
            "adsorbate_name",
            "total_pressure",
            "all_species_data",
            "compound_1_data",
            "compound_2_data",
        ]

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_string_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
        return sanitize_dataframe_strings(dataframe)

    # -------------------------------------------------------------------------
    def drop_excluded_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.drop(columns=self.raw_drop_cols, errors="ignore")

    # -------------------------------------------------------------------------
    def split_by_mixture_complexity(
        self, dataframe: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if dataframe.empty:
            return dataframe, dataframe
        dataframe = dataframe.copy()
        dataframe["num_guests"] = dataframe["adsorbates"].str.len()
        single_component = dataframe.loc[dataframe["num_guests"] == 1].copy()
        binary_mixture = dataframe.loc[dataframe["num_guests"] == 2].copy()
        return single_component, binary_mixture

    # -------------------------------------------------------------------------
    def is_single_component(self, dataframe: pd.DataFrame) -> bool:
        return (dataframe["num_guests"] == 1).all()

    # -------------------------------------------------------------------------
    def is_binary_mixture(self, dataframe: pd.DataFrame) -> bool:
        return (dataframe["num_guests"] == 2).all()

    # -------------------------------------------------------------------------
    def add_material_fields(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe["adsorbent_ID"] = dataframe["adsorbent"].apply(
            lambda x: (x or {}).get("hashkey")
        )
        dataframe["adsorbent_name"] = dataframe["adsorbent"].apply(
            lambda x: str((x or {}).get("name", "")).lower()
        )
        dataframe["adsorbates_ID"] = dataframe["adsorbates"].apply(
            lambda x: [
                item.get("InChIKey") for item in (x or []) if isinstance(item, dict)
            ]
        )
        dataframe["adsorbate_name"] = dataframe["adsorbates"].apply(
            lambda x: [
                str(item.get("name", "")).lower()
                for item in (x or [])
                if isinstance(item, dict)
            ]
        )
        return dataframe

    # -------------------------------------------------------------------------
    def add_single_component_fields(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe["pressure"] = dataframe["isotherm_data"].apply(
            lambda x: [
                item.get("pressure") for item in (x or []) if isinstance(item, dict)
            ]
        )
        dataframe["adsorbed_amount"] = dataframe["isotherm_data"].apply(
            lambda x: [
                item.get("total_adsorption")
                for item in (x or [])
                if isinstance(item, dict)
            ]
        )
        dataframe["adsorbate_name"] = dataframe["adsorbates"].apply(
            lambda x: (
                str(x[0].get("name", "")).lower() if isinstance(x, list) and x else ""
            )
        )
        return dataframe

    # -------------------------------------------------------------------------
    def add_binary_mixture_fields(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        data_placeholder = {"composition": 1.0, "adsorption": 1.0}
        dataframe["total_pressure"] = dataframe["isotherm_data"].apply(
            lambda x: [
                item.get("pressure") for item in (x or []) if isinstance(item, dict)
            ]
        )
        dataframe["all_species_data"] = dataframe["isotherm_data"].apply(
            lambda x: [
                item.get("species_data") for item in (x or []) if isinstance(item, dict)
            ]
        )
        dataframe["compound_1"] = dataframe["adsorbate_name"].apply(
            lambda x: str(x[0]).lower() if isinstance(x, list) and x else ""
        )
        dataframe["compound_2"] = dataframe["adsorbate_name"].apply(
            lambda x: str(x[1]).lower() if isinstance(x, list) and len(x) > 1 else ""
        )
        dataframe["compound_1_data"] = dataframe["all_species_data"].apply(
            lambda x: [item[0] if item else data_placeholder for item in (x or [])]
        )
        dataframe["compound_2_data"] = dataframe["all_species_data"].apply(
            lambda x: [
                item[1] if item and len(item) > 1 else data_placeholder
                for item in (x or [])
            ]
        )
        dataframe["compound_1_composition"] = dataframe["compound_1_data"].apply(
            lambda x: [item.get("composition") for item in (x or [])]
        )
        dataframe["compound_2_composition"] = dataframe["compound_2_data"].apply(
            lambda x: [item.get("composition") for item in (x or [])]
        )
        dataframe["compound_1_pressure"] = dataframe.apply(
            lambda row: [
                a * b
                for a, b in zip(
                    row["compound_1_composition"], row["total_pressure"], strict=False
                )
            ],
            axis=1,
        )
        dataframe["compound_2_pressure"] = dataframe.apply(
            lambda row: [
                a * b
                for a, b in zip(
                    row["compound_2_composition"], row["total_pressure"], strict=False
                )
            ],
            axis=1,
        )
        dataframe["compound_1_adsorption"] = dataframe["compound_1_data"].apply(
            lambda x: [item.get("adsorption") for item in (x or [])]
        )
        dataframe["compound_2_adsorption"] = dataframe["compound_2_data"].apply(
            lambda x: [item.get("adsorption") for item in (x or [])]
        )
        return dataframe

    # -------------------------------------------------------------------------
    def extract_nested_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if dataframe.empty:
            return dataframe

        dataframe = self.add_material_fields(dataframe)
        if self.is_single_component(dataframe):
            return self.add_single_component_fields(dataframe)
        if self.is_binary_mixture(dataframe):
            return self.add_binary_mixture_fields(dataframe)

        return dataframe

    # -------------------------------------------------------------------------
    def expand_dataset(
        self, single_component: pd.DataFrame, binary_mixture: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if single_component.empty:
            single_dataset = pd.DataFrame()
        else:
            single_dataset = single_component.explode(self.single_explode_cols)
            single_dataset = single_dataset.drop(
                columns=self.single_drop_cols, errors="ignore"
            )
            single_dataset = single_dataset.dropna().reset_index(drop=True)
            single_dataset = single_dataset.rename(
                columns={
                    "filename": "name",
                    "adsorptionUnits": "adsorption_units",
                    "pressureUnits": "pressure_units",
                    "adsorbent_name": "adsorbent",
                    "adsorbate_name": "adsorbate",
                }
            )
            single_dataset = self.normalize_string_columns(single_dataset)

        if binary_mixture.empty:
            binary_dataset = pd.DataFrame()
        else:
            binary_dataset = binary_mixture.explode(self.binary_explode_cols)
            binary_dataset = binary_dataset.drop(
                columns=self.binary_drop_cols, errors="ignore"
            )
            binary_dataset = binary_dataset.dropna().reset_index(drop=True)
            binary_dataset = binary_dataset.rename(
                columns={
                    "filename": "name",
                    "adsorptionUnits": "adsorption_units",
                    "pressureUnits": "pressure_units",
                }
            )
            binary_dataset = self.normalize_string_columns(binary_dataset)

        return single_dataset, binary_dataset

    # -------------------------------------------------------------------------
    def build_datasets(
        self, adsorption_data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if adsorption_data is None or adsorption_data.empty:
            return pd.DataFrame(), pd.DataFrame()

        cleaned = self.drop_excluded_columns(adsorption_data)
        if cleaned.empty:
            return pd.DataFrame(), pd.DataFrame()

        single_component, binary_mixture = self.split_by_mixture_complexity(cleaned)
        if not single_component.empty:
            single_component = self.extract_nested_data(single_component)
        if not binary_mixture.empty:
            binary_mixture = self.extract_nested_data(binary_mixture)

        return self.expand_dataset(single_component, binary_mixture)


###############################################################################
class NISTApiClient:
    def __init__(self, parallel_tasks: int) -> None:
        self.parallel_tasks = max(1, int(parallel_tasks))
        self.semaphore = asyncio.Semaphore(self.parallel_tasks)
        self.exp_identifier = "filename"
        self.guest_identifier = "InChIKey"
        self.host_identifier = "hashkey"
        self.url_isotherms = "https://adsorption.nist.gov/isodb/api/isotherms.json"
        self.url_guest_index = "https://adsorption.nist.gov/isodb/api/gases.json"
        self.url_host_index = "https://adsorption.nist.gov/matdb/api/materials.json"
        self.extra_guest_columns = [
            "molecular_weight",
            "molecular_formula",
            "smile_code",
        ]
        self.extra_host_columns = [
            "molecular_weight",
            "molecular_formula",
            "smile_code",
        ]

    # -------------------------------------------------------------------------
    async def fetch_json(
        self, client: httpx.AsyncClient, url: str
    ) -> dict[str, Any] | list[Any] | None:
        async with self.semaphore:
            try:
                response = await client.get(url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning("Failed to fetch %s: %s", url, exc)
                return None
            try:
                return decode_json_response_bytes(response.content)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                logger.warning("Invalid JSON from %s: %s", url, exc)
                return None

    # -------------------------------------------------------------------------
    async def fetch_multiple(
        self, client: httpx.AsyncClient, urls: list[str]
    ) -> list[Any]:
        if not urls:
            return []
        tasks = [self.fetch_json(client, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return [result for result in results if result is not None]

    # -------------------------------------------------------------------------
    async def fetch_experiments_index(self, client: httpx.AsyncClient) -> pd.DataFrame:
        payload = await self.fetch_json(client, self.url_isotherms)
        if payload is None:
            raise ValueError("Failed to retrieve NIST adsorption isotherm index.")
        return pd.DataFrame(payload)

    # -------------------------------------------------------------------------
    async def fetch_experiments_data(
        self,
        client: httpx.AsyncClient,
        experiments_index: pd.DataFrame,
        experiments_fraction: float,
    ) -> pd.DataFrame:
        if experiments_index.empty:
            return pd.DataFrame()
        if self.exp_identifier not in experiments_index.columns:
            raise ValueError("NIST isotherm index missing filename column.")
        identifiers = experiments_index[self.exp_identifier].tolist()
        if not identifiers or experiments_fraction <= 0:
            return pd.DataFrame()
        num_samples = int(math.ceil(experiments_fraction * len(identifiers)))
        urls = [
            f"https://adsorption.nist.gov/isodb/api/isotherm/{identifier}.json"
            for identifier in identifiers[:num_samples]
        ]
        results = await self.fetch_multiple(client, urls)
        return pd.DataFrame(results)

    # -------------------------------------------------------------------------
    async def fetch_experiments_data_by_identifiers(
        self, client: httpx.AsyncClient, identifiers: list[str]
    ) -> pd.DataFrame:
        filtered_identifiers = [identifier for identifier in identifiers if identifier]
        if not filtered_identifiers:
            return pd.DataFrame()
        urls = [
            f"https://adsorption.nist.gov/isodb/api/isotherm/{identifier}.json"
            for identifier in filtered_identifiers
        ]
        results = await self.fetch_multiple(client, urls)
        return pd.DataFrame(results)

    # -------------------------------------------------------------------------
    async def fetch_materials_index(
        self, client: httpx.AsyncClient
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        guest_payload = await self.fetch_json(client, self.url_guest_index)
        host_payload = await self.fetch_json(client, self.url_host_index)
        guest_data = pd.DataFrame(guest_payload or [])
        host_data = pd.DataFrame(host_payload or [])
        return guest_data, host_data

    # -------------------------------------------------------------------------
    async def fetch_guest_index(self, client: httpx.AsyncClient) -> pd.DataFrame:
        payload = await self.fetch_json(client, self.url_guest_index)
        return pd.DataFrame(payload or [])

    # -------------------------------------------------------------------------
    async def fetch_host_index(self, client: httpx.AsyncClient) -> pd.DataFrame:
        payload = await self.fetch_json(client, self.url_host_index)
        return pd.DataFrame(payload or [])

    # -------------------------------------------------------------------------
    def build_material_urls(
        self, identifiers: list[str], fraction: float, url_template: str
    ) -> list[str]:
        if not identifiers or fraction <= 0:
            return []
        num_samples = int(math.ceil(fraction * len(identifiers)))
        return [
            url_template.format(identifier=identifier)
            for identifier in identifiers[:num_samples]
        ]

    # -------------------------------------------------------------------------
    def prepare_materials_frame(
        self,
        results: list[Any],
        identifier_column: str,
        extra_columns: list[str],
        drop_columns: list[str],
    ) -> pd.DataFrame:
        data = pd.DataFrame(results)
        if data.empty:
            return pd.DataFrame(columns=[identifier_column, "name", *extra_columns])
        data = data.drop(columns=drop_columns, errors="ignore")
        data["name"] = data["name"].astype(str).str.lower()
        for col in extra_columns:
            data[col] = pd.NA
        return data

    # -------------------------------------------------------------------------
    async def fetch_material_dataset(
        self,
        client: httpx.AsyncClient,
        index: pd.DataFrame,
        identifier_column: str,
        fraction: float,
        url_template: str,
        extra_columns: list[str],
        drop_columns: list[str],
        label: str,
    ) -> pd.DataFrame:
        if index.empty or fraction <= 0:
            logger.warning("No %s index available for NIST fetch.", label)
            return pd.DataFrame()
        if identifier_column not in index.columns:
            raise ValueError(f"NIST {label} index missing {identifier_column} column.")
        identifiers = index[identifier_column].tolist()
        urls = self.build_material_urls(identifiers, fraction, url_template)
        results = await self.fetch_multiple(client, urls)
        return self.prepare_materials_frame(
            results, identifier_column, extra_columns, drop_columns
        )

    # -------------------------------------------------------------------------
    async def fetch_material_dataset_by_identifiers(
        self,
        client: httpx.AsyncClient,
        identifiers: list[str],
        identifier_column: str,
        url_template: str,
        extra_columns: list[str],
        drop_columns: list[str],
    ) -> pd.DataFrame:
        filtered_identifiers = [identifier for identifier in identifiers if identifier]
        if not filtered_identifiers:
            return pd.DataFrame()
        urls = [
            url_template.format(identifier=identifier)
            for identifier in filtered_identifiers
        ]
        results = await self.fetch_multiple(client, urls)
        return self.prepare_materials_frame(
            results,
            identifier_column=identifier_column,
            extra_columns=extra_columns,
            drop_columns=drop_columns,
        )

    # -------------------------------------------------------------------------
    async def fetch_materials_data(
        self,
        client: httpx.AsyncClient,
        guest_index: pd.DataFrame,
        host_index: pd.DataFrame,
        guest_fraction: float,
        host_fraction: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        guest_data = await self.fetch_material_dataset(
            client=client,
            index=guest_index,
            identifier_column=self.guest_identifier,
            fraction=guest_fraction,
            url_template="https://adsorption.nist.gov/isodb/api/gas/{identifier}.json",
            extra_columns=self.extra_guest_columns,
            drop_columns=["synonyms"],
            label="guest",
        )
        host_data = await self.fetch_material_dataset(
            client=client,
            index=host_index,
            identifier_column=self.host_identifier,
            fraction=host_fraction,
            url_template="https://adsorption.nist.gov/matdb/api/material/{identifier}.json",
            extra_columns=self.extra_host_columns,
            drop_columns=["External_Resources", "synonyms"],
            label="host",
        )

        return guest_data, host_data


###############################################################################
class PubChemClient:
    def __init__(self, parallel_tasks: int) -> None:
        self.parallel_tasks = max(1, int(parallel_tasks))
        self.semaphore = asyncio.Semaphore(self.parallel_tasks)
        self.max_retries = 3
        self.base_retry_delay_seconds = 0.5
        # PubChem emits noisy not-found messages at INFO; keep library logger quieter.
        logging.getLogger("pubchempy").setLevel(logging.WARNING)

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_name(value: object) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except TypeError:
            pass
        return str(value).strip().lower()

    # -------------------------------------------------------------------------
    @staticmethod
    def is_not_found_error(message: str) -> bool:
        lowered = message.lower()
        return "notfound" in lowered or "no cid found" in lowered or "404" in lowered

    # -------------------------------------------------------------------------
    @staticmethod
    def is_retryable_error(message: str) -> bool:
        lowered = message.lower()
        retry_keywords = ("serverbusy", "too many requests", "503", "timeout")
        return any(keyword in lowered for keyword in retry_keywords)

    # -------------------------------------------------------------------------
    async def fetch_properties_for_name(self, name: str) -> dict[str, Any]:
        normalized_name = self.normalize_name(name)
        if not normalized_name:
            return {
                "name": normalized_name,
                "molecular_weight": None,
                "molecular_formula": None,
                "smile": None,
            }

        compounds: list[Any] = []
        for attempt in range(self.max_retries):
            async with self.semaphore:
                try:
                    compounds = await asyncio.to_thread(
                        pcp.get_compounds, normalized_name, "name"
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    message = str(exc)
                    is_retryable = self.is_retryable_error(message)
                    has_next_attempt = attempt < self.max_retries - 1
                    if is_retryable and has_next_attempt:
                        delay = self.base_retry_delay_seconds * (2**attempt)
                        await asyncio.sleep(delay)
                        continue
                    if self.is_not_found_error(message):
                        logger.info(
                            "PubChem compound not found for %s", normalized_name
                        )
                    else:
                        logger.warning(
                            "PubChem lookup failed for %s: %s", normalized_name, exc
                        )
                    compounds = []
                    break

        if not compounds:
            return {
                "name": normalized_name,
                "molecular_weight": None,
                "molecular_formula": None,
                "smile": None,
            }

        compound = compounds[0]
        return {
            "name": normalized_name,
            "molecular_weight": compound.molecular_weight,
            "molecular_formula": compound.molecular_formula,
            "smile": compound.smiles,
        }

    # -------------------------------------------------------------------------
    async def fetch_properties_for_names(
        self, names: list[str]
    ) -> list[dict[str, Any]]:
        if not names:
            return []
        tasks = [self.fetch_properties_for_name(name) for name in names]
        return await asyncio.gather(*tasks)


###############################################################################
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
        api_client = NISTApiClient(server_settings.nist.parallel_tasks)
        if category == "experiments":
            return await api_client.fetch_experiments_index(client), api_client.exp_identifier
        if category == "guest":
            return await api_client.fetch_guest_index(client), api_client.guest_identifier
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
        api_client = NISTApiClient(server_settings.nist.parallel_tasks)
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
        api_client = NISTApiClient(server_settings.nist.parallel_tasks)

        if job_id:
            job_manager.update_progress(job_id, 5.0)

        timeout = httpx.Timeout(30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            index, identifier_column = await self.load_index_for_category(category, client)
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
                experiments_data = await api_client.fetch_experiments_data_by_identifiers(
                    client, identifiers_to_fetch
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

        pubchem = PubChemClient(server_settings.nist.pubchem_parallel_tasks)
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
    async def enrich_guest_properties(self, job_id: str | None = None) -> dict[str, Any]:
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
