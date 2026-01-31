from __future__ import annotations

import asyncio
import math
from time import monotonic
from typing import Any

import httpx
import pandas as pd
import pubchempy as pcp

from ADSMOD.server.configurations import server_settings
from ADSMOD.server.utils.logger import logger
from ADSMOD.server.repository.isodb import NISTDataSerializer
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
    def drop_excluded_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.drop(columns=self.raw_drop_cols, axis=1, errors="ignore")

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
        dataframe["composition"] = 1.0
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
                columns=self.single_drop_cols, axis=1, errors="ignore"
            )
            single_dataset = single_dataset.dropna().reset_index(drop=True)

        if binary_mixture.empty:
            binary_dataset = pd.DataFrame()
        else:
            binary_dataset = binary_mixture.explode(self.binary_explode_cols)
            binary_dataset = binary_dataset.drop(
                columns=self.binary_drop_cols, axis=1, errors="ignore"
            )
            binary_dataset = binary_dataset.dropna().reset_index(drop=True)

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
            "adsorbate_molecular_weight",
            "adsorbate_molecular_formula",
            "adsorbate_SMILE",
        ]
        self.extra_host_columns = [
            "adsorbent_molecular_weight",
            "adsorbent_molecular_formula",
            "adsorbent_SMILE",
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
                return response.json()
            except ValueError as exc:
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
    async def fetch_materials_index(
        self, client: httpx.AsyncClient
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        guest_payload = await self.fetch_json(client, self.url_guest_index)
        host_payload = await self.fetch_json(client, self.url_host_index)
        guest_data = pd.DataFrame(guest_payload or [])
        host_data = pd.DataFrame(host_payload or [])
        return guest_data, host_data

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

    # -------------------------------------------------------------------------
    async def fetch_properties_for_name(self, name: str) -> dict[str, Any]:
        if not name:
            return {
                "name": name,
                "molecular_weight": None,
                "molecular_formula": None,
                "smile": None,
            }

        async with self.semaphore:
            try:
                compounds = await asyncio.to_thread(pcp.get_compounds, name, "name")
            except Exception as exc:  # noqa: BLE001
                logger.warning("PubChem lookup failed for %s: %s", name, exc)
                compounds = []

        if not compounds:
            return {
                "name": name,
                "molecular_weight": None,
                "molecular_formula": None,
                "smile": None,
            }

        compound = compounds[0]
        return {
            "name": name,
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
        start_time = monotonic()
        api_client = NISTApiClient(server_settings.nist.parallel_tasks)
        timeout = httpx.Timeout(30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            experiments_index = await api_client.fetch_experiments_index(client)
            experiments_total = len(experiments_index)
            experiments_requested = 0
            if experiments_total > 0 and experiments_fraction > 0:
                experiments_requested = int(
                    math.ceil(experiments_fraction * experiments_total)
                )
            logger.info(
                "NIST experiments index loaded (total=%d, requested=%d)",
                experiments_total,
                experiments_requested,
            )
            experiments_data = await api_client.fetch_experiments_data(
                client, experiments_index, experiments_fraction
            )
            if job_id:
                job_manager.update_progress(job_id, 30.0)

            guest_index, host_index = await api_client.fetch_materials_index(client)
            guest_total = len(guest_index)
            host_total = len(host_index)
            guest_requested = 0
            host_requested = 0
            if guest_total > 0 and guest_fraction > 0:
                guest_requested = int(math.ceil(guest_fraction * guest_total))
            if host_total > 0 and host_fraction > 0:
                host_requested = int(math.ceil(host_fraction * host_total))
            logger.info(
                "NIST materials index loaded (guests=%d requested=%d, hosts=%d requested=%d)",
                guest_total,
                guest_requested,
                host_total,
                host_requested,
            )
            guest_data, host_data = await api_client.fetch_materials_data(
                client, guest_index, host_index, guest_fraction, host_fraction
            )
            if job_id:
                job_manager.update_progress(job_id, 60.0)

        logger.info(
            "NIST payload fetched (experiments=%d, guests=%d, hosts=%d)",
            len(experiments_data),
            len(guest_data),
            len(host_data),
        )
        single_component, binary_mixture = self.builder.build_datasets(experiments_data)
        logger.info(
            "NIST datasets built (single_component_rows=%d, binary_mixture_rows=%d)",
            len(single_component),
            len(binary_mixture),
        )
        if job_id:
            job_manager.update_progress(job_id, 80.0)

        await asyncio.to_thread(
            self.serializer.save_adsorption_datasets, single_component, binary_mixture
        )
        await asyncio.to_thread(
            self.serializer.save_materials_datasets, guest_data, host_data
        )
        elapsed = monotonic() - start_time
        logger.info(
            "NIST fetch stored (experiments=%d, single_component_rows=%d, binary_mixture_rows=%d, guests=%d, hosts=%d, elapsed_s=%.2f)",
            len(experiments_data),
            len(single_component),
            len(binary_mixture),
            len(guest_data),
            len(host_data),
            elapsed,
        )

        return {
            "experiments_count": len(experiments_data),
            "single_component_rows": len(single_component),
            "binary_mixture_rows": len(binary_mixture),
            "guest_rows": len(guest_data),
            "host_rows": len(host_data),
        }

    # -------------------------------------------------------------------------
    async def enrich_properties(self, target: str, job_id: str | None = None) -> dict[str, int]:
        logger.info("NIST properties enrichment starting (target=%s)", target)
        adsorption_data, guest_data, host_data = await asyncio.to_thread(
            self.serializer.load_adsorption_datasets
        )

        if target == "guest":
            data = guest_data.copy()
            name_series = pd.concat(
                [
                    adsorption_data.get("adsorbate_name", pd.Series(dtype=str)),
                    data.get("name", pd.Series(dtype=str)),
                ]
            )
            prefix = "adsorbate"
        elif target == "host":
            data = host_data.copy()
            name_series = pd.concat(
                [
                    adsorption_data.get("adsorbent_name", pd.Series(dtype=str)),
                    data.get("name", pd.Series(dtype=str)),
                ]
            )
            prefix = "adsorbent"
        else:
            raise ValueError("Target must be 'guest' or 'host'.")

        weight_col = f"{prefix}_molecular_weight"
        formula_col = f"{prefix}_molecular_formula"
        smile_col = f"{prefix}_SMILE"

        names = (
            name_series.dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .loc[lambda series: series != ""]
            .unique()
            .tolist()
        )

        if not names or data.empty:
            logger.info(
                "NIST properties enrichment skipped (target=%s, names=%d, data_empty=%s)",
                target,
                len(names),
                data.empty,
            )
            return {
                "names_requested": len(names),
                "names_matched": 0,
                "rows_updated": 0,
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
                "names_requested": len(names),
                "names_matched": 0,
                "rows_updated": 0,
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

        return {
            "names_requested": len(names),
            "names_matched": matched,
            "rows_updated": rows_updated,
        }

    # -------------------------------------------------------------------------
    async def get_status(self) -> dict[str, int | bool]:
        counts = await asyncio.to_thread(self.serializer.count_nist_rows)
        data_available = any(value > 0 for value in counts.values())
        return {
            "data_available": data_available,
            **counts,
        }
