from __future__ import annotations

import asyncio
import math
from typing import Any
from urllib.parse import quote

import httpx
import pandas as pd

from ADSMOD.server.utils.logger import logger
from ADSMOD.server.utils.repository.isodb import NISTDataSerializer


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
        dataframe["num_guests"] = dataframe["adsorbates"].str.len()
        single_component = dataframe[dataframe["num_guests"] == 1]
        binary_mixture = dataframe[dataframe["num_guests"] == 2]
        return single_component, binary_mixture

    # -------------------------------------------------------------------------
    def extract_nested_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if dataframe.empty:
            return dataframe

        dataframe["adsorbent_ID"] = dataframe["adsorbent"].apply(
            lambda x: (x or {}).get("hashkey")
        )
        dataframe["adsorbent_name"] = dataframe["adsorbent"].apply(
            lambda x: str((x or {}).get("name", "")).lower()
        )
        dataframe["adsorbates_ID"] = dataframe["adsorbates"].apply(
            lambda x: [item.get("InChIKey") for item in (x or []) if isinstance(item, dict)]
        )
        dataframe["adsorbate_name"] = dataframe["adsorbates"].apply(
            lambda x: [
                str(item.get("name", "")).lower()
                for item in (x or [])
                if isinstance(item, dict)
            ]
        )

        if (dataframe["num_guests"] == 1).all():
            dataframe["pressure"] = dataframe["isotherm_data"].apply(
                lambda x: [
                    item.get("pressure")
                    for item in (x or [])
                    if isinstance(item, dict)
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
                    str(x[0].get("name", "")).lower()
                    if isinstance(x, list) and x
                    else ""
                )
            )
            dataframe["composition"] = 1.0

        elif (dataframe["num_guests"] == 2).all():
            data_placeholder = {"composition": 1.0, "adsorption": 1.0}
            dataframe["total_pressure"] = dataframe["isotherm_data"].apply(
                lambda x: [
                    item.get("pressure")
                    for item in (x or [])
                    if isinstance(item, dict)
                ]
            )
            dataframe["all_species_data"] = dataframe["isotherm_data"].apply(
                lambda x: [
                    item.get("species_data")
                    for item in (x or [])
                    if isinstance(item, dict)
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
    async def fetch_experiments_index(
        self, client: httpx.AsyncClient
    ) -> pd.DataFrame:
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
    async def fetch_materials_data(
        self,
        client: httpx.AsyncClient,
        guest_index: pd.DataFrame,
        host_index: pd.DataFrame,
        guest_fraction: float,
        host_fraction: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        guest_data = pd.DataFrame()
        host_data = pd.DataFrame()

        if not guest_index.empty and guest_fraction > 0:
            if self.guest_identifier not in guest_index.columns:
                raise ValueError("NIST guest index missing InChIKey column.")
            guest_ids = guest_index[self.guest_identifier].tolist()
            guest_samples = int(math.ceil(guest_fraction * len(guest_ids)))
            guest_urls = [
                f"https://adsorption.nist.gov/isodb/api/gas/{identifier}.json"
                for identifier in guest_ids[:guest_samples]
            ]
            results = await self.fetch_multiple(client, guest_urls)
            guest_data = pd.DataFrame(results)
            if not guest_data.empty:
                guest_data = guest_data.drop(columns=["synonyms"], errors="ignore")
                guest_data["name"] = guest_data["name"].astype(str).str.lower()
                for col in self.extra_guest_columns:
                    guest_data[col] = pd.NA
            else:
                guest_data = pd.DataFrame(
                    columns=[self.guest_identifier, "name", *self.extra_guest_columns]
                )
        else:
            logger.warning("No guest index available for NIST fetch.")

        if not host_index.empty and host_fraction > 0:
            if self.host_identifier not in host_index.columns:
                raise ValueError("NIST host index missing hashkey column.")
            host_ids = host_index[self.host_identifier].tolist()
            host_samples = int(math.ceil(host_fraction * len(host_ids)))
            host_urls = [
                f"https://adsorption.nist.gov/matdb/api/material/{identifier}.json"
                for identifier in host_ids[:host_samples]
            ]
            results = await self.fetch_multiple(client, host_urls)
            host_data = pd.DataFrame(results)
            if not host_data.empty:
                host_data = host_data.drop(
                    columns=["External_Resources", "synonyms"], errors="ignore"
                )
                host_data["name"] = host_data["name"].astype(str).str.lower()
                for col in self.extra_host_columns:
                    host_data[col] = pd.NA
            else:
                host_data = pd.DataFrame(
                    columns=[self.host_identifier, "name", *self.extra_host_columns]
                )
        else:
            logger.warning("No host index available for NIST fetch.")

        return guest_data, host_data


###############################################################################
class PubChemClient:
    def __init__(self, parallel_tasks: int) -> None:
        self.parallel_tasks = max(1, int(parallel_tasks))
        self.semaphore = asyncio.Semaphore(self.parallel_tasks)
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    # -------------------------------------------------------------------------
    async def fetch_properties_for_name(
        self, client: httpx.AsyncClient, name: str
    ) -> dict[str, Any]:
        safe_name = quote(name)
        url = (
            f"{self.base_url}/compound/name/{safe_name}"
            "/property/MolecularWeight,MolecularFormula,IsomericSMILES,CanonicalSMILES/JSON"
        )
        async with self.semaphore:
            try:
                response = await client.get(url)
                response.raise_for_status()
            except httpx.HTTPError:
                return {
                    "name": name,
                    "molecular_weight": None,
                    "molecular_formula": None,
                    "smile": None,
                }
        try:
            payload = response.json()
        except ValueError:
            return {
                "name": name,
                "molecular_weight": None,
                "molecular_formula": None,
                "smile": None,
            }

        properties = payload.get("PropertyTable", {}).get("Properties", [])
        if not properties:
            return {
                "name": name,
                "molecular_weight": None,
                "molecular_formula": None,
                "smile": None,
            }
        entry = properties[0]
        return {
            "name": name,
            "molecular_weight": entry.get("MolecularWeight"),
            "molecular_formula": entry.get("MolecularFormula"),
            "smile": entry.get("IsomericSMILES") or entry.get("CanonicalSMILES"),
        }

    # -------------------------------------------------------------------------
    async def fetch_properties_for_names(
        self, client: httpx.AsyncClient, names: list[str]
    ) -> list[dict[str, Any]]:
        if not names:
            return []
        tasks = [self.fetch_properties_for_name(client, name) for name in names]
        return await asyncio.gather(*tasks)


###############################################################################
class NISTDataService:
    def __init__(self) -> None:
        self.serializer = NISTDataSerializer()
        self.builder = NISTDatasetBuilder()

    # -------------------------------------------------------------------------
    async def fetch_and_store(
        self,
        dataset_name: str,
        experiments_fraction: float,
        guest_fraction: float,
        host_fraction: float,
        parallel_tasks: int,
    ) -> dict[str, int]:
        dataset_name = dataset_name.strip()
        if not dataset_name:
            raise ValueError("Dataset name cannot be empty.")
        api_client = NISTApiClient(parallel_tasks)
        timeout = httpx.Timeout(30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            experiments_index = await api_client.fetch_experiments_index(client)
            experiments_data = await api_client.fetch_experiments_data(
                client, experiments_index, experiments_fraction
            )
            guest_index, host_index = await api_client.fetch_materials_index(client)
            guest_data, host_data = await api_client.fetch_materials_data(
                client, guest_index, host_index, guest_fraction, host_fraction
            )

        single_component, binary_mixture = self.builder.build_datasets(experiments_data)
        if not single_component.empty:
            single_component["dataset_name"] = dataset_name
        if not binary_mixture.empty:
            binary_mixture["dataset_name"] = dataset_name

        await asyncio.to_thread(
            self.serializer.save_adsorption_datasets, single_component, binary_mixture
        )
        await asyncio.to_thread(
            self.serializer.save_materials_datasets, guest_data, host_data
        )

        return {
            "experiments_count": len(experiments_data),
            "single_component_rows": len(single_component),
            "binary_mixture_rows": len(binary_mixture),
            "guest_rows": len(guest_data),
            "host_rows": len(host_data),
        }

    # -------------------------------------------------------------------------
    async def enrich_properties(
        self, target: str, parallel_tasks: int
    ) -> dict[str, int]:
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
            return {
                "names_requested": len(names),
                "names_matched": 0,
                "rows_updated": 0,
            }

        pubchem = PubChemClient(parallel_tasks)
        timeout = httpx.Timeout(30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            properties = await pubchem.fetch_properties_for_names(client, names)

        properties_frame = pd.DataFrame(properties)
        if properties_frame.empty:
            return {
                "names_requested": len(names),
                "names_matched": 0,
                "rows_updated": 0,
            }

        weight_col = f"{prefix}_molecular_weight"
        formula_col = f"{prefix}_molecular_formula"
        smile_col = f"{prefix}_SMILE"

        data["name"] = data["name"].astype(str).str.lower()
        properties_frame["name"] = properties_frame["name"].astype(str).str.lower()

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

        return {
            "names_requested": len(names),
            "names_matched": matched,
            "rows_updated": rows_updated,
        }
