# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit

import httpx
import pandas as pd

from server.common.utils.encoding import (
    decode_json_response_bytes,
    sanitize_dataframe_strings,
)
from server.common.utils.logger import logger

###############################################################################
class NISTDatasetBuilder:

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    def __init__(
        self,
        parallel_tasks: int,
        *,
        retry_attempts: int = 3,
        retry_backoff_max_seconds: float = 30.0,
        max_response_bytes: int = 16_777_216,
    ) -> None:
        self.parallel_tasks = max(1, int(parallel_tasks))
        self.semaphore = asyncio.Semaphore(self.parallel_tasks)
        self.max_attempts = max(1, int(retry_attempts))
        self.retry_backoff_max_seconds = max(0.1, float(retry_backoff_max_seconds))
        self.max_response_bytes = max(1024, int(max_response_bytes))
        self.last_failure: str | None = None
        self._request_stats = {
            "requests": 0,
            "retries": 0,
            "rate_limits": 0,
            "failures": 0,
        }
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
        parsed_url = urlsplit(url)
        if parsed_url.scheme != "https" or parsed_url.hostname != "adsorption.nist.gov":
            self._record_failure("the request target is outside the NIST HTTPS host")
            return None
        content: bytes | None = None
        for attempt in range(self.max_attempts):
            try:
                content, retry_delay = await self._fetch_attempt(
                    client, url, attempt
                )
                if retry_delay is not None:
                    await asyncio.sleep(retry_delay)
                    continue
                if content is None:
                    return None
                return self._decode_json(content)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt + 1 >= self.max_attempts:
                    self._record_failure(f"the request failed after retries ({exc})")
                    return None
                retry_delay = self._retry_delay(None, attempt)
                self._request_stats["retries"] += 1
                logger.warning(
                    "NIST transport request failed; retrying attempt %d/%d after %.2fs: %s",
                    attempt + 2,
                    self.max_attempts,
                    retry_delay,
                    exc,
                )
                await asyncio.sleep(retry_delay)

        return None

    # -------------------------------------------------------------------------
    async def _fetch_attempt(
        self, client: httpx.AsyncClient, url: str, attempt: int
    ) -> tuple[bytes | None, float | None]:
        self._request_stats["requests"] += 1
        async with self.semaphore:
            async with client.stream("GET", url) as response:
                status_code = response.status_code
                if status_code in {429, 500, 502, 503, 504}:
                    return self._retryable_response(response, attempt)
                if status_code >= 400 or 300 <= status_code < 400:
                    self._record_failure(f"NIST returned HTTP {status_code}")
                    return None, None
                content = await self._read_bounded_body(response)
                if content is None:
                    self._record_failure(
                        "the response exceeded the configured size limit of "
                        f"{self.max_response_bytes} bytes"
                    )
                return content, None

    # -------------------------------------------------------------------------
    def _retryable_response(
        self, response: httpx.Response, attempt: int
    ) -> tuple[None, float | None]:
        status_code = response.status_code
        if status_code == 429:
            self._request_stats["rate_limits"] += 1
        if attempt + 1 >= self.max_attempts:
            self._record_failure(f"NIST returned HTTP {status_code} after retries")
            return None, None
        delay = self._retry_delay(response.headers.get("Retry-After"), attempt)
        self._request_stats["retries"] += 1
        logger.warning(
            "NIST request returned HTTP %d; retrying attempt %d/%d after %.2fs",
            status_code,
            attempt + 2,
            self.max_attempts,
            delay,
        )
        return None, delay

    # -------------------------------------------------------------------------
    async def _read_bounded_body(self, response: httpx.Response) -> bytes | None:
        length = response.headers.get("Content-Length")
        if length and length.isdigit() and int(length) > self.max_response_bytes:
            return None
        body = bytearray()
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body) > self.max_response_bytes:
                return None
        return bytes(body)

    # -------------------------------------------------------------------------
    def _decode_json(self, content: bytes) -> dict[str, Any] | list[Any] | None:
        try:
            return decode_json_response_bytes(content)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self._record_failure(f"NIST returned malformed JSON ({exc})")
            return None

    # -------------------------------------------------------------------------
    def _record_failure(self, reason: str) -> None:
        logger.warning("NIST request failed: %s.", reason)
        self.last_failure = reason
        self._request_stats["failures"] += 1

    # -------------------------------------------------------------------------
    def request_statistics(self) -> dict[str, int]:
        return dict(self._request_stats)

    # -------------------------------------------------------------------------
    def _retry_delay(self, retry_after: str | None, attempt: int) -> float:
        delay: float | None = None
        if retry_after:
            try:
                delay = max(0.0, float(retry_after.strip()))
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=timezone.utc)
                    delay = max(
                        0.0,
                        (retry_at - datetime.now(timezone.utc)).total_seconds(),
                    )
                except (TypeError, ValueError, OverflowError):
                    delay = None
        if delay is None:
            delay = 0.5 * (2**attempt)
        return min(delay, self.retry_backoff_max_seconds)

    # -------------------------------------------------------------------------
    async def fetch_experiments_index(self, client: httpx.AsyncClient) -> pd.DataFrame:
        return await self._fetch_index(client, self.url_isotherms, "adsorption isotherm")

    # -------------------------------------------------------------------------
    async def _fetch_index(
        self, client: httpx.AsyncClient, url: str, label: str
    ) -> pd.DataFrame:
        payload = await self.fetch_json(client, url)
        if payload is None:
            raise ValueError(
                f"Failed to retrieve the NIST {label} index: "
                f"{self.last_failure or 'no response'}."
            )
        return pd.DataFrame(payload)

    # -------------------------------------------------------------------------
    async def fetch_guest_index(self, client: httpx.AsyncClient) -> pd.DataFrame:
        return await self._fetch_index(client, self.url_guest_index, "adsorbate species")

    # -------------------------------------------------------------------------
    async def fetch_host_index(self, client: httpx.AsyncClient) -> pd.DataFrame:
        return await self._fetch_index(client, self.url_host_index, "adsorbent material")

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
        data["name"] = data["name"].astype(str).str.strip()
        for col in extra_columns:
            data[col] = pd.NA
        # NIST catalogue records publish their own formula and InChI string.
        if "formula" in data.columns:
            data["molecular_formula"] = data["formula"].replace("", pd.NA)
        if "InChICode" in data.columns:
            data["inchi"] = data["InChICode"].replace("", pd.NA)
        return data



__all__ = [
    "NISTDatasetBuilder",
    "NISTApiClient",
]
