# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
import json
import random
import weakref
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

_REQUEST_BUDGETS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, asyncio.Semaphore
] = weakref.WeakKeyDictionary()


###############################################################################
def _shared_request_budget(limit: int) -> asyncio.Semaphore:
    """One NIST request budget per event loop, shared by every client and job."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.Semaphore(limit)
    budget = _REQUEST_BUDGETS.get(loop)
    if budget is None:
        budget = _REQUEST_BUDGETS[loop] = asyncio.Semaphore(limit)
    return budget


###############################################################################
class NISTDatasetBuilder:
    """Flatten NIST ISODB isotherm payloads into one long measurement frame.

    Every experiment, whatever its number of species, yields one row per
    (equilibrium point, species): ``pressure`` is the point's total pressure,
    ``adsorbed_amount`` the species uptake, and ``gas_mole_fraction`` the
    species gas-phase composition (empty for single-species experiments).
    """

    columns = [
        "name",
        "adsorbent",
        "adsorbate",
        "adsorbate_inchi_key",
        "point_index",
        "pressure",
        "adsorbed_amount",
        "gas_mole_fraction",
        "pressure_units",
        "adsorption_units",
        "temperature",
        "composition_type",
    ]
    required = [
        "name",
        "pressure",
        "adsorbed_amount",
        "pressure_units",
        "adsorption_units",
        "temperature",
        "adsorbent",
        "adsorbate",
    ]

    # -------------------------------------------------------------------------
    @staticmethod
    def _species(payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            item for item in payload.get("adsorbates") or [] if isinstance(item, dict)
        ]

    # -------------------------------------------------------------------------
    @staticmethod
    def _species_entry(
        entries: list[Any], species: dict[str, Any], position: int
    ) -> dict[str, Any]:
        """Match a point's species datum by InChIKey, falling back to position.

        A missing datum stays empty so the incomplete row is rejected later,
        never replaced by an invented measurement.
        """
        valid = [item for item in entries if isinstance(item, dict)]
        key = species.get("InChIKey")
        if key:
            for item in valid:
                if item.get("InChIKey") == key:
                    return item
            if any(item.get("InChIKey") for item in valid):
                return {}
        return valid[position] if position < len(valid) else {}

    # -------------------------------------------------------------------------
    def _rows(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        species = self._species(payload)
        if not species:
            return []
        adsorbent = payload.get("adsorbent") or {}
        base = {
            "name": payload.get("filename"),
            "adsorbent": str(adsorbent.get("name", "")).lower()
            if isinstance(adsorbent, dict)
            else "",
            "pressure_units": payload.get("pressureUnits"),
            "adsorption_units": payload.get("adsorptionUnits"),
            "temperature": payload.get("temperature"),
            "composition_type": payload.get("compositionType"),
        }
        rows: list[dict[str, Any]] = []
        points = [
            item
            for item in payload.get("isotherm_data") or []
            if isinstance(item, dict)
        ]
        for point_index, point in enumerate(points):
            for position, item in enumerate(species):
                if len(species) == 1:
                    uptake, fraction = point.get("total_adsorption"), None
                else:
                    datum = self._species_entry(
                        point.get("species_data") or [], item, position
                    )
                    uptake, fraction = datum.get("adsorption"), datum.get("composition")
                rows.append(
                    {
                        **base,
                        "adsorbate": str(item.get("name", "")).lower(),
                        "adsorbate_inchi_key": item.get("InChIKey"),
                        "point_index": point_index,
                        "pressure": point.get("pressure"),
                        "adsorbed_amount": uptake,
                        "gas_mole_fraction": fraction,
                    }
                )
        return rows

    # -------------------------------------------------------------------------
    def _drop_incomplete(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Drop only rows missing a field the measurement cannot exist without."""
        incomplete = dataframe[self.required].isna().any(axis=1) | (
            dataframe[["name", "adsorbent", "adsorbate"]]
            .astype(str)
            .apply(lambda column: column.str.strip() == "")
            .any(axis=1)
        )
        if incomplete.any():
            logger.warning(
                "Rejected %d NIST measurement row(s) missing required fields %s.",
                int(incomplete.sum()),
                self.required,
            )
        return dataframe.loc[~incomplete].reset_index(drop=True)

    # -------------------------------------------------------------------------
    def build_dataset(self, adsorption_data: pd.DataFrame | None) -> pd.DataFrame:
        if adsorption_data is None or adsorption_data.empty:
            return pd.DataFrame(columns=self.columns)
        rows = [
            row
            for payload in adsorption_data.to_dict("records")
            for row in self._rows(payload)
        ]
        if not rows:
            return pd.DataFrame(columns=self.columns)
        dataframe = pd.DataFrame(rows, columns=self.columns)
        return sanitize_dataframe_strings(self._drop_incomplete(dataframe))


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
        self.semaphore = _shared_request_budget(self.parallel_tasks)
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
                content, retry_delay = await self._fetch_attempt(client, url, attempt)
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
                except TypeError, ValueError, OverflowError:
                    delay = None
        if delay is None:
            # Jitter desynchronises concurrent retries; a server-provided
            # Retry-After is honoured exactly.
            delay = 0.5 * (2**attempt) * random.uniform(0.75, 1.25)
        return min(delay, self.retry_backoff_max_seconds)

    # -------------------------------------------------------------------------
    async def fetch_experiments_index(self, client: httpx.AsyncClient) -> pd.DataFrame:
        return await self._fetch_index(
            client, self.url_isotherms, "adsorption isotherm"
        )

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
        return await self._fetch_index(
            client, self.url_guest_index, "adsorbate species"
        )

    # -------------------------------------------------------------------------
    async def fetch_host_index(self, client: httpx.AsyncClient) -> pd.DataFrame:
        return await self._fetch_index(
            client, self.url_host_index, "adsorbent material"
        )

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
