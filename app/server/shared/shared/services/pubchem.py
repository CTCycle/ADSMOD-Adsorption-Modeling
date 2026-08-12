"""Shared PubChem enrichment client used by NIST and training workflows."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pandas as pd
import pubchempy as pcp

from shared.common.utils.logger import logger

###############################################################################
class PubChemClient:

    # -------------------------------------------------------------------------
    def __init__(self, parallel_tasks: int) -> None:
        self.parallel_tasks = max(1, int(parallel_tasks))
        self.semaphore = asyncio.Semaphore(self.parallel_tasks)
        self.max_retries = 3
        self.base_retry_delay_seconds = 0.5
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
                        logger.info("PubChem compound not found for %s", normalized_name)
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


__all__ = ["PubChemClient"]
