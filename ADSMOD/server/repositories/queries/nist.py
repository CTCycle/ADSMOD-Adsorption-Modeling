from __future__ import annotations

import pandas as pd

from ADSMOD.server.common.constants import COLUMN_ADSORBATE, COLUMN_ADSORBENT
from ADSMOD.server.common.utils.logger import logger
from ADSMOD.server.repositories.database.backend import database


###############################################################################
class NISTDataSerializer:
    SINGLE_COMPONENT_UNIQUE_COLUMNS = [
        COLUMN_ADSORBATE,
        COLUMN_ADSORBENT,
        "temperature",
        "pressure",
        "adsorbed_amount",
    ]

    # -------------------------------------------------------------------------
    @classmethod
    def deduplicate_single_component_rows(
        cls, single_component: pd.DataFrame
    ) -> pd.DataFrame:
        if single_component.empty:
            return single_component

        normalized = single_component.copy()
        for column in (COLUMN_ADSORBATE, COLUMN_ADSORBENT):
            if column in normalized.columns:
                normalized[column] = (
                    normalized[column].astype("string").str.strip().str.lower()
                )
        for column in ("temperature", "pressure", "adsorbed_amount"):
            if column in normalized.columns:
                normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

        missing_columns = [
            column
            for column in cls.SINGLE_COMPONENT_UNIQUE_COLUMNS
            if column not in normalized.columns
        ]
        if missing_columns:
            logger.warning(
                "Skipping NIST single-component deduplication: missing columns %s",
                missing_columns,
            )
            return normalized

        before_count = len(normalized)
        deduplicated = normalized.drop_duplicates(
            subset=cls.SINGLE_COMPONENT_UNIQUE_COLUMNS,
            keep="first",
        ).reset_index(drop=True)
        removed_count = before_count - len(deduplicated)
        if removed_count > 0:
            logger.info(
                "Removed %d duplicate NIST single-component rows before upsert",
                removed_count,
            )
        return deduplicated

    # -------------------------------------------------------------------------
    def load_adsorption_datasets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        adsorption_data = database.load_from_database("nist_single_component_adsorption")
        guest_data = database.load_from_database("adsorbates")
        host_data = database.load_from_database("adsorbents")

        return adsorption_data, guest_data, host_data

    # -------------------------------------------------------------------------
    def count_nist_rows(self) -> dict[str, int]:
        return {
            "single_component_rows": database.count_rows(
                "nist_single_component_adsorption"
            ),
            "binary_mixture_rows": database.count_rows(
                "nist_binary_mixture_adsorption"
            ),
            "guest_rows": database.count_rows("adsorbates"),
            "host_rows": database.count_rows("adsorbents"),
        }

    # -------------------------------------------------------------------------
    def save_materials_datasets(
        self,
        guest_data: pd.DataFrame | None = None,
        host_data: pd.DataFrame | None = None,
    ) -> None:
        if isinstance(guest_data, pd.DataFrame):
            database.upsert_into_database(guest_data, "adsorbates")
        if isinstance(host_data, pd.DataFrame):
            database.upsert_into_database(host_data, "adsorbents")

    # -------------------------------------------------------------------------
    def save_adsorption_datasets(
        self, single_component: pd.DataFrame, binary_mixture: pd.DataFrame
    ) -> None:
        if isinstance(single_component, pd.DataFrame):
            single_component = self.deduplicate_single_component_rows(single_component)
            database.upsert_into_database(
                single_component, "nist_single_component_adsorption"
            )
        if isinstance(binary_mixture, pd.DataFrame):
            database.upsert_into_database(
                binary_mixture, "nist_binary_mixture_adsorption"
            )
