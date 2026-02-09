from __future__ import annotations

import pandas as pd

from ADSMOD.server.repositories.database.backend import database


###############################################################################
class NISTDataSerializer:
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
            database.upsert_into_database(
                single_component, "nist_single_component_adsorption"
            )
        if isinstance(binary_mixture, pd.DataFrame):
            database.upsert_into_database(
                binary_mixture, "nist_binary_mixture_adsorption"
            )
