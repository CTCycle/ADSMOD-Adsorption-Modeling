from __future__ import annotations

import pandas as pd

from ADSMOD.server.database.database import database


###############################################################################
class NISTDataSerializer:
    
    # -------------------------------------------------------------------------
    def load_adsorption_datasets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        adsorption_data = database.load_from_database("SINGLE_COMPONENT_ADSORPTION")
        guest_data = database.load_from_database("ADSORBATES")
        host_data = database.load_from_database("ADSORBENTS")

        return adsorption_data, guest_data, host_data

    # -------------------------------------------------------------------------
    def save_materials_datasets(
        self,
        guest_data: pd.DataFrame | None = None,
        host_data: pd.DataFrame | None = None,
    ) -> None:
        if isinstance(guest_data, pd.DataFrame):
            database.upsert_into_database(guest_data, "ADSORBATES")
        if isinstance(host_data, pd.DataFrame):
            database.upsert_into_database(host_data, "ADSORBENTS")

    # -------------------------------------------------------------------------
    def save_adsorption_datasets(
        self, single_component: pd.DataFrame, binary_mixture: pd.DataFrame
    ) -> None:
        if isinstance(single_component, pd.DataFrame):
            database.upsert_into_database(
                single_component, "SINGLE_COMPONENT_ADSORPTION"
            )
        if isinstance(binary_mixture, pd.DataFrame):
            database.upsert_into_database(
                binary_mixture, "BINARY_MIXTURE_ADSORPTION"
            )
