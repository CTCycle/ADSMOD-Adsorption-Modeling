from __future__ import annotations

import pandas as pd

from shared.common.settings import get_server_settings
from shared.repositories.database.manager import DatabaseManager
from shared.repositories.database.sqlite import SQLiteRepository
from shared.repositories.queries.nist import NISTDataSerializer
from shared.repositories.schemas.models import Dataset


class DataSerializer:
    """Canonical DataFrame boundary for training readers.

    Uploaded adsorption data is represented by the canonical datasets/isotherms/
    observations aggregate; no processed-isotherm or fit serializer remains.
    """

    def __init__(self, database: DatabaseManager | None = None) -> None:
        self.database = database or DatabaseManager(get_server_settings().database, create_schema=True)
        self.nist = NISTDataSerializer(self.database)

    def load_table(self, table_name: str) -> pd.DataFrame:
        if table_name != "adsorption_data":
            return pd.DataFrame()
        from shared.repositories.schemas.models import Adsorbate, Adsorbent, Isotherm, IsothermComponent, Observation
        from sqlalchemy import select
        with self.database.session_factory() as session:
            rows = session.execute(select(Dataset.name.label("name"), Isotherm.external_key.label("experiment"), Isotherm.temperature_k.label("temperature"), Adsorbent.name.label("adsorbent_name"), Adsorbate.name.label("adsorbate_name"), Observation.pressure_canonical.label("pressure"), Observation.uptake_mol_kg.label("adsorbed_amount")).join(Isotherm, Isotherm.dataset_id == Dataset.id).join(Adsorbent, Adsorbent.id == Isotherm.adsorbent_id).join(Observation, Observation.isotherm_id == Isotherm.id).join(IsothermComponent, IsothermComponent.id == Observation.component_id).join(Adsorbate, Adsorbate.id == IsothermComponent.adsorbate_id)).mappings()
            return pd.DataFrame([dict(row) for row in rows])

    def save_raw_dataset(self, dataframe: pd.DataFrame) -> None:
        # Training ingestion now writes through its canonical repository; raw
        # adsorption uploads are committed by the core dataset service.
        return None

    def delete_raw_dataset(self, _dataset_name: str) -> bool:
        return False
