from __future__ import annotations

import pandas as pd

from ADSMOD.server.repositories.database.backend import ADSMODDatabase, database


###############################################################################
class DataRepositoryQueries:
    def __init__(self, db: ADSMODDatabase = database) -> None:
        self.database = db

    # -------------------------------------------------------------------------
    def load_table(
        self,
        table_name: str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> pd.DataFrame:
        return self.database.load_from_database(table_name, limit=limit, offset=offset)

    # -------------------------------------------------------------------------
    def save_table(self, dataset: pd.DataFrame, table_name: str) -> None:
        self.database.save_into_database(dataset, table_name)

    # -------------------------------------------------------------------------
    def upsert_table(self, dataset: pd.DataFrame, table_name: str) -> None:
        self.database.upsert_into_database(dataset, table_name)

