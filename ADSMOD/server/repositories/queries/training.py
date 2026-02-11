from __future__ import annotations

import pandas as pd

from ADSMOD.server.repositories.database.backend import ADSMODDatabase, database


###############################################################################
class TrainingRepositoryQueries:
    def __init__(self, db: ADSMODDatabase = database) -> None:
        self.database = db

    # -------------------------------------------------------------------------
    def load_training_dataset(self, limit: int | None = None) -> pd.DataFrame:
        return self.database.load_from_database(
            "training_dataset",
            limit=limit,
        )

    # -------------------------------------------------------------------------
    def save_training_dataset(self, dataset: pd.DataFrame) -> None:
        self.database.save_into_database(dataset, "training_dataset")

    # -------------------------------------------------------------------------
    def upsert_training_dataset(self, dataset: pd.DataFrame) -> None:
        self.database.upsert_into_database(dataset, "training_dataset")

    # -------------------------------------------------------------------------
    def load_training_metadata(self) -> pd.DataFrame:
        return self.database.load_from_database("training_metadata")

    # -------------------------------------------------------------------------
    def save_training_metadata(self, metadata: pd.DataFrame) -> None:
        self.database.save_into_database(metadata, "training_metadata")
