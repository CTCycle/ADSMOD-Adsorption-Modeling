from __future__ import annotations

import pandas as pd
from sqlalchemy import select

from shared.common.settings import DatabaseSettings
from shared.repositories.database.manager import DatabaseManager
from shared.repositories.queries.training import TrainingRepositoryQueries
from shared.repositories.schemas.models import Base, TrainingDataset, TrainingSample


def build_in_memory_database() -> DatabaseManager:
    settings = DatabaseSettings(
        embedded_database=True,
        engine=None,
        host=None,
        port=None,
        database_name=None,
        username=None,
        password=None,
        ssl=False,
        ssl_ca=None,
        connect_timeout=30,
        insert_batch_size=100,
        sqlite_path=":memory:",
    )
    manager = DatabaseManager(settings)
    Base.metadata.create_all(manager.engine)
    return manager


def test_training_dataset_metadata_round_trip_keeps_one_parent() -> None:
    manager = build_in_memory_database()
    try:
        queries = TrainingRepositoryQueries(database=manager)
        dataset = pd.DataFrame(
            [
                {
                    "name": "round_trip",
                    "split": "train",
                    "temperature": 298.15,
                    "pressure": [1.0, 2.0],
                    "adsorbed_amount": [0.1, 0.2],
                    "encoded_adsorbent": 1,
                    "adsorbate_molecular_weight": 44.0,
                    "adsorbate_encoded_smile": [1, 2],
                    "sample_key": "sample-1",
                },
                {
                    "name": "round_trip",
                    "split": "validation",
                    "temperature": 298.15,
                    "pressure": [3.0, 4.0],
                    "adsorbed_amount": [0.3, 0.4],
                    "encoded_adsorbent": 1,
                    "adsorbate_molecular_weight": 44.0,
                    "adsorbate_encoded_smile": [1, 2],
                    "sample_key": "sample-2",
                },
            ]
        )
        queries.upsert_training_dataset(dataset)

        metadata = pd.DataFrame(
            [
                {
                    "dataset_label": "round_trip",
                    "hashcode": "a" * 64,
                    "sample_size": 1.0,
                    "validation_size": 0.5,
                    "min_measurements": 1,
                    "max_measurements": 4,
                    "smile_sequence_size": 6,
                    "max_pressure": 4.0,
                    "max_uptake": 0.4,
                    "total_samples": 2,
                    "train_samples": 1,
                    "validation_samples": 1,
                    "test_samples": 0,
                    "smile_vocabulary": {"C": 1},
                    "adsorbent_vocabulary": {"carbon": 1},
                    "normalization_stats": {"pressure_mean": 2.0},
                }
            ]
        )
        queries.save_training_metadata(metadata)
        loaded_metadata = queries.load_training_metadata().iloc[0]
        queries.upsert_training_dataset(queries.load_training_dataset())

        with manager.session_factory() as session:
            parents = session.scalars(select(TrainingDataset)).all()
            samples = session.scalars(select(TrainingSample)).all()

        assert len(parents) == 1
        assert parents[0].content_hash == "a" * 64
        assert len(samples) == 2
        assert {sample.sample_key for sample in samples} == {"sample-1", "sample-2"}
        assert loaded_metadata["max_measurements"] == 4
        assert loaded_metadata["smile_sequence_size"] == 6
        assert loaded_metadata["max_pressure"] == 4.0
    finally:
        manager.dispose()
