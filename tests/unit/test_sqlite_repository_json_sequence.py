import numpy as np
import pandas as pd

from ADSMOD.server.repositories.database.postgres import PostgresRepository
from ADSMOD.server.repositories.database.sqlite import SQLiteRepository
from ADSMOD.server.repositories.schemas.models import TrainingDataset


def test_sqlite_prepare_for_storage_parses_json_sequence_strings() -> None:
    repository = SQLiteRepository.__new__(SQLiteRepository)
    raw = pd.DataFrame(
        [
            {
                "pressure": "[1.0, 2.0, 3.0]",
                "adsorbed_amount": "[0.1, 0.2, 0.3]",
                "adsorbate_encoded_smile": "[6.0, 4.0, 5.0]",
            }
        ]
    )

    prepared = repository.prepare_for_storage(raw, TrainingDataset)

    assert prepared.loc[0, "pressure"] == [1.0, 2.0, 3.0]
    assert prepared.loc[0, "adsorbed_amount"] == [0.1, 0.2, 0.3]
    assert prepared.loc[0, "adsorbate_encoded_smile"] == [6.0, 4.0, 5.0]


def test_sqlite_restore_after_load_decodes_double_encoded_sequences() -> None:
    repository = SQLiteRepository.__new__(SQLiteRepository)
    loaded = pd.DataFrame(
        [
            {
                "pressure": '"[1.0, 2.0, 3.0]"',
                "adsorbed_amount": '"[0.1, 0.2, 0.3]"',
                "adsorbate_encoded_smile": '"[6.0, 4.0, 5.0]"',
            }
        ]
    )

    restored = repository.restore_after_load(loaded, TrainingDataset)

    assert restored.loc[0, "pressure"] == [1.0, 2.0, 3.0]
    assert restored.loc[0, "adsorbed_amount"] == [0.1, 0.2, 0.3]
    assert restored.loc[0, "adsorbate_encoded_smile"] == [6.0, 4.0, 5.0]

    converted = np.asarray(restored.loc[0, "adsorbate_encoded_smile"], dtype=np.int32)
    assert converted.tolist() == [6, 4, 5]


def test_sqlite_and_postgres_json_value_parsing_match_for_sequences() -> None:
    sqlite_value = SQLiteRepository.parse_json_column_value('"[6.0, 4.0, 5.0]"')
    postgres_value = PostgresRepository.parse_json_column_value("[6.0, 4.0, 5.0]")

    assert sqlite_value == postgres_value
    assert sqlite_value == [6.0, 4.0, 5.0]
