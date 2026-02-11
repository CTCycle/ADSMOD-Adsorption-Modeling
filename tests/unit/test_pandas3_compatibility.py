from __future__ import annotations

import pandas as pd
from pandas.api.types import is_string_dtype

from ADSMOD.server.common.utils.encoding import sanitize_dataframe_strings
from ADSMOD.server.repositories.database.postgres import PostgresRepository
from ADSMOD.server.services.data.conversion import PressureConversion
from ADSMOD.server.services.data.sanitizer import DataSanitizer


def test_sanitize_dataframe_strings_handles_pandas_string_dtype() -> None:
    frame = pd.DataFrame({"name": ["zeolite\u200b-a", "na\u00a0y"]}).astype("string")

    sanitized = sanitize_dataframe_strings(frame)

    assert is_string_dtype(sanitized["name"])
    assert sanitized.loc[0, "name"] == "zeolite-a"
    assert sanitized.loc[1, "name"] == "na y"


def test_pressure_conversion_returns_frame_without_unit_column() -> None:
    converter = PressureConversion()
    frame = pd.DataFrame({"pressure": [1.0], "pressure_units": ["bar"]})

    converted = converter.convert_pressure_units(frame)

    assert "pressure_units" not in converted.columns
    assert converted.loc[0, "pressure"] == 100000.0


def test_exclude_oob_values_uses_copy_safe_assignment() -> None:
    sanitizer = DataSanitizer({"max_pressure": 10, "max_uptake": 20})
    frame = pd.DataFrame(
        {
            "temperature": [300, -1],
            "pressure": [[0.0, 12_000_000.0, 5.0], [1.0]],
            "adsorbed_amount": [[1.0, 2.0, 30.0], [1.0]],
        }
    )

    filtered = sanitizer.exclude_OOB_values(frame)

    assert len(filtered) == 1
    assert filtered.iloc[0]["pressure"] == [0.0]
    assert filtered.iloc[0]["adsorbed_amount"] == [1.0]
    assert frame.loc[0, "pressure"] == [0.0, 12_000_000.0, 5.0]


def test_parse_json_column_value_converts_json_strings_for_jsonb() -> None:
    parsed = PostgresRepository.parse_json_column_value('{"T0": 1, "T1": 2}')

    assert isinstance(parsed, dict)
    assert parsed["T0"] == 1
    assert parsed["T1"] == 2


def test_coerce_missing_values_replaces_nan_and_pd_na_with_none() -> None:
    frame = pd.DataFrame(
        {
            "InChIKey": [float("nan"), pd.NA, "ABC123"],
            "molecular_weight": [16.04, float("nan"), 44.01],
        }
    )

    coerced = PostgresRepository.coerce_missing_values(frame)
    records = coerced.to_dict(orient="records")

    assert records[0]["InChIKey"] is None
    assert records[1]["InChIKey"] is None
    assert records[0]["molecular_weight"] == 16.04
    assert records[1]["molecular_weight"] is None


def test_deduplicate_conflict_batch_keeps_last_conflict_record() -> None:
    batch = [
        {"sample_key": "A", "value": 1},
        {"sample_key": "B", "value": 2},
        {"sample_key": "A", "value": 3},
        {"sample_key": None, "value": 4},
        {"sample_key": None, "value": 5},
    ]

    deduplicated, dropped = PostgresRepository.deduplicate_conflict_batch(
        batch, ["sample_key"]
    )

    assert dropped == 1
    assert len(deduplicated) == 4
    assert deduplicated[0]["sample_key"] == "A"
    assert deduplicated[0]["value"] == 3
    assert deduplicated[1]["sample_key"] == "B"
    assert deduplicated[2]["sample_key"] is None
    assert deduplicated[3]["sample_key"] is None
