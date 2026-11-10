# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations
from pathlib import Path
import pytest
import pandas as pd
from server.domain.datasets import ImportMapping
from server.services.data.importer import AdsorptionImportEngine
from server.common.utils.encoding import (
    decode_json_response_bytes,
    normalize_unicode_text,
    sanitize_dataframe_strings,
)


# Canonical adsorption import

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


###############################################################################
def test_atomic_import_groups_rows_and_normalizes_units() -> None:
    payload = b"experiment_id,Pressure [bar],Uptake [mmol/g],Temperature [K],Adsorbate,Adsorbent\nEXP-1,0.1,0.42,298.15,CO2,13X\nEXP-1,0.2,0.73,298.15,CO2,13X\n"
    engine = AdsorptionImportEngine()
    preview = engine.preview(payload, "sample.csv")
    mapping = ImportMapping(
        dataset_name="sample",
        structure="atomic",
        column_roles={column.name: column.proposed_role for column in preview.columns},
        grouping_columns=["experiment_id"],
        pressure_basis="absolute",
    )
    bundle = engine.validate(payload, "sample.csv", mapping)
    assert bundle.response.status == "valid"
    assert bundle.response.experiment_count == 1
    assert bundle.response.observation_count == 2
    assert bundle.experiments[0]["observations"][0]["pressure_canonical"] == 10_000
    assert bundle.experiments[0]["observations"][0]["uptake_mol_kg"] == 0.42


###############################################################################
def test_aggregated_arrays_require_equal_lengths() -> None:
    payload = b'experiment_id,pressure,uptake,temperature,adsorbate,adsorbent\nEXP-1,"[1;2]","[3]",298.15,CO2,13X\n'
    engine = AdsorptionImportEngine()
    preview = engine.preview(payload, "sample.csv")
    mapping = ImportMapping(
        dataset_name="sample",
        structure="aggregated",
        column_roles={column.name: column.proposed_role for column in preview.columns},
        grouping_columns=["experiment_id"],
        pressure_basis="absolute",
        series_delimiter=";",
        unit_overrides={"pressure": "bar", "uptake": "mmol/g", "temperature": "K"},
    )
    bundle = engine.validate(payload, "sample.csv", mapping)
    assert bundle.response.status == "invalid"
    assert any(issue.code == "invalid_row" for issue in bundle.response.issues)


###############################################################################
def test_atomic_import_preserves_adsorbate_smiles_for_training() -> None:
    payload = b"experiment_id,pressure [kPa],uptake [mol/g],temperature [K],adsorbate,adsorbate_smiles,adsorbent\nEXP-1,10,0.10,298.15,CO2,O=C=O,Activated carbon\nEXP-1,20,0.18,298.15,CO2,O=C=O,Activated carbon\n"
    engine = AdsorptionImportEngine()
    preview = engine.preview(payload, "smiles.csv")
    mapping = ImportMapping(
        dataset_name="smiles",
        structure="atomic",
        column_roles={column.name: column.proposed_role for column in preview.columns},
        grouping_columns=["experiment_id"],
        pressure_basis="absolute",
    )

    bundle = engine.validate(payload, "smiles.csv", mapping)

    assert bundle.response.status == "valid"
    assert bundle.experiments[0]["adsorbates"][0]["smiles"] == "O=C=O"
    assert bundle.response.experiments[0].adsorbates == ["CO2"]


###############################################################################
def _wide_species_mapping(engine: AdsorptionImportEngine, payload: bytes):
    preview = engine.preview(payload, "mixture.csv")
    mapping = ImportMapping(
        dataset_name="mixture",
        structure="atomic",
        column_roles={
            column.name: "ignore"
            if column.name in {item.uptake_column for item in preview.proposed_species_columns}
            else column.proposed_role
            for column in preview.columns
        },
        grouping_columns=["experiment_id"],
        pressure_basis="absolute",
        species_columns=preview.proposed_species_columns,
    )
    return preview, mapping


###############################################################################
def test_wide_species_columns_become_one_multi_species_experiment() -> None:
    payload = (
        b"experiment_id,Pressure [kPa],uptake_CO2 [mmol/g],uptake_N2 [mmol/g],Temperature [K],Adsorbent\n"
        b"EXP-1,10,0.5,0.1,298.15,13X\n"
        b"EXP-1,20,0.9,0.2,298.15,13X\n"
        b"EXP-1,30,1.1,,298.15,13X\n"
    )
    engine = AdsorptionImportEngine()
    preview, mapping = _wide_species_mapping(engine, payload)
    bundle = engine.validate(payload, "mixture.csv", mapping)

    assert [item.adsorbate for item in preview.proposed_species_columns] == ["CO2", "N2"]
    assert bundle.response.status == "valid"
    assert bundle.response.experiment_count == 1
    assert bundle.response.experiments[0].adsorbates == ["CO2", "N2"]
    experiment = bundle.experiments[0]
    assert [item["name"] for item in experiment["adsorbates"]] == ["CO2", "N2"]
    by_species = {
        name: [item for item in experiment["observations"] if item["adsorbate"] == name]
        for name in ("CO2", "N2")
    }
    assert [item["sequence_index"] for item in by_species["CO2"]] == [0, 1, 2]
    assert [item["sequence_index"] for item in by_species["N2"]] == [0, 1]
    assert [item["source_row"] for item in by_species["N2"]] == [2, 3]
    assert by_species["N2"][0]["uptake_mol_kg"] == 0.1
    assert by_species["N2"][0]["pressure_canonical"] == 10_000


###############################################################################
def test_long_form_species_column_groups_species_under_one_experiment() -> None:
    payload = (
        b"experiment_id,pressure [kPa],uptake [mmol/g],temperature [K],adsorbate,adsorbent\n"
        b"EXP-1,10,0.5,298.15,CO2,13X\nEXP-1,20,0.9,298.15,CO2,13X\n"
        b"EXP-1,10,0.1,298.15,N2,13X\nEXP-1,20,0.2,298.15,N2,13X\n"
    )
    engine = AdsorptionImportEngine()
    preview = engine.preview(payload, "long.csv")
    mapping = ImportMapping(
        dataset_name="long",
        structure="atomic",
        column_roles={column.name: column.proposed_role for column in preview.columns},
        grouping_columns=["experiment_id"],
        pressure_basis="absolute",
    )
    bundle = engine.validate(payload, "long.csv", mapping)

    assert preview.proposed_species_columns == []
    assert bundle.response.status == "valid"
    assert bundle.response.experiment_count == 1
    assert bundle.response.experiments[0].adsorbates == ["CO2", "N2"]
    assert bundle.response.observation_count == 4


###############################################################################
def test_conflicting_species_smiles_in_one_experiment_blocks_validation() -> None:
    payload = (
        b"experiment_id,pressure [kPa],uptake [mmol/g],temperature [K],adsorbate,adsorbate_smiles,adsorbent\n"
        b"EXP-1,10,0.5,298.15,CO2,O=C=O,13X\n"
        b"EXP-1,20,0.9,298.15,CO2,C(=O)=O,13X\n"
    )
    engine = AdsorptionImportEngine()
    preview = engine.preview(payload, "conflict.csv")
    mapping = ImportMapping(
        dataset_name="conflict",
        structure="atomic",
        column_roles={column.name: column.proposed_role for column in preview.columns},
        grouping_columns=["experiment_id"],
        pressure_basis="absolute",
    )

    bundle = engine.validate(payload, "conflict.csv", mapping)

    assert bundle.response.status == "invalid"
    issue = next(
        item
        for item in bundle.response.issues
        if item.code == "conflicting_species_metadata"
    )
    assert issue.source_row == 3
    assert "CO2" in issue.message
    assert "O=C=O" in issue.message and "C(=O)=O" in issue.message


###############################################################################
def test_wide_species_columns_with_different_units_are_rejected() -> None:
    payload = (
        b"experiment_id,Pressure [kPa],uptake_CO2 [mmol/g],uptake_N2 [mol/kg],Temperature [K],Adsorbent\n"
        b"EXP-1,10,0.5,0.1,298.15,13X\nEXP-1,20,0.9,0.2,298.15,13X\n"
    )
    engine = AdsorptionImportEngine()
    _, mapping = _wide_species_mapping(engine, payload)
    bundle = engine.validate(payload, "mixture.csv", mapping)

    assert bundle.response.status == "invalid"
    assert any(issue.code == "species_unit_mismatch" for issue in bundle.response.issues)


###############################################################################
@pytest.mark.parametrize("extension", [".xls", ".xlsx"])
def test_excel_workbooks_preview_and_validate_as_canonical_datasets(
    extension: str,
) -> None:
    payload = (FIXTURES_DIR / f"sample_adsorption{extension}").read_bytes()
    engine = AdsorptionImportEngine()

    preview = engine.preview(payload, f"sample_adsorption{extension}")
    mapping = ImportMapping(
        dataset_name="excel-sample",
        structure=preview.detected_structure,
        column_roles={column.name: column.proposed_role for column in preview.columns},
        grouping_columns=preview.proposed_grouping_columns,
        pressure_basis=preview.proposed_pressure_basis or "absolute",
        unit_overrides={
            column.proposed_role: column.detected_unit
            for column in preview.columns
            if column.detected_unit
            and column.proposed_role in {"pressure", "uptake", "temperature"}
        },
    )
    bundle = engine.validate(payload, f"sample_adsorption{extension}", mapping)

    assert preview.row_count == 6
    assert preview.column_count == 9
    assert bundle.response.status == "valid"
    assert bundle.response.experiment_count == 2
    assert bundle.response.observation_count == 6
    assert bundle.experiments[0]["observations"][0]["pressure_canonical"] == 10_000
    assert bundle.experiments[0]["observations"][0]["uptake_mol_kg"] == 0.12


###############################################################################
def test_import_rejects_extensions_outside_the_canonical_policy() -> None:
    payload = b"pressure,uptake,temperature,adsorbate,adsorbent\n1,2,298,CO2,13X\n"

    with pytest.raises(ValueError, match=r"Allowed file types: \.csv, \.xls, \.xlsx"):
        AdsorptionImportEngine().preview(payload, "sample.json")


###############################################################################
def test_import_engine_accepts_extensions_supplied_by_runtime_configuration() -> None:
    preview = AdsorptionImportEngine(allowed_extensions=(".json",)).preview(
        b'[{"pressure": 1, "uptake": 2, "temperature": 298, "adsorbate": "CO2", "adsorbent": "13X"}]',
        "sample.json",
    )

    assert preview.filename == "sample.json"


# Encoding unicode


###############################################################################
def test_normalize_unicode_text_preserves_scientific_unicode() -> None:
    raw = "⁰¹²³⁴⁵⁶⁷⁸⁹ ₀₁₂₃₄₅₆₇₈₉ μµ αβγΔΩ ×÷±∓∞≈≠≤≥ √∑∏∫∂∇ ° ′ ″ µm Å Å Ω °C °F áéíóú ñçü åøæß “quotes” ‘single’ – — … • ‧"
    assert normalize_unicode_text(raw) == raw


###############################################################################
def test_normalize_unicode_text_removes_zero_width_and_normalizes_nbsp() -> None:
    raw = "A\u00a0B\u200bC\u200dD"
    assert normalize_unicode_text(raw) == "A BCD"


###############################################################################
def test_decode_json_response_bytes_utf8_round_trip() -> None:
    payload = decode_json_response_bytes(
        b'{"name":"CO\xe2\x82\x82","unit":"\xc2\xb5mol/g","sup":"\xe2\x81\xb0"}'
    )
    assert payload == {"name": "CO₂", "unit": "µmol/g", "sup": "⁰"}


###############################################################################
def test_sanitize_dataframe_strings_handles_unicode_without_loss() -> None:
    frame = pd.DataFrame(
        {
            "name": ["zeolite\u200b-a", "na\u00a0y"],
            "unit": ["µmol/g", "Å"],
        }
    )
    sanitized = sanitize_dataframe_strings(frame)
    assert sanitized.loc[0, "name"] == "zeolite-a"
    assert sanitized.loc[1, "name"] == "na y"
    assert sanitized.loc[0, "unit"] == "µmol/g"
    assert sanitized.loc[1, "unit"] == "Å"


###############################################################################
def _atomic_mapping(engine: AdsorptionImportEngine, payload: bytes, **overrides):  # type: ignore[no-untyped-def]
    preview = engine.preview(payload, "sample.csv")
    return ImportMapping(
        dataset_name="sample",
        structure="atomic",
        column_roles={column.name: column.proposed_role for column in preview.columns},
        grouping_columns=["experiment_id"],
        pressure_basis="absolute",
        **overrides,
    )


UNIT_PAYLOAD = b"experiment_id,Pressure [bar],Uptake [mmol/g],Temperature [K],Adsorbate,Adsorbent\nEXP-1,0.1,0.42,298.15,CO2,13X\nEXP-1,0.2,0.73,298.15,CO2,13X\n"


###############################################################################
def test_unknown_unit_override_gives_one_upfront_error_not_one_per_row() -> None:
    engine = AdsorptionImportEngine()
    mapping = _atomic_mapping(engine, UNIT_PAYLOAD, unit_overrides={"uptake": "furlong"})

    issues = engine.validate(UNIT_PAYLOAD, "sample.csv", mapping).response.issues

    assert [issue.code for issue in issues] == ["unsupported_unit"]
    assert issues[0].column == "uptake"


###############################################################################
def test_label_only_custom_unit_blocks_import_with_conversion_error() -> None:
    engine = AdsorptionImportEngine()
    mapping = _atomic_mapping(
        engine,
        UNIT_PAYLOAD,
        unit_overrides={"uptake": "mmol/cc"},
        unit_definitions=[{"quantity": "uptake", "symbol": "mmol/cc"}],
    )

    bundle = engine.validate(UNIT_PAYLOAD, "sample.csv", mapping)

    assert bundle.response.status == "invalid"
    assert "no conversion definition" in bundle.response.issues[0].message
    assert bundle.experiments == []


###############################################################################
def test_custom_unit_with_factor_and_override_warning() -> None:
    engine = AdsorptionImportEngine()
    mapping = _atomic_mapping(
        engine,
        UNIT_PAYLOAD,
        unit_overrides={"pressure": "cmHg"},
        unit_definitions=[
            {"quantity": "pressure", "symbol": "cmHg", "factor_to_canonical": 1333.22387415}
        ],
    )

    bundle = engine.validate(UNIT_PAYLOAD, "sample.csv", mapping)

    assert bundle.response.status == "valid"
    first = bundle.experiments[0]["observations"][0]
    assert first["pressure_original_unit"] == "cmHg"
    assert first["pressure_canonical"] == pytest.approx(133.322387415)
    assert [issue.code for issue in bundle.response.issues] == ["unit_override_applied"]
