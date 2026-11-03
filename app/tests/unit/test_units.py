# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import pytest

from server.common.units import (
    BUILT_IN_UNIT_DETAILS,
    CustomUnit,
    UnitCatalog,
    UnitConversionError,
    UnitRegistry,
)


###############################################################################
@pytest.mark.parametrize(
    ("unit", "value", "expected"),
    [
        ("µmol/g", 2.0, 0.002),
        ("mmol/kg", 5.0, 0.005),
        ("mol/g", 1.0, 1000.0),
        ("cm³(STP)/g", 22.41396954, 1.0),
        ("cm3(STP)/kg", 22413.96954, 1.0),
    ],
)
def test_catalog_uptake_units_convert_to_mol_per_kg(unit, value, expected) -> None:  # type: ignore[no-untyped-def]
    assert UnitRegistry.convert_uptake(value, unit).canonical_value == pytest.approx(
        expected
    )


###############################################################################
def test_mmhg_is_distinct_from_torr() -> None:
    assert UnitRegistry.pressure_unit("mmHg") == "mmHg"
    assert UnitRegistry.pressure_unit("Torr") == "torr"


###############################################################################
def test_every_catalog_entry_resolves_from_its_label() -> None:
    for detail in BUILT_IN_UNIT_DETAILS:
        assert UnitRegistry.resolve(detail.quantity, detail.label) == detail.symbol


###############################################################################
def test_custom_unit_without_factor_is_a_label_and_never_converts() -> None:
    catalog = UnitCatalog([CustomUnit("uptake", "mmol/cc")])

    assert catalog.custom_unit("uptake", " MMOL/CC ") is not None
    with pytest.raises(UnitConversionError, match="no conversion definition"):
        catalog.check("uptake", "mmol/cc")
    with pytest.raises(UnitConversionError, match="no conversion definition"):
        catalog.convert_uptake(1.0, "mmol/cc")


###############################################################################
def test_custom_unit_with_factor_converts_explicitly() -> None:
    catalog = UnitCatalog([CustomUnit("pressure", "cmHg", 1333.22387415)])

    result = catalog.convert_pressure(2.0, "cmhg", "absolute")

    assert result.canonical_value == pytest.approx(2666.4477483)
    assert "user-defined" in result.rule
    with pytest.raises(UnitConversionError):
        catalog.convert_pressure(0.5, "cmHg", "relative")


###############################################################################
def test_built_in_units_take_precedence_over_custom_labels() -> None:
    catalog = UnitCatalog([CustomUnit("uptake", "MMOL/G", 123.0)])

    assert catalog.convert_uptake(1.0, "mmol/g").canonical_value == 1.0


###############################################################################
def test_scientifically_distinct_units_do_not_merge() -> None:
    assert UnitRegistry.unit_key("mmol/g") != UnitRegistry.unit_key("mmol/kg")
    assert UnitRegistry.unit_key("mmol / g") == UnitRegistry.unit_key("MMOL/G")


###############################################################################
@pytest.mark.parametrize(
    ("quantity", "factor"),
    [("pressure", 0.0), ("uptake", -1.0), ("temperature", 1.0)],
)
def test_invalid_custom_unit_definitions_are_rejected(quantity, factor) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(UnitConversionError):
        CustomUnit(quantity, "x", factor)
