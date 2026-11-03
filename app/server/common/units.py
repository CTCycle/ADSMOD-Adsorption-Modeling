# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal


PressureBasis = Literal["absolute", "partial", "relative"]

###############################################################################
class UnitConversionError(ValueError):
    pass

###############################################################################
@dataclass(frozen=True)
class ConvertedValue:
    original_value: float
    original_unit: str
    canonical_value: float
    canonical_unit: str
    rule: str

###############################################################################
def normalize_token(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    text = text.replace("−", "-").replace("·", " ").replace("³", "3")
    return " ".join(text.split())

###############################################################################
def parse_number(value: Any, decimal_separator: str = "auto") -> float:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid measurements.")
    if isinstance(value, (int, float)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Measurement must be finite.")
        return result

    text = normalize_token(value).replace("\u00a0", " ").strip()
    if not text:
        raise ValueError("Measurement is empty.")
    match = re.fullmatch(
        r"\s*([+-]?(?:\d[\d\s.,']*|\d*[.,]\d+)(?:[eE][+-]?\d+)?)\s*(?:[^\d].*)?",
        text,
    )
    if match is None:
        raise ValueError(f"'{value}' is not a valid numerical value.")
    numeric = match.group(1).replace(" ", "").replace("'", "")

    if decimal_separator not in {"auto", ".", ","}:
        raise ValueError("Decimal separator must be 'auto', '.', or ','.")
    if decimal_separator == ".":
        numeric = numeric.replace(",", "")
    elif decimal_separator == ",":
        numeric = numeric.replace(".", "").replace(",", ".")
    elif "." in numeric and "," in numeric:
        decimal = "." if numeric.rfind(".") > numeric.rfind(",") else ","
        thousands = "," if decimal == "." else "."
        numeric = numeric.replace(thousands, "").replace(decimal, ".")
    elif "," in numeric:
        if numeric.count(",") == 1 and len(numeric.rsplit(",", 1)[1]) == 3:
            raise ValueError(
                f"'{value}' is ambiguous; select comma-decimal or comma-thousands explicitly."
            )
        numeric = numeric.replace(",", ".")
    elif (
        "." in numeric
        and numeric.count(".") == 1
        and len(numeric.rsplit(".", 1)[1]) == 3
    ):
        raise ValueError(
            f"'{value}' is ambiguous; select decimal-point or thousands-point explicitly."
        )

    try:
        result = float(numeric)
    except ValueError as exc:
        raise ValueError(f"'{value}' is not a valid numerical value.") from exc
    if not math.isfinite(result):
        raise ValueError("Measurement must be finite.")
    return result

###############################################################################
class UnitRegistry:
    GAS_CONSTANT_J_MOL_K = 8.31446261815324
    STP_MOLAR_VOLUME_L_MOL = 22.41396954

    PRESSURE_ALIASES = {
        "pa": "Pa",
        "pascal": "Pa",
        "pascals": "Pa",
        "kpa": "kPa",
        "kilopascal": "kPa",
        "mpa": "MPa",
        "megapascal": "MPa",
        "bar": "bar",
        "mbar": "mbar",
        "atm": "atm",
        "atmosphere": "atm",
        "torr": "torr",
        "mmhg": "mmHg",
        "psi": "psi",
        "p/p0": "1",
        "p/p₀": "1",
        "relative": "1",
        "relative pressure": "1",
        "1": "1",
        "%": "%",
        "percent": "%",
    }
    PRESSURE_TO_PA = {
        "Pa": 1.0,
        "kPa": 1_000.0,
        "MPa": 1_000_000.0,
        "bar": 100_000.0,
        "mbar": 100.0,
        "atm": 101_325.0,
        "torr": 133.32236842105263,
        "mmHg": 133.322387415,
        "psi": 6_894.757293168,
    }
    UPTAKE_ALIASES = {
        "mol/kg": "mol/kg",
        "mol kg-1": "mol/kg",
        "mol kg^-1": "mol/kg",
        "mmol/g": "mmol/g",
        "mmol g-1": "mmol/g",
        "mmol g^-1": "mmol/g",
        "mmol/kg": "mmol/kg",
        "µmol/g": "umol/g",
        "umol/g": "umol/g",
        "mol/g": "mol/g",
        "mg/g": "mg/g",
        "g/g": "g/g",
        "mass fraction": "g/g",
        "wt%": "wt%",
        "weight %": "wt%",
        "g/100g": "wt%",
        "g adsorbate / 100g adsorbent": "wt%",
        "cm3(stp)/g": "cm3(STP)/g",
        "cm3 stp/g": "cm3(STP)/g",
        "cm3/g stp": "cm3(STP)/g",
        "ml(stp)/g": "cm3(STP)/g",
        "ml stp/g": "cm3(STP)/g",
        "cm3(stp)/kg": "cm3(STP)/kg",
        "cm3 stp/kg": "cm3(STP)/kg",
        "ml(stp)/kg": "cm3(STP)/kg",
    }
    TEMPERATURE_ALIASES = {
        "k": "K",
        "kelvin": "K",
        "°c": "degC",
        "c": "degC",
        "degc": "degC",
        "celsius": "degC",
        "°f": "degF",
        "f": "degF",
        "degf": "degF",
        "fahrenheit": "degF",
    }

    # -------------------------------------------------------------------------
    @staticmethod
    def unit_key(unit: object) -> str:
        return normalize_token(unit).replace(" ", "")

    # -------------------------------------------------------------------------
    @classmethod
    def aliases_for(cls, quantity: str) -> dict[str, str]:
        table = {
            "pressure": cls.PRESSURE_ALIASES,
            "uptake": cls.UPTAKE_ALIASES,
            "temperature": cls.TEMPERATURE_ALIASES,
        }.get(quantity)
        if table is None:
            raise UnitConversionError(f"Unknown physical quantity '{quantity}'.")
        return {cls.unit_key(key): value for key, value in table.items()}

    # -------------------------------------------------------------------------
    @classmethod
    def resolve(cls, quantity: str, unit: object) -> str:
        resolved = cls.aliases_for(quantity).get(cls.unit_key(unit))
        if resolved is None:
            raise UnitConversionError(f"Unsupported {quantity} unit '{unit}'.")
        return resolved

    # -------------------------------------------------------------------------
    @classmethod
    def pressure_unit(cls, unit: object) -> str:
        return cls.resolve("pressure", unit)

    # -------------------------------------------------------------------------
    @classmethod
    def uptake_unit(cls, unit: object) -> str:
        return cls.resolve("uptake", unit)

    # -------------------------------------------------------------------------
    @classmethod
    def temperature_unit(cls, unit: object) -> str:
        return cls.resolve("temperature", unit)

    # -------------------------------------------------------------------------
    @classmethod
    def convert_pressure(
        cls, value: float, unit: object, basis: PressureBasis
    ) -> ConvertedValue:
        resolved = cls.pressure_unit(unit)
        if basis == "relative":
            if resolved == "1":
                canonical = value
                rule = "dimensionless identity"
            elif resolved == "%":
                canonical = value / 100.0
                rule = "percent / 100"
            else:
                raise UnitConversionError(
                    "Relative pressure requires p/p0, dimensionless, or percent units."
                )
            if not 0 <= canonical <= 1:
                raise UnitConversionError("Relative pressure must be between 0 and 1.")
            return ConvertedValue(value, resolved, canonical, "1", rule)

        if resolved not in cls.PRESSURE_TO_PA:
            raise UnitConversionError(
                f"{basis.capitalize()} pressure requires a dimensional pressure unit."
            )
        factor = cls.PRESSURE_TO_PA[resolved]
        canonical = value * factor
        if canonical < 0:
            raise UnitConversionError("Pressure must not be negative.")
        return ConvertedValue(
            value, resolved, canonical, "Pa", f"{resolved} * {factor:g}"
        )

    # -------------------------------------------------------------------------
    @classmethod
    def convert_temperature(cls, value: float, unit: object) -> ConvertedValue:
        resolved = cls.temperature_unit(unit)
        if resolved == "K":
            canonical = value
            rule = "kelvin identity"
        elif resolved == "degC":
            canonical = value + 273.15
            rule = "degC + 273.15"
        else:
            canonical = (value - 32.0) * 5.0 / 9.0 + 273.15
            rule = "(degF - 32) * 5/9 + 273.15"
        if canonical <= 0:
            raise UnitConversionError("Temperature must be above absolute zero.")
        return ConvertedValue(value, resolved, canonical, "K", rule)

    # -------------------------------------------------------------------------
    @classmethod
    def uptake_factor_to_mol_kg(
        cls, unit: object, molar_mass_g_mol: float | None = None
    ) -> tuple[str, float, str]:
        resolved = cls.uptake_unit(unit)
        if resolved in {"mol/kg", "mmol/g"}:
            return resolved, 1.0, f"{resolved} == mol/kg"
        if resolved in {"mmol/kg", "umol/g"}:
            return resolved, 0.001, f"{resolved} / 1000"
        if resolved == "mol/g":
            return resolved, 1000.0, "mol/g * 1000"
        if resolved == "cm3(STP)/g":
            factor = 1.0 / cls.STP_MOLAR_VOLUME_L_MOL
            return (
                resolved,
                factor,
                f"cm3(STP)/g / {cls.STP_MOLAR_VOLUME_L_MOL:g} L/mol",
            )
        if resolved == "cm3(STP)/kg":
            factor = 0.001 / cls.STP_MOLAR_VOLUME_L_MOL
            return (
                resolved,
                factor,
                f"cm3(STP)/kg / 1000 / {cls.STP_MOLAR_VOLUME_L_MOL:g} L/mol",
            )
        if molar_mass_g_mol is None or molar_mass_g_mol <= 0:
            raise UnitConversionError(
                f"Uptake unit '{resolved}' requires a positive adsorbate molar mass."
            )
        if resolved == "mg/g":
            return resolved, 1.0 / molar_mass_g_mol, "mg/g / molar mass"
        if resolved == "g/g":
            return resolved, 1000.0 / molar_mass_g_mol, "g/g * 1000 / molar mass"
        if resolved == "wt%":
            return resolved, 10.0 / molar_mass_g_mol, "wt% * 10 / molar mass"
        raise UnitConversionError(f"Unsupported uptake unit '{unit}'.")

    # -------------------------------------------------------------------------
    @classmethod
    def convert_uptake(
        cls, value: float, unit: object, molar_mass_g_mol: float | None = None
    ) -> ConvertedValue:
        resolved, factor, rule = cls.uptake_factor_to_mol_kg(unit, molar_mass_g_mol)
        canonical = value * factor
        if canonical < 0:
            raise UnitConversionError("Adsorption uptake must not be negative.")
        return ConvertedValue(value, resolved, canonical, "mol/kg", rule)

    # -------------------------------------------------------------------------
    @classmethod
    def pressure_from_pa(cls, value_pa: float, unit: object) -> float:
        resolved = cls.pressure_unit(unit)
        if resolved not in cls.PRESSURE_TO_PA:
            raise UnitConversionError(
                "A dimensional display pressure unit is required."
            )
        return value_pa / cls.PRESSURE_TO_PA[resolved]

    # -------------------------------------------------------------------------
    @classmethod
    def uptake_from_mol_kg(
        cls, value: float, unit: object, molar_mass_g_mol: float | None = None
    ) -> float:
        _, factor, _ = cls.uptake_factor_to_mol_kg(unit, molar_mass_g_mol)
        return value / factor


HEADER_UNIT_PATTERN = re.compile(r"(?:\[|\()([^\])]+)(?:\]|\))")

###############################################################################
def detect_header_unit(header: str, quantity: str) -> str | None:
    candidates = [match.strip() for match in HEADER_UNIT_PATTERN.findall(header)]
    candidates.extend(
        token
        for token in (
            "p/p0",
            "bar",
            "kpa",
            "mpa",
            "pa",
            "mmol/g",
            "mol/kg",
            "mg/g",
            "cm3(stp)/g",
            "cm3(stp)/kg",
            "umol/g",
            "°c",
            "°f",
            "kelvin",
        )
        if token in normalize_token(header)
    )
    resolver = {
        "pressure": UnitRegistry.pressure_unit,
        "uptake": UnitRegistry.uptake_unit,
        "temperature": UnitRegistry.temperature_unit,
    }[quantity]
    for candidate in candidates:
        try:
            return resolver(candidate)
        except UnitConversionError:
            continue
    return None


KNOWN_MOLAR_MASSES_G_MOL = {
    "co2": 44.0095,
    "carbon dioxide": 44.0095,
    "n2": 28.0134,
    "nitrogen": 28.0134,
    "ch4": 16.0425,
    "methane": 16.0425,
    "o2": 31.9988,
    "h2": 2.01588,
    "hydrogen": 2.01588,
}

###############################################################################
@dataclass(frozen=True)
class UnitDefinition:
    quantity: str
    symbol: str
    label: str
    name: str
    group: str
    note: str = ""

###############################################################################
# Display metadata for every built-in unit. Conversion factors stay in
# UnitRegistry; this table only describes them for selection and documentation.
_STP_NOTE = "Standard conditions: 0 °C and 1 atm (22.414 L/mol)."
_MOLAR_MASS_NOTE = "Requires the adsorbate molar mass."
BUILT_IN_UNIT_DETAILS: tuple[UnitDefinition, ...] = (
    UnitDefinition("pressure", "Pa", "Pa", "pascal", "Absolute pressure"),
    UnitDefinition("pressure", "kPa", "kPa", "kilopascal", "Absolute pressure"),
    UnitDefinition("pressure", "MPa", "MPa", "megapascal", "Absolute pressure"),
    UnitDefinition("pressure", "bar", "bar", "bar", "Absolute pressure"),
    UnitDefinition("pressure", "mbar", "mbar", "millibar", "Absolute pressure"),
    UnitDefinition("pressure", "atm", "atm", "standard atmosphere", "Absolute pressure"),
    UnitDefinition("pressure", "torr", "Torr", "torr", "Absolute pressure"),
    UnitDefinition("pressure", "mmHg", "mmHg", "millimetre of mercury", "Absolute pressure"),
    UnitDefinition("pressure", "psi", "psi", "pound per square inch", "Absolute pressure"),
    UnitDefinition(
        "pressure",
        "1",
        "p/p₀",
        "relative pressure",
        "Relative pressure",
        "Dimensionless fraction of the saturation pressure (0 to 1).",
    ),
    UnitDefinition(
        "pressure",
        "%",
        "%",
        "relative pressure, percent",
        "Relative pressure",
        "Percent of the saturation pressure (0 to 100).",
    ),
    UnitDefinition("uptake", "mol/kg", "mol/kg", "mole per kilogram", "Molar uptake"),
    UnitDefinition("uptake", "mmol/g", "mmol/g", "millimole per gram", "Molar uptake"),
    UnitDefinition("uptake", "mmol/kg", "mmol/kg", "millimole per kilogram", "Molar uptake"),
    UnitDefinition("uptake", "umol/g", "µmol/g", "micromole per gram", "Molar uptake"),
    UnitDefinition("uptake", "mol/g", "mol/g", "mole per gram", "Molar uptake"),
    UnitDefinition(
        "uptake", "mg/g", "mg/g", "milligram per gram", "Mass uptake", _MOLAR_MASS_NOTE
    ),
    UnitDefinition(
        "uptake",
        "g/g",
        "g/g",
        "gram per gram (mass fraction)",
        "Mass uptake",
        "Mass of adsorbate per mass of adsorbent. " + _MOLAR_MASS_NOTE,
    ),
    UnitDefinition(
        "uptake",
        "wt%",
        "wt%",
        "weight percent",
        "Mass uptake",
        "Grams of adsorbate per 100 g of adsorbent. " + _MOLAR_MASS_NOTE,
    ),
    UnitDefinition(
        "uptake",
        "cm3(STP)/g",
        "cm³(STP)/g",
        "cubic centimetre (STP) per gram",
        "Volumetric uptake",
        _STP_NOTE,
    ),
    UnitDefinition(
        "uptake",
        "cm3(STP)/kg",
        "cm³(STP)/kg",
        "cubic centimetre (STP) per kilogram",
        "Volumetric uptake",
        _STP_NOTE,
    ),
    UnitDefinition("temperature", "K", "K", "kelvin", "Temperature"),
    UnitDefinition("temperature", "degC", "°C", "degree Celsius", "Temperature"),
    UnitDefinition("temperature", "degF", "°F", "degree Fahrenheit", "Temperature"),
)

###############################################################################
@dataclass(frozen=True)
class CustomUnit:
    quantity: str
    symbol: str
    factor_to_canonical: float | None = None

    def __post_init__(self) -> None:
        if self.quantity not in {"pressure", "uptake", "temperature"}:
            raise UnitConversionError(f"Unknown physical quantity '{self.quantity}'.")
        if not self.symbol.strip():
            raise UnitConversionError("A unit symbol is required.")
        if self.factor_to_canonical is not None:
            if self.quantity == "temperature":
                raise UnitConversionError(
                    "Custom temperature units cannot define a conversion factor "
                    "because temperature scales may need an offset."
                )
            if (
                not math.isfinite(self.factor_to_canonical)
                or self.factor_to_canonical <= 0
            ):
                raise UnitConversionError(
                    "A conversion factor must be positive and finite."
                )

    @property
    def key(self) -> str:
        return UnitRegistry.unit_key(self.symbol)

###############################################################################
class UnitCatalog:
    """Built-in units plus user-defined labels, without implicit conversions.

    A custom unit converts only when it carries an explicit factor to the
    canonical unit (Pa for pressure, mol/kg for uptake). Otherwise it is a valid
    label whose conversion raises UnitConversionError.
    """

    # -------------------------------------------------------------------------
    def __init__(self, custom_units: Iterable[CustomUnit] = ()) -> None:
        self._custom = {(unit.quantity, unit.key): unit for unit in custom_units}

    # -------------------------------------------------------------------------
    def custom_unit(self, quantity: str, unit: object) -> CustomUnit | None:
        key = UnitRegistry.unit_key(unit)
        if key in UnitRegistry.aliases_for(quantity):
            return None
        return self._custom.get((quantity, key))

    # -------------------------------------------------------------------------
    def check(self, quantity: str, unit: object) -> str:
        """Return the unit symbol, or raise if it is unknown or not convertible."""
        custom = self.custom_unit(quantity, unit)
        if custom is None:
            return UnitRegistry.resolve(quantity, unit)
        if custom.factor_to_canonical is None:
            raise UnitConversionError(self._missing_conversion(quantity, custom.symbol))
        return custom.symbol

    # -------------------------------------------------------------------------
    @staticmethod
    def _missing_conversion(quantity: str, symbol: str) -> str:
        return (
            f"The {quantity} unit '{symbol}' is saved as a label but has no conversion "
            "definition, so values cannot be converted safely. Choose a supported "
            "unit, convert the source values, or define a conversion factor."
        )

    # -------------------------------------------------------------------------
    def convert_pressure(
        self, value: float, unit: object, basis: PressureBasis
    ) -> ConvertedValue:
        custom = self.custom_unit("pressure", unit)
        if custom is None:
            return UnitRegistry.convert_pressure(value, unit, basis)
        if custom.factor_to_canonical is None:
            raise UnitConversionError(
                self._missing_conversion("pressure", custom.symbol)
            )
        if basis == "relative":
            raise UnitConversionError(
                "Relative pressure requires p/p0, dimensionless, or percent units."
            )
        canonical = value * custom.factor_to_canonical
        if canonical < 0:
            raise UnitConversionError("Pressure must not be negative.")
        return ConvertedValue(
            value,
            custom.symbol,
            canonical,
            "Pa",
            f"{custom.symbol} * {custom.factor_to_canonical:g} (user-defined)",
        )

    # -------------------------------------------------------------------------
    def convert_uptake(
        self, value: float, unit: object, molar_mass_g_mol: float | None = None
    ) -> ConvertedValue:
        custom = self.custom_unit("uptake", unit)
        if custom is None:
            return UnitRegistry.convert_uptake(value, unit, molar_mass_g_mol)
        if custom.factor_to_canonical is None:
            raise UnitConversionError(self._missing_conversion("uptake", custom.symbol))
        canonical = value * custom.factor_to_canonical
        if canonical < 0:
            raise UnitConversionError("Adsorption uptake must not be negative.")
        return ConvertedValue(
            value,
            custom.symbol,
            canonical,
            "mol/kg",
            f"{custom.symbol} * {custom.factor_to_canonical:g} (user-defined)",
        )

    # -------------------------------------------------------------------------
    def convert_temperature(self, value: float, unit: object) -> ConvertedValue:
        custom = self.custom_unit("temperature", unit)
        if custom is None:
            return UnitRegistry.convert_temperature(value, unit)
        raise UnitConversionError(
            self._missing_conversion("temperature", custom.symbol)
        )
