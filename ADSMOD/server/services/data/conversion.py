from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from ADSMOD.server.utils.logger import logger


def map_values(
    values: list[int | float] | int | float | None,
    converter: Callable[[float], float],
) -> list[float] | float | None:
    if values is None:
        return None
    if isinstance(values, (list, tuple)):
        converted: list[float] = []
        for value in values:
            if value is None or pd.isna(value):
                converted.append(float("nan"))
                continue
            converted.append(converter(float(value)))
        return converted
    if pd.isna(values):
        return None
    return converter(float(values))


# [CONVERSION OF PRESSURE]
###############################################################################
class PressureConversion:
    def __init__(self) -> None:
        self.P_COL = "pressure"
        self.P_UNIT_COL = "pressure_units"
        self.P_UNIT_FALLBACK_COL = "pressureUnits"
        self.conversions: dict[
            str,
            Callable[
                [list[int | float] | int | float | None],
                list[float] | float | None,
            ],
        ] = {
            "bar": self.bar_to_pascal,
            "pa": self.pascal_identity,
            "kpa": self.kpa_to_pascal,
            "mpa": self.mpa_to_pascal,
            "atm": self.atm_to_pascal,
            "torr": self.torr_to_pascal,
            "mmhg": self.torr_to_pascal,
            "psi": self.psi_to_pascal,
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_unit(unit: object) -> str:
        if unit is None or pd.isna(unit):
            return ""
        return " ".join(str(unit).strip().lower().split())

    # -------------------------------------------------------------------------
    @staticmethod
    def bar_to_pascal(
        p_vals: list[int | float] | int | float | None,
    ) -> list[float] | float | None:
        return map_values(p_vals, lambda value: value * 100000.0)

    # -------------------------------------------------------------------------
    @staticmethod
    def pascal_identity(
        p_vals: list[int | float] | int | float | None,
    ) -> list[float] | float | None:
        return map_values(p_vals, lambda value: value)

    # -------------------------------------------------------------------------
    @staticmethod
    def kpa_to_pascal(
        p_vals: list[int | float] | int | float | None,
    ) -> list[float] | float | None:
        return map_values(p_vals, lambda value: value * 1000.0)

    # -------------------------------------------------------------------------
    @staticmethod
    def mpa_to_pascal(
        p_vals: list[int | float] | int | float | None,
    ) -> list[float] | float | None:
        return map_values(p_vals, lambda value: value * 1000000.0)

    # -------------------------------------------------------------------------
    @staticmethod
    def atm_to_pascal(
        p_vals: list[int | float] | int | float | None,
    ) -> list[float] | float | None:
        return map_values(p_vals, lambda value: value * 101325.0)

    # -------------------------------------------------------------------------
    @staticmethod
    def torr_to_pascal(
        p_vals: list[int | float] | int | float | None,
    ) -> list[float] | float | None:
        return map_values(p_vals, lambda value: value * 133.322)

    # -------------------------------------------------------------------------
    @staticmethod
    def psi_to_pascal(
        p_vals: list[int | float] | int | float | None,
    ) -> list[float] | float | None:
        return map_values(p_vals, lambda value: value * 6894.757)

    # -------------------------------------------------------------------------
    def convert_pressure_units(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        unit_column = None
        if self.P_UNIT_COL in dataframe.columns:
            unit_column = self.P_UNIT_COL
        elif self.P_UNIT_FALLBACK_COL in dataframe.columns:
            unit_column = self.P_UNIT_FALLBACK_COL

        if (
            unit_column is None
            or self.P_COL not in dataframe.columns
        ):
            logger.debug("Pressure conversion skipped (missing pressure columns).")
            return dataframe

        dataframe[self.P_COL] = dataframe.apply(
            lambda row: self.conversions.get(
                self.normalize_unit(row.get(unit_column)), lambda x: x
            )(row.get(self.P_COL)),
            axis=1,
        )
        dataframe.drop(columns=unit_column, inplace=True)

        return dataframe


# [CONVERSION OF UPTAKE]
###############################################################################
class UptakeConversion:
    def __init__(self) -> None:
        self.Q_COL = "adsorbed_amount"
        self.Q_UNIT_COL = "adsorption_units"
        self.Q_UNIT_FALLBACK_COL = "adsorptionUnits"
        self.mol_weight = "adsorbate_molecular_weight"

        self.weight_units = {
            "mg/g",
            "g/g",
            "wt%",
            "g adsorbate / 100g adsorbent",
            "g/100g",
        }
        self.conversions: dict[str, Callable[..., list[float] | float | None]] = {
            "mmol/g": self.convert_mmol_g_or_mol_kg,
            "mol/kg": self.convert_mmol_g_or_mol_kg,
            "mmol/kg": self.convert_mmol_kg,
            "mg/g": self.convert_mg_g,
            "g/g": self.convert_g_g,
            "wt%": self.convert_wt_percent,
            "g adsorbate / 100g adsorbent": self.convert_g_adsorbate_per_100g_adsorbent,
            "g/100g": self.convert_g_adsorbate_per_100g_adsorbent,
            "ml(stp)/g": self.convert_ml_stp_g_or_cm3_stp_g,
            "cm3(stp)/g": self.convert_ml_stp_g_or_cm3_stp_g,
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_unit(unit: object) -> str:
        if unit is None or pd.isna(unit):
            return ""
        return " ".join(str(unit).strip().lower().split())

    # -------------------------------------------------------------------------
    @staticmethod
    def convert_mmol_g_or_mol_kg(
        q_vals: list[int | float] | int | float | None,
    ) -> list[float] | float | None:
        return map_values(q_vals, lambda value: value)

    # -------------------------------------------------------------------------
    @staticmethod
    def convert_mmol_kg(
        q_vals: list[int | float] | int | float | None,
    ) -> list[float] | float | None:
        return map_values(q_vals, lambda value: value / 1000.0)

    # -------------------------------------------------------------------------
    @staticmethod
    def convert_mg_g(
        q_vals: list[int | float] | int | float | None, mol_weight: float
    ) -> list[float] | float | None:
        return map_values(q_vals, lambda value: value / float(mol_weight))

    # -------------------------------------------------------------------------
    @staticmethod
    def convert_g_g(
        q_vals: list[int | float] | int | float | None, mol_weight: float
    ) -> list[float] | float | None:
        return map_values(q_vals, lambda value: value / float(mol_weight) * 1000.0)

    # -------------------------------------------------------------------------
    @staticmethod
    def convert_wt_percent(
        q_vals: list[int | float] | int | float | None, mol_weight: float
    ) -> list[float] | float | None:
        return map_values(
            q_vals, lambda value: (value / 100.0) / float(mol_weight) * 1000.0
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def convert_g_adsorbate_per_100g_adsorbent(
        q_vals: list[int | float] | int | float | None, mol_weight: float
    ) -> list[float] | float | None:
        return UptakeConversion.convert_wt_percent(q_vals, mol_weight)

    # -------------------------------------------------------------------------
    @staticmethod
    def convert_ml_stp_g_or_cm3_stp_g(
        q_vals: list[int | float] | int | float | None,
    ) -> list[float] | float | None:
        return map_values(q_vals, lambda value: value / 22.414)

    # -------------------------------------------------------------------------
    def convert_uptake_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        unit_column = None
        if self.Q_UNIT_COL in dataframe.columns:
            unit_column = self.Q_UNIT_COL
        elif self.Q_UNIT_FALLBACK_COL in dataframe.columns:
            unit_column = self.Q_UNIT_FALLBACK_COL

        if (
            unit_column is None
            or self.Q_COL not in dataframe.columns
        ):
            logger.debug("Uptake conversion skipped (missing adsorption columns).")
            return dataframe

        def convert_row(row: pd.Series) -> list[float] | float | None:
            unit = self.normalize_unit(row.get(unit_column))
            values = row.get(self.Q_COL)
            converter = self.conversions.get(unit)
            if converter is None:
                return values
            if unit in self.weight_units:
                mol_weight = row.get(self.mol_weight)
                if mol_weight in (None, 0, "") or pd.isna(mol_weight):
                    return values
                return converter(values, mol_weight)
            return converter(values)

        dataframe[self.Q_COL] = dataframe.apply(convert_row, axis=1)
        dataframe.drop(columns=unit_column, inplace=True)

        return dataframe


###############################################################################
def PQ_units_conversion(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert pressure to Pascal and uptake to mmol/g, removing unit columns."""
    if dataframe.empty:
        return dataframe

    P_converter = PressureConversion()
    Q_converter = UptakeConversion()
    converted_data = P_converter.convert_pressure_units(dataframe)
    converted_data = Q_converter.convert_uptake_data(converted_data)

    return converted_data
