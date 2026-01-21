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
        self.P_UNIT_COL = "pressureUnits"
        self.conversions: dict[str, Callable[[list[int | float]], list[float]]] = {
            "bar": self.bar_to_pascal,
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def bar_to_pascal(
        p_vals: list[int | float] | int | float | None,
    ) -> list[float] | float | None:
        return map_values(p_vals, lambda value: value * 100000.0)

    # -------------------------------------------------------------------------
    def convert_pressure_units(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if self.P_UNIT_COL not in dataframe.columns or self.P_COL not in dataframe.columns:
            logger.debug("Pressure conversion skipped (missing pressure columns).")
            return dataframe

        dataframe[self.P_COL] = dataframe.apply(
            lambda row: self.conversions.get(row[self.P_UNIT_COL], lambda x: x)(
                row[self.P_COL]
            ),
            axis=1,
        )
        dataframe.drop(columns=self.P_UNIT_COL, inplace=True)

        return dataframe


# [CONVERSION OF UPTAKE]
###############################################################################
class UptakeConversion:
    def __init__(self) -> None:
        self.Q_COL = "adsorbed_amount"
        self.Q_UNIT_COL = "adsorptionUnits"
        self.mol_W = "adsorbate_molecular_weight"

        self.conversions: dict[str, Callable[..., list[float]]] = {
            "mmol/g": self.convert_mmol_g_or_mol_kg,
            "mol/kg": self.convert_mmol_g_or_mol_kg,
            "mmol/kg": self.convert_mmol_kg,
            "mg/g": self.convert_mg_g,
            "g/g": self.convert_g_g,
            "wt%": self.convert_wt_percent,
            "g Adsorbate / 100g Adsorbent": self.convert_g_adsorbate_per_100g_adsorbent,
            "g/100g": self.convert_g_adsorbate_per_100g_adsorbent,
            "ml(STP)/g": self.convert_ml_stp_g_or_cm3_stp_g,
            "cm3(STP)/g": self.convert_ml_stp_g_or_cm3_stp_g,
        }

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
        return map_values(
            q_vals, lambda value: (value / 100.0) / float(mol_weight) * 1000.0
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def convert_ml_stp_g_or_cm3_stp_g(
        q_vals: list[int | float] | int | float | None,
    ) -> list[float] | float | None:
        return map_values(q_vals, lambda value: value / 22.414)

    # -------------------------------------------------------------------------
    def convert_uptake_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if self.Q_UNIT_COL not in dataframe.columns or self.Q_COL not in dataframe.columns:
            logger.debug("Uptake conversion skipped (missing adsorption columns).")
            return dataframe

        def convert_row(row: pd.Series) -> list[float]:
            unit = row.get(self.Q_UNIT_COL)
            values = row.get(self.Q_COL)
            converter = self.conversions.get(unit)
            if converter is None:
                return values
            if unit in {"mg/g", "g/g", "wt%", "g Adsorbate / 100g Adsorbent", "g/100g"}:
                mol_weight = row.get(self.mol_W)
                if mol_weight in (None, 0, ""):
                    return values
                return converter(values, mol_weight)
            return converter(values)

        dataframe[self.Q_COL] = dataframe.apply(convert_row, axis=1)
        dataframe.drop(columns=self.Q_UNIT_COL, inplace=True)

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
