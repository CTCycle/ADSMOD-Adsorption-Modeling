from __future__ import annotations

import pandas as pd

from ADSMOD.server.common.constants import DEFAULT_DATASET_COLUMN_MAPPING
from ADSMOD.server.common.utils.logger import logger
from ADSMOD.server.domain.fitting import (
    NISTFittingDatasetPayload,
    NISTFittingDatasetResponse,
)
from ADSMOD.server.repositories.queries.nist import NISTDataSerializer
from ADSMOD.server.services.data.conversion import (
    PQ_units_conversion,
    PressureConversion,
    UptakeConversion,
)


###############################################################################
class FittingNISTDatasetService:
    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_uptake_to_mol_g(
        value: float | list[float] | None,
    ) -> float | list[float] | None:
        if value is None:
            return None
        if isinstance(value, list):
            return [val / 1000.0 for val in value]
        return float(value) / 1000.0

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_unit_series(series: pd.Series) -> pd.Series:
        return series.astype("string").str.strip().str.lower()

    # -------------------------------------------------------------------------
    def prepare_nist_dataframe(
        self,
        nist_df: pd.DataFrame,
        adsorbates_df: pd.DataFrame,
    ) -> pd.DataFrame:
        cleaned = nist_df.copy().rename(
            columns={
                "name": "filename",
                "adsorption_units": "adsorptionUnits",
                "pressure_units": "pressureUnits",
            }
        )
        required_cols = [
            "filename",
            "adsorbent",
            "adsorbate",
            "temperature",
            "pressure",
            "adsorbed_amount",
        ]
        missing = [column for column in required_cols if column not in cleaned.columns]
        if missing:
            raise ValueError(f"NIST dataset missing required columns: {missing}")

        cleaned = cleaned.dropna(subset=required_cols)
        cleaned["filename"] = cleaned["filename"].astype("string").str.strip()
        cleaned["adsorbent"] = (
            cleaned["adsorbent"].astype("string").str.strip().str.lower()
        )
        cleaned["adsorbate"] = (
            cleaned["adsorbate"].astype("string").str.strip().str.lower()
        )
        cleaned = cleaned[cleaned["filename"] != ""]
        cleaned = cleaned[cleaned["adsorbent"] != ""]
        cleaned = cleaned[cleaned["adsorbate"] != ""]

        if (
            not adsorbates_df.empty
            and "name" in adsorbates_df.columns
            and "molecular_weight" in adsorbates_df.columns
        ):
            weights = adsorbates_df[["name", "molecular_weight"]].copy()
            weights = weights.rename(
                columns={"molecular_weight": "adsorbate_molecular_weight"}
            )
            weights["name"] = weights["name"].astype("string").str.strip().str.lower()
            weights = weights.dropna(subset=["name"]).drop_duplicates(
                subset=["name"], keep="first"
            )
            cleaned = cleaned.merge(
                weights, left_on="adsorbate", right_on="name", how="left"
            )

        pressure_converter = PressureConversion()
        uptake_converter = UptakeConversion()
        valid_mask = pd.Series(True, index=cleaned.index)

        if "pressureUnits" in cleaned.columns:
            cleaned["pressureUnits"] = self.normalize_unit_series(
                cleaned["pressureUnits"]
            )
            valid_mask &= cleaned["pressureUnits"].isin(
                pressure_converter.conversions.keys()
            )

        if "adsorptionUnits" in cleaned.columns:
            cleaned["adsorptionUnits"] = self.normalize_unit_series(
                cleaned["adsorptionUnits"]
            )
            valid_mask &= cleaned["adsorptionUnits"].isin(
                uptake_converter.conversions.keys()
            )
            if "adsorbate_molecular_weight" in cleaned.columns:
                mol_weight = pd.to_numeric(
                    cleaned["adsorbate_molecular_weight"], errors="coerce"
                )
                requires_weight = cleaned["adsorptionUnits"].isin(
                    uptake_converter.weight_units
                )
                valid_mask &= ~requires_weight | mol_weight.notna()

        if not valid_mask.all():
            removed = int((~valid_mask).sum())
            logger.info("Filtered %s NIST rows due to unsupported units", removed)
        cleaned = cleaned.loc[valid_mask].copy()

        converted = PQ_units_conversion(cleaned)
        if "adsorbed_amount" in converted.columns:
            converted["adsorbed_amount"] = converted["adsorbed_amount"].apply(
                self.normalize_uptake_to_mol_g
            )

        for column in ("temperature", "pressure", "adsorbed_amount"):
            if column in converted.columns:
                converted[column] = pd.to_numeric(converted[column], errors="coerce")

        converted = converted.dropna(
            subset=["temperature", "pressure", "adsorbed_amount"]
        )
        converted = converted[converted["temperature"] > 0]
        converted = converted[converted["pressure"] >= 0]
        converted = converted[converted["adsorbed_amount"] >= 0]

        converted["experiment"] = (
            converted["filename"].astype("string").str.strip()
            + "_"
            + converted["adsorbent"].astype("string").str.strip()
            + "_"
            + converted["adsorbate"].astype("string").str.strip()
            + "_"
            + converted["temperature"].astype(str)
            + "K"
        )
        return converted

    # -------------------------------------------------------------------------
    def load_for_fitting(self) -> NISTFittingDatasetResponse:
        serializer = NISTDataSerializer()
        nist_df, adsorbates_df, _ = serializer.load_adsorption_datasets()
        if nist_df.empty:
            raise ValueError(
                "No NIST single-component data available. Please fetch data first."
            )

        converted_df = self.prepare_nist_dataframe(nist_df, adsorbates_df)
        if converted_df.empty:
            raise ValueError(
                "No valid NIST rows were available after unit normalization."
            )

        final_columns = {
            "pressure": DEFAULT_DATASET_COLUMN_MAPPING["pressure"],
            "adsorbed_amount": DEFAULT_DATASET_COLUMN_MAPPING["uptake"],
            "temperature": DEFAULT_DATASET_COLUMN_MAPPING["temperature"],
        }
        converted_df = converted_df.rename(columns=final_columns)

        required_cols = [
            "filename",
            "experiment",
            DEFAULT_DATASET_COLUMN_MAPPING["temperature"],
            DEFAULT_DATASET_COLUMN_MAPPING["pressure"],
            DEFAULT_DATASET_COLUMN_MAPPING["uptake"],
        ]
        available_cols = [column for column in required_cols if column in converted_df.columns]
        output_df = converted_df[available_cols].copy()
        output_df = output_df.where(pd.notna(output_df))
        records = [
            {key: (None if pd.isna(value) else value) for key, value in row.items()}
            for row in output_df.to_dict(orient="records")
        ]
        logger.info("Loaded %s NIST rows for fitting", output_df.shape[0])
        return NISTFittingDatasetResponse(
            status="success",
            dataset=NISTFittingDatasetPayload(
                dataset_name="nist_single_component",
                columns=list(output_df.columns),
                records=records,
                row_count=int(output_df.shape[0]),
            ),
        )
