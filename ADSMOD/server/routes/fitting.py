from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, status
from ADSMOD.server.schemas.fitting import FittingRequest, FittingResponse
from ADSMOD.server.utils.constants import (
    DEFAULT_DATASET_COLUMN_MAPPING,
    FITTING_NIST_DATASET_ENDPOINT,
    FITTING_ROUTER_PREFIX,
    FITTING_RUN_ENDPOINT,
)
from ADSMOD.server.utils.logger import logger
from ADSMOD.server.utils.repository.isodb import NISTDataSerializer
from ADSMOD.server.utils.services.conversion import (
    PQ_units_conversion,
    PressureConversion,
    UptakeConversion,
)
from ADSMOD.server.utils.services.fitting import FittingPipeline

router = APIRouter(prefix=FITTING_ROUTER_PREFIX, tags=["fitting"])


###############################################################################
class FittingEndpoint:
    """Endpoint for adsorption model fitting operations."""

    def __init__(self, router: APIRouter, pipeline: FittingPipeline) -> None:
        self.router = router
        self.pipeline = pipeline

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
        required_cols = [
            "filename",
            "adsorbent_name",
            "adsorbate_name",
            "temperature",
            "pressure",
            "adsorbed_amount",
        ]
        missing = [column for column in required_cols if column not in nist_df.columns]
        if missing:
            raise ValueError(f"NIST dataset missing required columns: {missing}")

        cleaned = nist_df.copy()
        cleaned = cleaned.dropna(subset=required_cols)
        cleaned["filename"] = cleaned["filename"].astype("string").str.strip()
        cleaned["adsorbent_name"] = (
            cleaned["adsorbent_name"].astype("string").str.strip().str.lower()
        )
        cleaned["adsorbate_name"] = (
            cleaned["adsorbate_name"].astype("string").str.strip().str.lower()
        )
        cleaned = cleaned[cleaned["filename"] != ""]
        cleaned = cleaned[cleaned["adsorbent_name"] != ""]
        cleaned = cleaned[cleaned["adsorbate_name"] != ""]

        if not adsorbates_df.empty and "name" in adsorbates_df.columns:
            weights = adsorbates_df[["name", "adsorbate_molecular_weight"]].copy()
            weights["name"] = (
                weights["name"].astype("string").str.strip().str.lower()
            )
            weights = weights.dropna(subset=["name"]).drop_duplicates(
                subset=["name"], keep="first"
            )
            cleaned = cleaned.merge(
                weights, left_on="adsorbate_name", right_on="name", how="left"
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
                converted[column] = pd.to_numeric(
                    converted[column], errors="coerce"
                )

        converted = converted.dropna(
            subset=["temperature", "pressure", "adsorbed_amount"]
        )
        converted = converted[converted["temperature"] > 0]
        converted = converted[converted["pressure"] >= 0]
        converted = converted[converted["adsorbed_amount"] >= 0]

        converted["experiment"] = (
            converted["filename"].astype("string").str.strip()
            + "_"
            + converted["adsorbent_name"].astype("string").str.strip()
            + "_"
            + converted["adsorbate_name"].astype("string").str.strip()
            + "_"
            + converted["temperature"].astype(str)
            + "K"
        )

        return converted

    # -------------------------------------------------------------------------
    async def run_fitting_job(self, payload: FittingRequest) -> Any:
        """Execute the fitting pipeline for the provided dataset and configuration."""
        logger.info(
            "Received fitting request: iterations=%s, method=%s",
            payload.max_iterations,
            payload.optimization_method,
        )

        try:
            response = await asyncio.to_thread(
                self.pipeline.run,
                payload.dataset.model_dump(),
                {
                    name: config.model_dump()
                    for name, config in payload.parameter_bounds.items()
                },
                payload.max_iterations,
                payload.optimization_method,
            )
        except ValueError as exc:
            logger.warning("Invalid fitting request: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        except Exception as exc:  # noqa: BLE001
            error_type = type(exc).__name__
            error_msg = str(exc).split("\n")[0][:120]
            logger.error("Fitting job failed: %s - %s", error_type, error_msg)
            logger.debug("Fitting job error details", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to complete the fitting job.",
            ) from exc

        logger.info(
            "Fitting job completed successfully with %s experiments",
            response.get("processed_rows"),
        )
        return response

    # -------------------------------------------------------------------------
    async def get_nist_dataset_for_fitting(self) -> Any:
        """Load NIST single-component data as a DatasetPayload for fitting.

        Applies unit conversion (pressure to Pa, uptake to mol/g) and builds
        experiment identifiers from filename + adsorbent + adsorbate + temperature.
        """
        try:
            serializer = NISTDataSerializer()
            nist_df, adsorbates_df, _ = serializer.load_adsorption_datasets()
            if nist_df.empty:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No NIST single-component data available. Please fetch data first.",
                )

            converted_df = self.prepare_nist_dataframe(nist_df, adsorbates_df)
            if converted_df.empty:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No valid NIST rows were available after unit normalization.",
                )

            final_columns = {
                "pressure": DEFAULT_DATASET_COLUMN_MAPPING["pressure"],
                "adsorbed_amount": DEFAULT_DATASET_COLUMN_MAPPING["uptake"],
                "temperature": DEFAULT_DATASET_COLUMN_MAPPING["temperature"],
            }
            converted_df = converted_df.rename(columns=final_columns)

            required_cols = [
                "experiment",
                DEFAULT_DATASET_COLUMN_MAPPING["temperature"],
                DEFAULT_DATASET_COLUMN_MAPPING["pressure"],
                DEFAULT_DATASET_COLUMN_MAPPING["uptake"],
            ]
            available_cols = [c for c in required_cols if c in converted_df.columns]
            output_df = converted_df[available_cols].copy()

            output_df = output_df.where(pd.notna(output_df), None)
            payload = {
                "status": "success",
                "dataset": {
                    "dataset_name": "nist_single_component",
                    "columns": list(output_df.columns),
                    "records": output_df.to_dict(orient="records"),
                    "row_count": int(output_df.shape[0]),
                },
            }
            logger.info("Loaded %s NIST rows for fitting", output_df.shape[0])
            return payload

        except HTTPException:
            raise
        except ValueError as exc:
            logger.warning("Invalid NIST dataset: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        except Exception as exc:  # noqa: BLE001
            error_type = type(exc).__name__
            error_msg = str(exc).split("\n")[0][:120]
            logger.error("NIST dataset load failed: %s - %s", error_type, error_msg)
            logger.debug("NIST dataset load error details", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to load NIST data.",
            ) from exc

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        """Register all fitting-related routes with the router."""
        self.router.add_api_route(
            FITTING_RUN_ENDPOINT,
            self.run_fitting_job,
            methods=["POST"],
            response_model=FittingResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            FITTING_NIST_DATASET_ENDPOINT,
            self.get_nist_dataset_for_fitting,
            methods=["GET"],
            status_code=status.HTTP_200_OK,
        )


###############################################################################
pipeline = FittingPipeline()
fitting_endpoint = FittingEndpoint(router=router, pipeline=pipeline)
fitting_endpoint.add_routes()
