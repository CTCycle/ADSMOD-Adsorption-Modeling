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
from ADSMOD.server.utils.services.conversion import PQ_units_conversion
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
            logger.exception("ADSMOD fitting job failed")
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

            if not adsorbates_df.empty and "name" in adsorbates_df.columns:
                nist_df = nist_df.merge(
                    adsorbates_df[["name", "adsorbate_molecular_weight"]],
                    left_on="adsorbate_name",
                    right_on="name",
                    how="left",
                )

            converted_df = PQ_units_conversion(nist_df.copy())

            if "adsorbed_amount" in converted_df.columns:
                converted_df["adsorbed_amount"] = converted_df["adsorbed_amount"].apply(
                    self.normalize_uptake_to_mol_g
                )

            converted_df["experiment"] = (
                converted_df["filename"].astype(str)
                + "_"
                + converted_df["adsorbent_name"].astype(str)
                + "_"
                + converted_df["adsorbate_name"].astype(str)
                + "_"
                + converted_df["temperature"].astype(str)
                + "K"
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
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load NIST dataset for fitting")
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
