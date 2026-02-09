from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, status

from ADSMOD.server.entities.fitting import FittingRequest
from ADSMOD.server.entities.jobs import (
    JobListResponse,
    JobStartResponse,
    JobStatusResponse,
)
from ADSMOD.server.configurations import server_settings
from ADSMOD.server.common.constants import (
    DEFAULT_DATASET_COLUMN_MAPPING,
    FITTING_JOBS_ENDPOINT,
    FITTING_JOB_STATUS_ENDPOINT,
    FITTING_NIST_DATASET_ENDPOINT,
    FITTING_ROUTER_PREFIX,
    FITTING_RUN_ENDPOINT,
)
from ADSMOD.server.common.utils.logger import logger
from ADSMOD.server.repositories.isodb import NISTDataSerializer
from ADSMOD.server.services.data.conversion import (
    PQ_units_conversion,
    PressureConversion,
    UptakeConversion,
)
from ADSMOD.server.services.modeling.fitting import FittingPipeline
from ADSMOD.server.services.jobs import job_manager

router = APIRouter(prefix=FITTING_ROUTER_PREFIX, tags=["fitting"])


###############################################################################
class FittingEndpoint:
    JOB_TYPE = "fitting"

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

        cleaned = nist_df.copy().rename(
            columns={
                "name": "filename",
                "adsorption_units": "adsorptionUnits",
                "pressure_units": "pressureUnits",
            }
        )
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
            + converted["adsorbent_name"].astype("string").str.strip()
            + "_"
            + converted["adsorbate_name"].astype("string").str.strip()
            + "_"
            + converted["temperature"].astype(str)
            + "K"
        )

        return converted

    # -------------------------------------------------------------------------
    def _run_fitting_sync(
        self,
        dataset_dict: dict,
        parameter_bounds_dict: dict,
        max_iterations: int,
        optimization_method: str,
    ) -> dict:
        return self.pipeline.run(
            dataset_dict,
            parameter_bounds_dict,
            max_iterations,
            optimization_method,
        )

    # -------------------------------------------------------------------------
    def start_fitting_job(self, payload: FittingRequest) -> JobStartResponse:
        if job_manager.is_job_running(self.JOB_TYPE):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A fitting job is already running.",
            )

        logger.info(
            "Received fitting request: iterations=%s, method=%s",
            payload.max_iterations,
            payload.optimization_method,
        )

        dataset_dict = payload.dataset.model_dump()
        parameter_bounds_dict = {
            name: config.model_dump()
            for name, config in payload.parameter_bounds.items()
        }

        job_id = job_manager.start_job(
            job_type=self.JOB_TYPE,
            runner=self._run_fitting_sync,
            args=(
                dataset_dict,
                parameter_bounds_dict,
                payload.max_iterations,
                payload.optimization_method,
            ),
        )
        logger.info("Started fitting job %s", job_id)
        return JobStartResponse(
            job_id=job_id,
            job_type=self.JOB_TYPE,
            status="running",
            message="Fitting job started.",
            poll_interval=server_settings.jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def get_job_status(self, job_id: str) -> JobStatusResponse:
        job_status = job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found.",
            )
        return JobStatusResponse(
            job_id=job_status["job_id"],
            job_type=job_status["job_type"],
            status=job_status["status"],
            progress=job_status["progress"],
            result=job_status["result"],
            error=job_status["error"],
            poll_interval=server_settings.jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def list_jobs(self) -> JobListResponse:
        all_jobs = job_manager.list_jobs(self.JOB_TYPE)
        return JobListResponse(
            jobs=[
                JobStatusResponse(
                    job_id=j["job_id"],
                    job_type=j["job_type"],
                    status=j["status"],
                    progress=j["progress"],
                    result=j["result"],
                    error=j["error"],
                    poll_interval=server_settings.jobs.polling_interval,
                )
                for j in all_jobs
            ]
        )

    # -------------------------------------------------------------------------
    def cancel_job(self, job_id: str) -> dict:
        success = job_manager.cancel_job(job_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job {job_id} cannot be cancelled (not found or already completed).",
            )
        return {"status": "cancelled", "job_id": job_id}

    # -------------------------------------------------------------------------
    def get_nist_dataset_for_fitting(self) -> Any:
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
                "filename",
                "experiment",
                DEFAULT_DATASET_COLUMN_MAPPING["temperature"],
                DEFAULT_DATASET_COLUMN_MAPPING["pressure"],
                DEFAULT_DATASET_COLUMN_MAPPING["uptake"],
            ]
            available_cols = [c for c in required_cols if c in converted_df.columns]
            output_df = converted_df[available_cols].copy()

            output_df = output_df.where(pd.notna(output_df))
            records = [
                {key: (None if pd.isna(value) else value) for key, value in row.items()}
                for row in output_df.to_dict(orient="records")
            ]
            payload = {
                "status": "success",
                "dataset": {
                    "dataset_name": "nist_single_component",
                    "columns": list(output_df.columns),
                    "records": records,
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
        self.router.add_api_route(
            FITTING_RUN_ENDPOINT,
            self.start_fitting_job,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            FITTING_NIST_DATASET_ENDPOINT,
            self.get_nist_dataset_for_fitting,
            methods=["GET"],
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            FITTING_JOBS_ENDPOINT,
            self.list_jobs,
            methods=["GET"],
            response_model=JobListResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            FITTING_JOB_STATUS_ENDPOINT,
            self.get_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            FITTING_JOB_STATUS_ENDPOINT,
            self.cancel_job,
            methods=["DELETE"],
            status_code=status.HTTP_200_OK,
        )


###############################################################################
pipeline = FittingPipeline()
fitting_endpoint = FittingEndpoint(router=router, pipeline=pipeline)
fitting_endpoint.add_routes()
