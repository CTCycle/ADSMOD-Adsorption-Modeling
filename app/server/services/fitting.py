# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from server.configurations.settings import AdsmodConfig, MINIMIZE_MODEL_METRICS
from server.domain.fitting import (
    ExperimentFitRequest,
    ExperimentFitResponse,
    FitMetrics,
    FittedParameter,
    FittingDatasetReadiness,
    FittingExperimentSummary,
    FittingSessionDetail,
    FittingSessionListResponse,
    FittingSessionRequest,
    FittingSessionStartResponse,
    FittingSessionSummary,
    ModelCatalogResponse,
    ModelDefinition,
    ModelSettingsUpdate,
    ParameterBounds,
    SessionModelResult,
)
from server.services.modeling.fitting import (
    FittingCancelledError,
    MODEL_VERSION,
    FitComputation,
    FittingPipeline,
    display_parameter_value,
    parameter_unit,
    pressure_factor,
    raise_if_cancelled,
)
from server.services.modeling.models import AdsorptionModels, ModelSpec
from server.common.units import UnitRegistry
from server.common.utils.logger import logger
from server.domain.jobs import (
    JobCancelResponse,
    JobListResponse,
    JobStatusResponse,
)
from server.repositories.datasets import DatasetRepository
from server.repositories.fitting import FittingRepository, SessionQuery
from server.repositories.model_settings import ModelSettingsRepository
from server.repositories.schemas.models import FitResult, FittingSession
from server.services.job_responses import JobResponseFactory
from server.services.jobs import JobManager

METRIC_NAMES = (
    "sse",
    "rmse",
    "mae",
    "r_squared",
    "adjusted_r_squared",
    "chi_square",
    "aic",
    "aicc",
    "bic",
)
TERMINAL_JOB_STATES = {"completed", "failed", "cancelled"}
INTERRUPTED_MESSAGE = "Interrupted by application restart."

###############################################################################
@dataclass(frozen=True)
class DatasetExperiment:
    isotherm_id: int
    name: str
    series: dict[str, Any] | None
    reason: str | None

###############################################################################
class FittingService:
    JOB_TYPE = "fitting"

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        config: AdsmodConfig,
        datasets: DatasetRepository,
        results: FittingRepository,
        model_settings: ModelSettingsRepository,
        job_manager: JobManager | None = None,
        pipeline: FittingPipeline | None = None,
    ) -> None:
        self.config = config
        self.datasets = datasets
        self.results = results
        self.model_settings = model_settings
        self.job_manager = job_manager or JobManager(logger=logger)
        self.pipeline = pipeline or FittingPipeline()
        self._start_lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._session_ids_by_job_id: dict[str, int] = {}
        self._job_ids_by_session_id: dict[int, str] = {}

    # Global model catalog ----------------------------------------------------
    # -------------------------------------------------------------------------
    @staticmethod
    def _spec(model_key: str) -> ModelSpec:
        try:
            return AdsorptionModels.get_spec(model_key)
        except ValueError as exc:
            raise LookupError(str(exc)) from exc

    # -------------------------------------------------------------------------
    def _stored_bounds(self) -> tuple[dict[str, bool], dict[str, dict[str, ParameterBounds]]]:
        stored = self.model_settings.all()
        enabled = {key: item.enabled for key, item in stored.items()}
        bounds = {
            key: {
                name: ParameterBounds.model_validate(values)
                for name, values in item.parameter_bounds.items()
            }
            for key, item in stored.items()
        }
        return enabled, bounds

    # -------------------------------------------------------------------------
    def model_catalog(self) -> ModelCatalogResponse:
        enabled, bounds = self._stored_bounds()
        return ModelCatalogResponse(
            models=self.pipeline.catalog(enabled=enabled, bounds=bounds)
        )

    # -------------------------------------------------------------------------
    def _catalog_entry(self, model_key: str) -> ModelDefinition:
        return next(
            item for item in self.model_catalog().models if item.key == model_key
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_bounds(
        spec: ModelSpec,
        parameters: dict[str, ParameterBounds],
        *,
        prune_canonical: bool = True,
    ) -> dict[str, dict[str, float]]:
        known = {parameter.name: parameter for parameter in spec.parameters}
        unknown = sorted(set(parameters) - set(known))
        if unknown:
            raise ValueError(
                f"{spec.name} has no parameter(s) named {', '.join(unknown)}."
            )
        customized: dict[str, dict[str, float]] = {}
        for name, bounds in parameters.items():
            if bounds.lower <= 0:
                raise ValueError(
                    f"{spec.name} parameter '{name}' requires a strictly positive lower bound."
                )
            canonical = known[name]
            if (
                prune_canonical
                and math.isclose(bounds.lower, canonical.lower)
                and math.isclose(bounds.upper, canonical.upper)
            ):
                continue
            customized[name] = {"lower": bounds.lower, "upper": bounds.upper}
        return customized

    # -------------------------------------------------------------------------
    def update_model(
        self, model_key: str, update: ModelSettingsUpdate
    ) -> ModelDefinition:
        spec = self._spec(model_key)
        self.model_settings.save(
            spec.key,
            enabled=update.enabled,
            parameter_bounds=self._validate_bounds(spec, update.parameters),
        )
        return self._catalog_entry(spec.key)

    # -------------------------------------------------------------------------
    def reset_model(self, model_key: str) -> ModelDefinition:
        spec = self._spec(model_key)
        enabled, _ = self._stored_bounds()
        self.model_settings.save(
            spec.key, enabled=enabled.get(spec.key, True), parameter_bounds={}
        )
        return self._catalog_entry(spec.key)

    # Dataset readiness -------------------------------------------------------
    # -------------------------------------------------------------------------
    def _load_dataset(self, dataset_id: int) -> tuple[str, list[DatasetExperiment]]:
        dataset_name, identities = self.datasets.experiment_identities(dataset_id)
        experiments: list[DatasetExperiment] = []
        for isotherm_id, name in identities:
            try:
                series = self.datasets.fitting_series(dataset_id, isotherm_id)
            except ValueError as exc:
                experiments.append(DatasetExperiment(isotherm_id, name, None, str(exc)))
                continue
            experiments.append(DatasetExperiment(isotherm_id, name, series, None))
        return dataset_name, experiments

    # -------------------------------------------------------------------------
    @staticmethod
    def _model_incompatibility(
        experiments: list[DatasetExperiment],
    ) -> dict[str, str]:
        """Explain which models cannot be fitted to every eligible experiment."""
        loaded = [item for item in experiments if item.series is not None]
        reasons: dict[str, str] = {}
        for key in AdsorptionModels.model_names:
            spec = AdsorptionModels.get_spec(key)
            for item in loaded:
                series = item.series or {}
                pressure = series["pressure"]
                reason = None
                if len(pressure) <= len(spec.parameters):
                    reason = (
                        f"Needs more than {len(spec.parameters)} measurements per "
                        f"experiment; '{item.name}' has {len(pressure)}."
                    )
                elif key in {"freundlich", "temkin"} and any(
                    value <= 0 for value in pressure
                ):
                    reason = (
                        f"Requires strictly positive pressure; '{item.name}' "
                        "contains a zero-pressure point."
                    )
                elif key == "dubinin_radushkevich" and not (
                    series["pressure_basis"] == "relative"
                    or (series.get("saturation_pressure_pa") or 0) > 0
                ):
                    reason = (
                        "Requires relative pressure or a saturation pressure; "
                        f"'{item.name}' has neither."
                    )
                if reason is not None:
                    reasons[key] = reason
                    break
        return reasons

    # -------------------------------------------------------------------------
    @staticmethod
    def _uncertainty_complete(series: dict[str, Any]) -> bool:
        values = series.get("uptake_stddev") or []
        return bool(values) and all(value is not None and value > 0 for value in values)

    # -------------------------------------------------------------------------
    def dataset_readiness(self, dataset_id: int) -> FittingDatasetReadiness:
        dataset_name, experiments = self._load_dataset(dataset_id)
        eligible = [item for item in experiments if item.series is not None]
        return FittingDatasetReadiness(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            experiment_count=len(experiments),
            eligible_count=len(eligible),
            ready=bool(experiments) and len(eligible) == len(experiments),
            inverse_sigma_available=bool(eligible)
            and all(self._uncertainty_complete(item.series or {}) for item in eligible),
            model_incompatibility=self._model_incompatibility(experiments),
            experiments=[
                FittingExperimentSummary(
                    isotherm_id=item.isotherm_id,
                    name=item.name,
                    eligible=item.series is not None,
                    reason=item.reason,
                )
                for item in experiments
            ],
        )

    # Session execution -------------------------------------------------------
    # -------------------------------------------------------------------------
    def _resolve_models(
        self, payload: FittingSessionRequest
    ) -> tuple[list[str], dict[str, dict[str, ParameterBounds]], dict[str, dict[str, ParameterBounds]]]:
        enabled, global_bounds = self._stored_bounds()
        keys: list[str] = []
        for model in payload.models:
            spec = self._spec(model)
            if not enabled.get(spec.key, True):
                raise ValueError(
                    f"{spec.name} is disabled in the adsorption model catalog."
                )
            if spec.key not in keys:
                keys.append(spec.key)
        overrides: dict[str, dict[str, ParameterBounds]] = {}
        for model, parameters in payload.parameter_overrides.items():
            spec = self._spec(model)
            if spec.key not in keys:
                raise ValueError(
                    f"Parameter overrides were given for {spec.name}, which is not selected."
                )
            # Session overrides are explicit: keep them even when they equal canonical bounds.
            validated = self._validate_bounds(spec, parameters, prune_canonical=False)
            overrides[spec.key] = {
                name: ParameterBounds(**values) for name, values in validated.items()
            }
        effective: dict[str, dict[str, ParameterBounds]] = {}
        for key in keys:
            spec = AdsorptionModels.get_spec(key)
            effective[key] = {
                parameter.name: overrides.get(key, {}).get(parameter.name)
                or global_bounds.get(key, {}).get(parameter.name)
                or ParameterBounds(lower=parameter.lower, upper=parameter.upper)
                for parameter in spec.parameters
            }
        return keys, effective, overrides

    # -------------------------------------------------------------------------
    def start_session(self, payload: FittingSessionRequest) -> FittingSessionStartResponse:
        fitting_config = self.config.application.fitting
        poll_interval = self.config.application.jobs.polling_interval
        if payload.max_evaluations > fitting_config.max_iterations_upper_bound:
            raise ValueError(
                "max_evaluations exceeds the configured maximum iteration ceiling."
            )
        keys, effective, overrides = self._resolve_models(payload)
        with self._start_lock:
            if self.job_manager.is_job_running(self.JOB_TYPE):
                raise ValueError("A fitting session is already running.")
            dataset_name, experiments = self._load_dataset(payload.dataset_id)
            if not experiments:
                raise ValueError("The selected dataset has no experiments to fit.")
            ineligible = [item for item in experiments if item.series is None]
            if ineligible and not payload.skip_ineligible:
                details = "; ".join(f"{item.name}: {item.reason}" for item in ineligible)
                raise ValueError(
                    f"{len(ineligible)} experiment(s) cannot be fitted. {details}"
                )
            if len(ineligible) == len(experiments):
                raise ValueError("No experiment in the dataset can be fitted.")
            incompatible = self._model_incompatibility(experiments)
            blocked = [key for key in keys if key in incompatible]
            if blocked:
                raise ValueError(
                    "; ".join(
                        f"{AdsorptionModels.get_spec(key).name}: {incompatible[key]}"
                        for key in blocked
                    )
                )
            series_list = [item.series for item in experiments if item.series is not None]
            if payload.weighting == "inverse_sigma" and not all(
                self._uncertainty_complete(series) for series in series_list
            ):
                raise ValueError(
                    "Inverse-sigma weighting requires complete positive uncertainties "
                    "for every experiment in the dataset."
                )
            hashes = [self.pipeline.input_hash(series) for series in series_list]
            configuration = {
                "models": keys,
                "optimizer": payload.optimizer,
                "weighting": payload.weighting,
                "max_evaluations": payload.max_evaluations,
                "model_bounds": {
                    key: {name: bounds.model_dump() for name, bounds in values.items()}
                    for key, values in effective.items()
                },
                "session_overrides": {
                    key: {name: bounds.model_dump() for name, bounds in values.items()}
                    for key, values in overrides.items()
                },
                "bounds_unit_system": {"pressure": "Pa", "uptake": "mol/kg"},
                "display_units": payload.display_units.model_dump(),
                "best_model_metric": fitting_config.best_model_metric,
                "model_version": MODEL_VERSION,
                "experiments": [
                    {"isotherm_id": series["isotherm_id"], "name": series["isotherm_name"], "input_sha256": digest}
                    for series, digest in zip(series_list, hashes, strict=True)
                ],
                "skipped_experiments": [
                    {"isotherm_id": item.isotherm_id, "name": item.name, "reason": item.reason}
                    for item in ineligible
                ],
            }
            session_id = self.results.create_session(
                name=payload.name,
                dataset_id=payload.dataset_id,
                dataset_name=dataset_name,
                isotherm_ids=[series["isotherm_id"] for series in series_list],
                models=keys,
                input_sha256=hashlib.sha256("".join(hashes).encode("ascii")).hexdigest(),
                optimizer=payload.optimizer,
                weighting=payload.weighting,
                max_evaluations=payload.max_evaluations,
                pressure_display_unit=payload.display_units.pressure,
                uptake_display_unit=payload.display_units.uptake,
                configuration=configuration,
            )
            try:
                job_id = self.job_manager.start_job(
                    job_type=self.JOB_TYPE,
                    runner=self._run_session_sync,
                    args=(session_id, configuration, series_list),
                )
            except Exception as exc:
                self.results.fail_session(session_id, str(exc))
                raise
            with self._lifecycle_lock:
                self._session_ids_by_job_id[job_id] = session_id
                self._job_ids_by_session_id[session_id] = job_id
            return FittingSessionStartResponse(
                session_id=session_id, job_id=job_id, poll_interval=poll_interval
            )

    # -------------------------------------------------------------------------
    @staticmethod
    def _persistence_record(
        computation: FitComputation, series: dict[str, Any]
    ) -> dict[str, Any]:
        metrics = computation.metrics.values
        pressure_basis = series["pressure_basis"]
        parameters: list[dict[str, Any]] = []
        if computation.parameters is not None:
            related = {
                parameter.name: computation.parameters[index]
                for index, parameter in enumerate(computation.spec.parameters)
            }
            related_value = float(related.get("n", related.get("beta", 1.0)))
            for index, parameter in enumerate(computation.spec.parameters):
                parameters.append(
                    {
                        "name": parameter.name,
                        "position": index,
                        "value_canonical": float(computation.parameters[index]),
                        "standard_error_canonical": (
                            float(computation.standard_errors[index])
                            if computation.standard_errors is not None
                            else None
                        ),
                        "ci95_low_canonical": (
                            float(computation.ci95_low[index])
                            if computation.ci95_low is not None
                            else None
                        ),
                        "ci95_high_canonical": (
                            float(computation.ci95_high[index])
                            if computation.ci95_high is not None
                            else None
                        ),
                        "unit_canonical": parameter_unit(
                            parameter,
                            pressure_unit="1" if pressure_basis == "relative" else "Pa",
                            uptake_unit="mol/kg",
                            related_value=related_value,
                        ),
                    }
                )
        return {
            "isotherm_id": series["isotherm_id"],
            "isotherm_name": series["isotherm_name"],
            "model_name": computation.spec.key,
            "model_version": MODEL_VERSION,
            "status": computation.status,
            "convergence_message": computation.message,
            "function_evaluations": computation.function_evaluations,
            "jacobian_rank": computation.jacobian_rank,
            "condition_number": computation.condition_number,
            "observation_count": len(series["pressure"]),
            "parameter_count": len(computation.spec.parameters),
            **metrics,
            "predicted_observations": [float(value) for value in computation.predicted],
            "predicted_curve": [
                {
                    "pressure": float(pressure),
                    "pressure_unit": "1" if pressure_basis == "relative" else "Pa",
                    "uptake_mol_kg": float(uptake),
                }
                for pressure, uptake in zip(
                    computation.curve_pressure,
                    computation.curve_uptake,
                    strict=True,
                )
            ],
            "warnings": computation.warnings,
            "rank": computation.rank,
            "parameters": parameters,
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def _rank(computations: list[FitComputation], metric: str) -> str:
        """Rank by `metric`, falling back to RMSE then R² when it is undefined for all models.

        Returns the metric actually used so the caller can note a fallback.
        """
        for item in computations:
            item.rank = None
        for candidate in dict.fromkeys((metric, "rmse", "r_squared")):
            ranked = [
                item
                for item in computations
                if item.status != "failed"
                and item.metrics.values.get(candidate) is not None
                and np.isfinite(float(item.metrics.values[candidate]))
            ]
            if ranked:
                break
        else:
            return metric
        ranked.sort(
            key=lambda item: float(item.metrics.values[candidate]),
            reverse=candidate not in MINIMIZE_MODEL_METRICS,
        )
        for rank, item in enumerate(ranked, start=1):
            item.rank = rank
        return candidate

    # -------------------------------------------------------------------------
    def _publish_progress(
        self, job_id: str | None, session_id: int, log: list[str], progress: float
    ) -> None:
        if job_id is None:
            return
        self.job_manager.update_progress(job_id, progress)
        self.job_manager.update_result(job_id, {"session_id": session_id, "log": list(log)})

    # -------------------------------------------------------------------------
    def _run_session_sync(
        self,
        session_id: int,
        configuration: dict[str, Any],
        series_list: list[dict[str, Any]],
        stop_event: Any | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        request = ExperimentFitRequest.model_validate(
            {
                key: configuration[key]
                for key in ("models", "optimizer", "max_evaluations", "weighting", "model_bounds")
            }
        )
        metric = configuration["best_model_metric"]
        log: list[str] = [
            f"[INFO] Fitting {len(request.models)} model(s) to {len(series_list)} experiment(s)."
        ]

        self._publish_progress(job_id, session_id, log, 0.0)
        records: list[dict[str, Any]] = []
        statuses: list[str] = []
        try:
            for index, series in enumerate(series_list, start=1):
                raise_if_cancelled(stop_event)
                log.append(f"[INFO] ({index}/{len(series_list)}) {series['isotherm_name']}")
                self._publish_progress(
                    job_id, session_id, log, 100.0 * (index - 1) / len(series_list)
                )
                computations = self.pipeline.run(series, request, stop_event=stop_event)
                used_metric = self._rank(computations, metric)
                if used_metric != metric:
                    log.append(
                        f"[WARN]   {metric.upper()} is undefined for every model "
                        f"(too few measurements); best model chosen by {used_metric}."
                    )
                for item in computations:
                    statuses.append(item.status)
                    if item.status == "failed":
                        log.append(f"[WARN]   {item.spec.name}: {item.message}")
                best = next((item.spec.name for item in computations if item.rank == 1), None)
                log.append(f"[INFO]   Best model: {best or 'none'}")
                records.extend(self._persistence_record(item, series) for item in computations)
            status, summary = self._session_outcome(statuses, len(series_list))
            log.append(f"[{'ERROR' if status == 'failed' else 'INFO'}] {summary}")
            with self._lifecycle_lock:
                raise_if_cancelled(stop_event)
                self.results.complete_session(
                    session_id, status=status, message=summary, results=records, log=log
                )
            self._publish_progress(job_id, session_id, log, 100.0)
            return {"session_id": session_id, "status": status, "summary": summary, "log": log}
        except FittingCancelledError as exc:
            log.append(f"[WARN] {exc}")
            with self._lifecycle_lock:
                self.results.cancel_session(session_id, str(exc), log)
            raise
        except Exception as exc:
            log.append(f"[ERROR] {exc}")
            with self._lifecycle_lock:
                if stop_event is not None and stop_event.is_set():
                    self.results.cancel_session(session_id, log=log)
                else:
                    self.results.fail_session(session_id, str(exc), log)
            raise

    # -------------------------------------------------------------------------
    @staticmethod
    def _session_outcome(statuses: list[str], experiment_count: int) -> tuple[str, str]:
        fitted = sum(status != "failed" for status in statuses)
        warned = statuses.count("warning")
        status = (
            "failed"
            if fitted == 0
            else "warning"
            if warned or fitted < len(statuses)
            else "completed"
        )
        summary = (
            f"{fitted} of {len(statuses)} model fits succeeded across "
            f"{experiment_count} experiment(s)"
            + (f", {warned} with warnings." if warned else ".")
        )
        return status, summary

    # -------------------------------------------------------------------------
    def fail_interrupted_sessions(self) -> int:
        return self.results.fail_interrupted_sessions(INTERRUPTED_MESSAGE)

    # Jobs ----------------------------------------------------------------------
    # -------------------------------------------------------------------------
    def _forget_job(self, job_id: str) -> None:
        session_id = self._session_ids_by_job_id.pop(job_id, None)
        if session_id is not None:
            self._job_ids_by_session_id.pop(session_id, None)

    # -------------------------------------------------------------------------
    def get_job_status(self, job_id: str) -> JobStatusResponse:
        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise LookupError(f"Job {job_id} not found.")
        if job_status["status"] in TERMINAL_JOB_STATES:
            with self._lifecycle_lock:
                self._forget_job(job_id)
        return JobResponseFactory.status(
            job_status=job_status,
            poll_interval=self.config.application.jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def list_jobs(self) -> JobListResponse:
        return JobResponseFactory.list(
            job_statuses=self.job_manager.list_jobs(self.JOB_TYPE),
            poll_interval=self.config.application.jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def cancel_job(self, job_id: str) -> JobCancelResponse:
        with self._lifecycle_lock:
            session_id = self._session_ids_by_job_id.get(job_id)
            if session_id is not None and not self.results.is_session_running(session_id):
                raise ValueError(
                    f"Job {job_id} cannot be cancelled (not found or already completed)."
                )
            if not self.job_manager.cancel_job(job_id):
                raise ValueError(
                    f"Job {job_id} cannot be cancelled (not found or already completed)."
                )
            if session_id is not None:
                self.results.cancel_session(session_id)
                self._forget_job(job_id)
            return JobResponseFactory.cancelled(job_id)

    # Session reads -------------------------------------------------------------
    # -------------------------------------------------------------------------
    @staticmethod
    def _best_models(record: FittingSession) -> dict[str, str | None]:
        best: dict[str, str | None] = {}
        for result in record.results:
            best.setdefault(result.isotherm_name, None)
            if result.rank == 1:
                best[result.isotherm_name] = result.model_name
        return best

    # -------------------------------------------------------------------------
    def _summary_fields(self, record: FittingSession) -> dict[str, Any]:
        duration = (
            (record.completed_at - record.created_at).total_seconds()
            if record.completed_at is not None
            else None
        )
        return {
            "id": record.id,
            "name": record.name,
            "dataset_id": record.dataset_id,
            "dataset_name": record.dataset_name,
            "status": record.status,
            "message": record.message,
            "models": [item.model_key for item in record.models],
            "experiment_count": len(record.isotherm_ids),
            "optimizer": record.optimizer,
            "weighting": record.weighting,
            "max_evaluations": record.max_evaluations,
            "created_at": record.created_at.isoformat(),
            "completed_at": record.completed_at.isoformat() if record.completed_at else None,
            "duration_seconds": duration,
            "best_models": self._best_models(record),
        }

    # -------------------------------------------------------------------------
    def list_sessions(self, query: SessionQuery) -> FittingSessionListResponse:
        records, total = self.results.list_sessions(query)
        return FittingSessionListResponse(
            items=[FittingSessionSummary(**self._summary_fields(item)) for item in records],
            total=total,
            page=query.page,
            page_size=query.page_size,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _display_result(
        result: FitResult, pressure_unit: str, uptake_unit: str
    ) -> SessionModelResult:
        spec = AdsorptionModels.get_spec(result.model_name)
        relative = bool(result.predicted_curve) and result.predicted_curve[0].get("pressure_unit") == "1"
        shown_pressure = "1" if relative else pressure_unit
        p_factor = pressure_factor(shown_pressure, "relative" if relative else "absolute")
        _, q_factor, _ = UnitRegistry.uptake_factor_to_mol_kg(uptake_unit, None)
        stored = {item.name: item for item in result.parameters}
        related_param = stored.get("n") or stored.get("beta")
        related = float(related_param.value_canonical) if related_param else 1.0
        parameters: list[FittedParameter] = []
        for parameter in spec.parameters:
            item = stored.get(parameter.name)
            if item is None:
                continue

            shown = partial(
                display_parameter_value,
                parameter=parameter,
                pressure_factor_value=p_factor,
                uptake_factor_value=q_factor,
                related_value=related,
            )
            parameters.append(
                FittedParameter(
                    name=parameter.name,
                    label=parameter.label,
                    value=float(shown(item.value_canonical) or 0.0),
                    standard_error=shown(item.standard_error_canonical),
                    ci95_low=shown(item.ci95_low_canonical),
                    ci95_high=shown(item.ci95_high_canonical),
                    unit=parameter_unit(
                        parameter,
                        pressure_unit=shown_pressure,
                        uptake_unit=uptake_unit,
                        related_value=related,
                    ),
                )
            )
        return SessionModelResult(
            model=result.model_name,
            name=spec.name,
            status=result.status,  # type: ignore[arg-type]
            convergence_message=result.convergence_message,
            rank=result.rank,
            observation_count=result.observation_count,
            parameters=parameters,
            metrics=FitMetrics(**{name: getattr(result, name) for name in METRIC_NAMES}),
            warnings=list(result.warnings),
        )

    # -------------------------------------------------------------------------
    def get_session(self, session_id: int) -> FittingSessionDetail:
        record = self.results.get_session(session_id)
        grouped: dict[str, list[FitResult]] = {}
        identities: dict[str, int | None] = {}
        fitted_order = [item.get("name") for item in record.configuration.get("experiments", [])]
        for result in sorted(
            record.results,
            key=lambda item: (
                fitted_order.index(item.isotherm_name)
                if item.isotherm_name in fitted_order
                else len(fitted_order),
                item.id,
            ),
        ):
            grouped.setdefault(result.isotherm_name, []).append(result)
            identities[result.isotherm_name] = result.isotherm_id
        experiments = [
            ExperimentFitResponse(
                isotherm_id=identities[name],
                experiment_name=name,
                pressure_unit=record.pressure_display_unit,
                uptake_unit=record.uptake_display_unit,
                observation_count=results[0].observation_count,
                best_model=next((item.model_name for item in results if item.rank == 1), None),
                results=[
                    self._display_result(
                        item, record.pressure_display_unit, record.uptake_display_unit
                    )
                    for item in sorted(results, key=lambda item: (item.rank is None, item.rank or 0, item.model_name))
                ],
            )
            for name, results in grouped.items()
        ]
        stale, stale_reason = self._session_staleness(record)
        return FittingSessionDetail(
            **self._summary_fields(record),
            configuration=record.configuration,
            log=list(record.log),
            experiments=experiments,
            job_id=self._job_ids_by_session_id.get(record.id),
            stale=stale,
            stale_reason=stale_reason,
        )

    # -------------------------------------------------------------------------
    def _session_staleness(self, record: FittingSession) -> tuple[bool, str | None]:
        """Compare stored experiment fingerprints with the current dataset content."""
        for stored in record.configuration.get("experiments", []):
            name = stored.get("name")
            try:
                series = self.datasets.fitting_series(
                    record.dataset_id, int(stored["isotherm_id"])
                )
            except (LookupError, ValueError):
                return True, f"Experiment '{name}' is no longer available or fittable."
            if self.pipeline.input_hash(series) != stored.get("input_sha256"):
                return True, (
                    f"Experiment '{name}' changed after this session was fitted; "
                    "results may no longer match the dataset."
                )
        return False, None

    # -------------------------------------------------------------------------
    def delete_session(self, session_id: int) -> None:
        self.results.delete_session(session_id)

