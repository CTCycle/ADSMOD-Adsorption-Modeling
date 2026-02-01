from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize

from ADSMOD.server.configurations import server_settings
from ADSMOD.server.utils.constants import (
    COLUMN_AIC,
    COLUMN_AICC,
    COLUMN_BEST_MODEL,
    COLUMN_DATASET_NAME,
    COLUMN_EXPERIMENT,
    COLUMN_EXPERIMENT_NAME,
    COLUMN_FILENAME,
    COLUMN_MAX_PRESSURE,
    COLUMN_MAX_UPTAKE,
    COLUMN_MEASUREMENT_COUNT,
    COLUMN_MIN_PRESSURE,
    COLUMN_MIN_UPTAKE,
    COLUMN_SCORE,
    COLUMN_WORST_MODEL,
    MODEL_PARAMETER_DEFAULTS,
)
from ADSMOD.server.utils.logger import logger
from ADSMOD.server.repositories.serializer import DataSerializer
from ADSMOD.server.services.modeling.models import AdsorptionModels
from ADSMOD.server.services.data.processing import (
    AdsorptionDataProcessor,
    DatasetAdapter,
    DatasetColumns,
)

SUPPORTED_OPTIMIZATION_METHODS: tuple[str, ...] = (
    "LSS",
    "BFGS",
    "L-BFGS-B",
    "Nelder-Mead",
    "Powell",
)
BOUNDS_COMPATIBLE_METHODS = {"L-BFGS-B", "Powell"}
DEFAULT_OPTIMIZATION_METHOD = "LSS"

PARAMETER_ALIAS_MAP: dict[str, dict[str, str]] = {
    model_name: {} for model_name in MODEL_PARAMETER_DEFAULTS
}
PARAMETER_ALIAS_MAP["Langmuir"].update(
    {
        "qm": "qsat",
        "b": "k",
    }
)
PARAMETER_ALIAS_MAP["Freundlich"].update(
    {
        "Kf": "k",
        "n": "exponent",
    }
)
PARAMETER_ALIAS_MAP["Sips"].update(
    {
        "qm": "qsat",
        "b": "k",
        "n": "exponent",
    }
)


###############################################################################
class ModelSolver:
    def __init__(self) -> None:
        self.collection = AdsorptionModels()

    # -------------------------------------------------------------------------
    @staticmethod
    def compute_information_metrics(
        score: float,
        sample_size: int,
        parameter_count: int,
    ) -> tuple[float, float]:
        valid_dimensions = (
            parameter_count > 0 and sample_size > 0 and np.isfinite(score)
        )
        if not valid_dimensions:
            return (np.inf, np.inf)

        safe_score = max(float(score), float(np.finfo(np.float64).tiny))
        ratio = safe_score / float(sample_size)
        aic = float(sample_size * np.log(ratio) + sample_size + 2.0 * parameter_count)
        if not np.isfinite(aic):
            return (np.inf, np.inf)

        correction_denominator = sample_size - parameter_count - 1
        if correction_denominator <= 0:
            return (aic, np.inf)

        aicc = float(
            aic
            + (2.0 * parameter_count * (parameter_count + 1.0)) / correction_denominator
        )
        return (aic, aicc)

    # -------------------------------------------------------------------------
    def single_experiment_fit(
        self,
        pressure: np.ndarray,
        uptake: np.ndarray,
        experiment_name: str,
        configuration: dict[str, Any],
        max_iterations: int,
        optimization_method: str,
    ) -> dict[str, dict[str, Any]]:
        """Fit every configured model against a single experiment dataset.

        Keyword arguments:
        pressure -- Pressure observations expressed as a NumPy array.
        uptake -- Measured uptakes corresponding to the pressure values.
        experiment_name -- Identifier of the current experiment, used for logging.
        configuration -- Per-model fitting configuration, including bounds and initial
        guesses.
        max_iterations -- Maximum number of solver evaluations allowed by the optimizer.

        Return value:
        Dictionary keyed by model names containing optimal parameters, errors, and
        diagnostics.
        """
        results: dict[str, dict[str, Any]] = {}
        evaluations = max(1, int(max_iterations))
        fitting_settings = server_settings.fitting
        sample_size = int(uptake.shape[0])
        normalized_method = self.normalize_method(optimization_method)
        for model_name, model_config in configuration.items():
            model = self.collection.get_model(model_name)
            signature = inspect.signature(model)
            param_names = list(signature.parameters.keys())[1:]
            # ``curve_fit`` expects ordered arrays for initial guess and bounds, so we
            # align configuration dictionaries with the model signature parameters.
            initial = [
                model_config.get("initial", {}).get(
                    param, fitting_settings.parameter_initial_default
                )
                for param in param_names
            ]
            lower = [
                model_config.get("min", {}).get(
                    param, fitting_settings.parameter_min_default
                )
                for param in param_names
            ]
            upper = [
                model_config.get("max", {}).get(
                    param, fitting_settings.parameter_max_default
                )
                for param in param_names
            ]

            try:
                optimal_params, covariance, errors, predicted = self.solve_model(
                    normalized_method,
                    model,
                    pressure,
                    uptake,
                    initial,
                    lower,
                    upper,
                    evaluations,
                )
                optimal_list = optimal_params.tolist()
                covariance_list = (
                    covariance.tolist() if covariance is not None else None
                )
                error_list = (
                    errors.tolist() if isinstance(errors, np.ndarray) else errors
                )
                if error_list is None:
                    error_list = [np.nan] * len(param_names)
                else:
                    error_list = list(error_list)
                score = float(np.sum((uptake - predicted) ** 2, dtype=np.float64))
                parameter_count = len(param_names)
                aic, aicc = self.compute_information_metrics(
                    score,
                    sample_size,
                    parameter_count,
                )
                results[model_name] = {
                    "optimal_params": optimal_list,
                    "covariance": covariance_list,
                    "errors": error_list,
                    "score": score,
                    "aic": aic,
                    "aicc": aicc,
                    "optimization_method": normalized_method,
                    "arguments": param_names,
                    "measurement_count": sample_size,
                    "parameter_count": parameter_count,
                }
            except Exception as exc:  # noqa: BLE001
                error_type = type(exc).__name__
                error_msg = str(exc).split("\n")[0][:120]
                logger.error(
                    "Fitting failed for %s [%s]: %s - %s",
                    experiment_name,
                    model_name,
                    error_type,
                    error_msg,
                )
                logger.debug(
                    "Full traceback for %s [%s]",
                    experiment_name,
                    model_name,
                    exc_info=True,
                )
                results[model_name] = {
                    "optimal_params": [np.nan] * len(param_names),
                    "covariance": None,
                    "errors": [np.nan] * len(param_names),
                    "score": np.nan,
                    "aic": np.nan,
                    "aicc": np.nan,
                    "optimization_method": normalized_method,
                    "arguments": param_names,
                    "measurement_count": sample_size,
                    "parameter_count": len(param_names),
                    "exception": exc,
                }
        return results

    # -------------------------------------------------------------------------
    def solve_model(
        self,
        method: str,
        model: Callable[..., np.ndarray],
        pressure: np.ndarray,
        uptake: np.ndarray,
        initial: list[float],
        lower: list[float],
        upper: list[float],
        evaluations: int,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
        normalized_method = self.normalize_method(method)
        if normalized_method == "LSS":
            return self.solve_with_curve_fit(
                model,
                pressure,
                uptake,
                initial,
                lower,
                upper,
                evaluations,
            )
        return self.solve_with_minimize(
            normalized_method,
            model,
            pressure,
            uptake,
            initial,
            lower,
            upper,
            evaluations,
        )

    # -------------------------------------------------------------------------
    def solve_with_curve_fit(
        self,
        model: Callable[..., np.ndarray],
        pressure: np.ndarray,
        uptake: np.ndarray,
        initial: list[float],
        lower: list[float],
        upper: list[float],
        evaluations: int,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
        optimal_params, covariance = curve_fit(
            model,
            pressure,
            uptake,
            p0=initial,
            bounds=(lower, upper),
            maxfev=evaluations,
            check_finite=True,
            absolute_sigma=False,
        )
        optimal_array = np.asarray(optimal_params, dtype=np.float64)
        covariance_array = (
            np.asarray(covariance, dtype=np.float64) if covariance is not None else None
        )
        predicted = model(pressure, *optimal_array)
        errors = (
            np.sqrt(np.diag(covariance_array)).astype(float)
            if covariance_array is not None
            else None
        )
        return optimal_array, covariance_array, errors, predicted

    # -------------------------------------------------------------------------
    def solve_with_minimize(
        self,
        method: str,
        model: Callable[..., np.ndarray],
        pressure: np.ndarray,
        uptake: np.ndarray,
        initial: list[float],
        lower: list[float],
        upper: list[float],
        evaluations: int,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
        lower_bounds = np.asarray(lower, dtype=np.float64)
        upper_bounds = np.asarray(upper, dtype=np.float64)
        initial_guess = np.asarray(initial, dtype=np.float64)
        clipped_initial = np.clip(initial_guess, lower_bounds, upper_bounds)

        def project(params: np.ndarray) -> np.ndarray:
            return np.clip(params, lower_bounds, upper_bounds)

        residual_scale = float(np.sum(uptake * uptake, dtype=np.float64))
        penalty = max(1.0, residual_scale) * 1e6

        def objective(params: np.ndarray) -> float:
            projected = project(params)
            with np.errstate(all="ignore"):
                predicted = model(pressure, *projected)
            if not np.all(np.isfinite(predicted)):
                return penalty
            residuals = uptake - predicted
            if not np.all(np.isfinite(residuals)):
                return penalty
            return float(np.sum(residuals * residuals, dtype=np.float64))

        bounds = None
        if method in BOUNDS_COMPATIBLE_METHODS:
            bounds = list(zip(lower_bounds, upper_bounds))

        options = {"maxiter": evaluations}
        evaluations_per_param = evaluations * max(1, len(initial_guess))
        if method == "L-BFGS-B":
            options["maxfun"] = evaluations_per_param
        if method in {"Powell", "Nelder-Mead"}:
            options["maxfev"] = evaluations_per_param

        result = minimize(
            objective,
            clipped_initial,
            method=method,
            bounds=bounds,
            options=options,
        )
        if not result.success:
            logger.warning(
                "%s optimization did not converge: %s", method, result.message
            )

        optimal = np.asarray(result.x, dtype=np.float64)
        optimal = project(optimal)
        with np.errstate(all="ignore"):
            predicted = model(pressure, *optimal)
        if not np.all(np.isfinite(predicted)):
            predicted = np.full_like(pressure, np.nan, dtype=np.float64)
        covariance = self.extract_covariance_matrix(result)
        errors = (
            np.sqrt(np.diag(covariance)).astype(float)
            if covariance is not None
            else None
        )
        return optimal, covariance, errors, predicted

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_method(method: str) -> str:
        normalized = method.replace("_", "-").strip().upper()
        for candidate in SUPPORTED_OPTIMIZATION_METHODS:
            if candidate.upper() == normalized:
                return candidate
        return DEFAULT_OPTIMIZATION_METHOD

    # -------------------------------------------------------------------------
    @staticmethod
    def extract_covariance_matrix(result: Any) -> np.ndarray | None:
        hessian_inverse = getattr(result, "hess_inv", None)
        if hessian_inverse is None:
            return None
        matrix = hessian_inverse
        if hasattr(matrix, "todense"):
            matrix = matrix.todense()
        try:
            covariance = np.asarray(matrix, dtype=np.float64)
        except Exception:  # noqa: BLE001
            return None
        if covariance.ndim != 2:
            return None
        return covariance

    # -------------------------------------------------------------------------
    def bulk_data_fitting(
        self,
        dataset: pd.DataFrame,
        configuration: dict[str, Any],
        pressure_col: str,
        uptake_col: str,
        max_iterations: int,
        optimization_method: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[dict[str, list[dict[str, Any]]], list[int], list[str]]:
        """Iterate over the dataset and fit every experiment with the configured models."""
        results: dict[str, list[dict[str, Any]]] = {
            model: [] for model in configuration.keys()
        }
        valid_indices: list[int] = []
        skipped_experiments: list[str] = []
        total_experiments = dataset.shape[0]
        normalized_method = self.normalize_method(optimization_method)
        for index, row in dataset.iterrows():
            experiment_name_value = row.get(COLUMN_EXPERIMENT)
            if experiment_name_value is None or pd.isna(experiment_name_value):
                experiment_name = f"experiment_{index}"
            else:
                experiment_name = str(experiment_name_value)
            try:
                pressure = np.asarray(row[pressure_col], dtype=np.float64)
                uptake = np.asarray(row[uptake_col], dtype=np.float64)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping experiment %s due to invalid measurements: %s",
                    experiment_name,
                    exc,
                )
                skipped_experiments.append(experiment_name)
                if progress_callback is not None:
                    progress_callback(index + 1, total_experiments)
                continue

            if not self.validate_measurements(pressure, uptake):
                logger.info(
                    "Skipping experiment %s due to invalid measurements",
                    experiment_name,
                )
                skipped_experiments.append(experiment_name)
                if progress_callback is not None:
                    progress_callback(index + 1, total_experiments)
                continue

            try:
                experiment_results = self.single_experiment_fit(
                    pressure,
                    uptake,
                    experiment_name,
                    configuration,
                    max_iterations,
                    normalized_method,
                )
            except Exception as exc:  # noqa: BLE001
                error_type = type(exc).__name__
                error_msg = str(exc).split("\n")[0][:120]
                logger.error(
                    "Unexpected fitting error for %s: %s - %s",
                    experiment_name,
                    error_type,
                    error_msg,
                )
                logger.debug(
                    "Full traceback for experiment %s",
                    experiment_name,
                    exc_info=True,
                )
                skipped_experiments.append(experiment_name)
                if progress_callback is not None:
                    progress_callback(index + 1, total_experiments)
                continue

            for model_name, data in experiment_results.items():
                results[model_name].append(data)
            valid_indices.append(index)

            if progress_callback is not None:
                progress_callback(index + 1, total_experiments)

        return results, valid_indices, skipped_experiments

    # -------------------------------------------------------------------------
    @staticmethod
    def validate_measurements(pressure: np.ndarray, uptake: np.ndarray) -> bool:
        if pressure.ndim != 1 or uptake.ndim != 1:
            return False
        if pressure.size < 2 or uptake.size < 2:
            return False
        if pressure.size != uptake.size:
            return False
        if not np.all(np.isfinite(pressure)):
            return False
        if not np.all(np.isfinite(uptake)):
            return False
        return True


###############################################################################
class FittingPipeline:
    def __init__(self) -> None:
        self.serializer = DataSerializer()
        self.solver = ModelSolver()
        self.adapter = DatasetAdapter()

    # -------------------------------------------------------------------------
    def run(
        self,
        dataset_payload: dict[str, Any],
        configuration: dict[str, dict[str, dict[str, float]]],
        max_iterations: int,
        optimization_method: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        dataframe = self.build_dataframe(dataset_payload)
        if dataframe.empty:
            raise ValueError("Uploaded dataset is empty.")

        dataset_name = dataset_payload.get("dataset_name")
        if isinstance(dataset_name, str):
            dataset_name = dataset_name.strip() or None
        else:
            dataset_name = None

        processor = AdsorptionDataProcessor(dataframe)
        processed, detected_columns, stats = processor.preprocess(detect_columns=True)
        processed = self.attach_experiment_name(processed, dataset_name)

        logger.info("Processed dataset contains %s experiments", processed.shape[0])
        logger.debug("Detected dataset statistics:\n%s", stats)

        if processed.empty:
            raise ValueError(
                "No valid experiments found after preprocessing the dataset."
            )

        requested_models = sorted(configuration.keys())
        model_configuration, skipped_models = self.normalize_configuration(
            configuration
        )
        if not model_configuration:
            raise ValueError("No supported models were selected for fitting.")
        logger.debug("Running solver with configuration: %s", model_configuration)

        results, valid_indices, skipped_experiments = self.solver.bulk_data_fitting(
            processed,
            model_configuration,
            detected_columns.pressure,
            detected_columns.uptake,
            max_iterations,
            optimization_method,
            progress_callback=progress_callback,
        )

        if skipped_experiments:
            logger.info(
                "Skipped %s experiments during fitting", len(skipped_experiments)
            )

        if valid_indices:
            processed = processed.loc[valid_indices].reset_index(drop=True)
        else:
            processed = processed.iloc[0:0]

        if processed.empty:
            raise ValueError(
                "No valid experiments were available after fitting validation."
            )

        combined = self.adapter.combine_results(results, processed)
        storage_dataset = self.trim_fitting_dataset(combined, detected_columns)
        self.serializer.save_fitting_results(storage_dataset)

        ranking_metric = server_settings.fitting.best_model_metric
        normalized_metric = self.adapter.normalize_metric(ranking_metric)
        best_frame = self.adapter.compute_best_models(
            storage_dataset, normalized_metric
        )
        self.serializer.save_best_fit(best_frame)

        experiment_count = int(processed.shape[0])
        configured_models = sorted(model_configuration.keys())
        model_outcomes = self.summarize_model_outcomes(results)
        response: dict[str, Any] = {
            "status": "success",
            "processed_rows": experiment_count,
            "models": configured_models,
            "best_model_saved": True,
        }

        if best_frame is not None:
            response["best_model_preview"] = self.build_preview(best_frame)

        summary_lines: list[str] = [
            "[INFO] ADSMOD fitting completed.",
            f"Dataset: {dataset_name or 'unknown'}",
            f"Raw rows: {int(dataframe.shape[0])}",
            f"Experiments processed: {experiment_count}",
            f"Optimization method: {self.solver.normalize_method(optimization_method)}",
            f"Max iterations: {int(max_iterations)}",
            f"Ranking metric: {normalized_metric}",
        ]
        if skipped_experiments:
            summary_lines.append(
                f"[WARN] Skipped experiments: {len(skipped_experiments)}"
            )
        if requested_models:
            summary_lines.append("Requested models:")
            summary_lines.extend([f"  - {model}" for model in requested_models])
        if configured_models:
            summary_lines.append("Configured models:")
            summary_lines.extend([f"  - {model}" for model in configured_models])
        if skipped_models:
            summary_lines.append("[WARN] Skipped unsupported models:")
            summary_lines.extend([f"  - {model}" for model in sorted(skipped_models)])
        if model_outcomes:
            summary_lines.append("Model results (success/total):")
            for model_name in configured_models:
                stats = model_outcomes.get(model_name)
                if not stats:
                    continue
                total = stats["total"]
                failures = stats["failures"]
                successes = max(0, total - failures)
                summary_lines.append(f"  - {model_name}: {successes}/{total}")
                if failures:
                    summary_lines.append(
                        f"[WARN] {model_name}: {failures} failures out of {total}"
                    )
        summary_lines.append("Best model selection stored in database.")
        response["summary"] = "\n".join(summary_lines)

        return response

    # -------------------------------------------------------------------------
    def build_dataframe(self, payload: dict[str, Any]) -> pd.DataFrame:
        records = payload.get("records")
        columns = payload.get("columns")
        if isinstance(records, list):
            dataframe = pd.DataFrame.from_records(records, columns=columns)
        else:
            dataframe = pd.DataFrame()
        dataset_name = payload.get(COLUMN_DATASET_NAME)
        if isinstance(dataset_name, str):
            dataset_name = dataset_name.strip()
            if dataset_name:
                dataframe[COLUMN_DATASET_NAME] = dataset_name
        return dataframe

    # -------------------------------------------------------------------------
    def attach_experiment_name(
        self, dataset: pd.DataFrame, dataset_name: str | None
    ) -> pd.DataFrame:
        if dataset.empty or COLUMN_EXPERIMENT_NAME in dataset.columns:
            return dataset
        if COLUMN_EXPERIMENT not in dataset.columns:
            return dataset

        base_series = None
        if COLUMN_FILENAME in dataset.columns:
            base_series = dataset[COLUMN_FILENAME]
        elif COLUMN_DATASET_NAME in dataset.columns:
            base_series = dataset[COLUMN_DATASET_NAME]
        elif dataset_name:
            base_series = pd.Series([dataset_name] * len(dataset), index=dataset.index)

        experiment_series = dataset[COLUMN_EXPERIMENT].astype("string").str.strip()
        if base_series is None:
            dataset[COLUMN_EXPERIMENT_NAME] = experiment_series
            return dataset

        base_series = base_series.astype("string").str.strip()
        if COLUMN_FILENAME in dataset.columns:
            name_series = base_series
        else:
            name_series = base_series + "_" + experiment_series
            name_series = name_series.where(base_series != "", experiment_series)

        dataset[COLUMN_EXPERIMENT_NAME] = name_series.fillna(experiment_series)
        return dataset

    # -------------------------------------------------------------------------
    def normalize_configuration(
        self, configuration: dict[str, dict[str, dict[str, float]]]
    ) -> tuple[dict[str, dict[str, dict[str, float]]], list[str]]:
        normalized: dict[str, dict[str, dict[str, float]]] = {}
        skipped: list[str] = []
        supported = {
            self.normalize_model_key(name): name
            for name in self.solver.collection.model_names
        }
        for model_name, config in configuration.items():
            normalized_key = self.normalize_model_key(model_name)
            resolved_name = self.resolve_model_name(model_name)
            if normalized_key not in supported or resolved_name is None:
                logger.warning(
                    "Skipping unsupported model configuration: %s", model_name
                )
                skipped.append(model_name)
                continue

            defaults = MODEL_PARAMETER_DEFAULTS.get(resolved_name, {})
            alias_map = PARAMETER_ALIAS_MAP.get(resolved_name, {})
            normalized_entry: dict[str, dict[str, float]] = {
                "min": {},
                "max": {},
                "initial": {},
            }

            for parameter, (lower_default, upper_default) in defaults.items():
                normalized_entry["min"][parameter] = float(lower_default)
                normalized_entry["max"][parameter] = float(upper_default)
                normalized_entry["initial"][parameter] = float(
                    lower_default + (upper_default - lower_default) / 2
                )

            self.apply_configuration_overrides(
                normalized_entry,
                config,
                alias_map,
            )

            parameters = set().union(
                normalized_entry["min"].keys(),
                normalized_entry["max"].keys(),
                normalized_entry["initial"].keys(),
            )

            for parameter in parameters:
                lower = float(
                    normalized_entry["min"].get(
                        parameter, server_settings.fitting.parameter_min_default
                    )
                )
                upper = float(
                    normalized_entry["max"].get(
                        parameter, server_settings.fitting.parameter_max_default
                    )
                )
                if upper < lower:
                    lower, upper = upper, lower
                normalized_entry["min"][parameter] = lower
                normalized_entry["max"][parameter] = upper
                if parameter not in normalized_entry["initial"]:
                    normalized_entry["initial"][parameter] = float(
                        lower + (upper - lower) / 2
                    )

            normalized[resolved_name] = normalized_entry
        return normalized, skipped

    # -------------------------------------------------------------------------
    @staticmethod
    def summarize_model_outcomes(
        results: dict[str, list[dict[str, Any]]],
    ) -> dict[str, dict[str, int]]:
        summary: dict[str, dict[str, int]] = {}
        for model_name, entries in results.items():
            total = len(entries)
            failures = sum(1 for entry in entries if entry.get("exception"))
            summary[model_name] = {"total": total, "failures": failures}
        return summary

    # -------------------------------------------------------------------------
    def stringify_sequences(self, dataset: pd.DataFrame) -> pd.DataFrame:
        converted = dataset.copy()
        for column in converted.columns:
            if (
                converted[column]
                .apply(lambda value: isinstance(value, (list, tuple)))
                .any()
            ):
                converted[column] = converted[column].apply(
                    lambda value: json.dumps(value)
                    if isinstance(value, (list, tuple))
                    else value
                )
        return converted

    # -------------------------------------------------------------------------
    def build_preview(self, dataset: pd.DataFrame) -> list[dict[str, Any]]:
        preview_columns = [
            column
            for column in dataset.columns
            if column.endswith(COLUMN_SCORE)
            or column.endswith(COLUMN_AIC)
            or column.endswith(COLUMN_AICC)
        ]
        preview_columns.extend(
            [
                column
                for column in dataset.columns
                if column in {COLUMN_EXPERIMENT, COLUMN_BEST_MODEL, COLUMN_WORST_MODEL}
            ]
        )
        trimmed = dataset.loc[:, dict.fromkeys(preview_columns).keys()]
        limited = trimmed.head(server_settings.fitting.preview_row_limit)
        limited = limited.replace({np.nan: None})
        return limited.to_dict(orient="records")

    # -------------------------------------------------------------------------
    @staticmethod
    def trim_fitting_dataset(
        dataset: pd.DataFrame, detected_columns: DatasetColumns
    ) -> pd.DataFrame:
        drop_columns = [
            detected_columns.temperature,
            detected_columns.pressure,
            detected_columns.uptake,
            COLUMN_MEASUREMENT_COUNT,
            COLUMN_MIN_PRESSURE,
            COLUMN_MAX_PRESSURE,
            COLUMN_MIN_UPTAKE,
            COLUMN_MAX_UPTAKE,
        ]
        return dataset.drop(columns=drop_columns, errors="ignore")

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_model_key(model_name: str) -> str:
        normalized = model_name.replace("\u2013", "-").replace("\u2014", "-")
        normalized = re.sub(r"\s*\([^)]*\)\s*", " ", normalized)
        normalized = " ".join(normalized.split()).strip()
        return normalized.replace("-", "_").replace(" ", "_").upper()

    # -------------------------------------------------------------------------
    @staticmethod
    def resolve_model_name(model_name: str) -> str | None:
        normalized = FittingPipeline.normalize_model_key(model_name)
        for candidate in MODEL_PARAMETER_DEFAULTS:
            if FittingPipeline.normalize_model_key(candidate) == normalized:
                return candidate
        return None

    # -------------------------------------------------------------------------
    @staticmethod
    def apply_configuration_overrides(
        target: dict[str, dict[str, float]],
        source: dict[str, dict[str, float]],
        alias_map: dict[str, str],
    ) -> None:
        for bound_type in ("min", "max", "initial"):
            overrides = source.get(bound_type, {})
            for parameter, value in overrides.items():
                backend_param = alias_map.get(parameter, parameter)
                target[bound_type][backend_param] = float(value)
