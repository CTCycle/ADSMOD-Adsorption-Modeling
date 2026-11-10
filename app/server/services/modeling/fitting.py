# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from server.domain.fitting import (
    ExperimentFitRequest,
    ModelDefinition,
    ModelParameterDefinition,
    ParameterBounds,
)
from server.common.units import UnitConversionError, UnitRegistry
from server.services.modeling.models import (
    AdsorptionModels,
    ModelSpec,
    ParameterSpec,
)


CONDITION_WARNING = 1e12
CURVE_POINT_COUNT = 200
MODEL_VERSION = "2.0"


###############################################################################
class FittingCancelledError(RuntimeError):
    """Raised when a fitting job receives a cooperative cancellation request."""


###############################################################################
def raise_if_cancelled(stop_event: Any | None) -> None:
    if stop_event is not None and stop_event.is_set():
        raise FittingCancelledError("Fitting cancelled.")


###############################################################################
@dataclass
class MetricResult:
    values: dict[str, float | None]
    warnings: list[str] = field(default_factory=list)


###############################################################################
@dataclass
class FitComputation:
    spec: ModelSpec
    status: str
    message: str
    parameters: np.ndarray | None
    standard_errors: np.ndarray | None
    ci95_low: np.ndarray | None
    ci95_high: np.ndarray | None
    metrics: MetricResult
    predicted: np.ndarray
    curve_pressure: np.ndarray
    curve_uptake: np.ndarray
    function_evaluations: int | None
    jacobian_rank: int | None
    condition_number: float | None
    warnings: list[str] = field(default_factory=list)
    rank: int | None = None


###############################################################################
def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


###############################################################################
def information_criteria(
    log_likelihood: float | None, likelihood_k: int, n: int, warnings: list[str]
) -> tuple[float | None, float | None, float | None]:
    if log_likelihood is None:
        return None, None, None
    aic = -2.0 * log_likelihood + 2.0 * likelihood_k
    bic = -2.0 * log_likelihood + likelihood_k * math.log(n) if n > 0 else None
    aicc = (
        aic + (2.0 * likelihood_k * (likelihood_k + 1.0)) / (n - likelihood_k - 1.0)
        if n > likelihood_k + 1
        else None
    )
    if aicc is None:
        warnings.append(
            "AICc is undefined because the sample size is too small for the likelihood parameter count."
        )
    return aic, aicc, bic


###############################################################################
def compute_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    parameter_count: int,
    uncertainty: np.ndarray | None = None,
    weighting: str = "unweighted",
) -> MetricResult:
    residuals = observed - predicted
    n = int(observed.size)
    k = int(parameter_count)
    warnings: list[str] = []
    sse = float(np.sum(residuals**2, dtype=np.float64))
    rmse = math.sqrt(sse / n) if n > 0 else None
    mae = float(np.mean(np.abs(residuals))) if n > 0 else None

    centered = observed - float(np.mean(observed)) if n else observed
    tss = float(np.sum(centered**2, dtype=np.float64))
    if n == 0 or tss <= 0:
        r_squared = None
        warnings.append(
            "R² is undefined because the observed uptake has zero variance."
        )
    else:
        r_squared = 1.0 - sse / tss

    if r_squared is None or n <= k + 1:
        adjusted_r_squared = None
        warnings.append(
            "Adjusted R² is undefined because there are insufficient residual degrees of freedom."
        )
    else:
        adjusted_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / (n - k - 1)

    if uncertainty is None:
        chi_square = None
    elif (
        uncertainty.size != n
        or np.any(~np.isfinite(uncertainty))
        or np.any(uncertainty <= 0)
    ):
        chi_square = None
        warnings.append(
            "Chi-square is undefined because uptake uncertainties are incomplete or invalid."
        )
    else:
        chi_square = float(np.sum((residuals / uncertainty) ** 2))

    if (
        weighting == "inverse_sigma"
        and uncertainty is not None
        and uncertainty.size == n
        and np.all(np.isfinite(uncertainty))
        and np.all(uncertainty > 0)
    ):
        # Known-sigma Gaussian likelihood; model parameters only are counted.
        chi_square = float(np.sum((residuals / uncertainty) ** 2))
        log_likelihood = float(
            -0.5 * (chi_square + np.sum(np.log(2.0 * math.pi * uncertainty**2)))
        )
        likelihood_k = k
    elif n > 0 and sse > 0:
        # Unknown common variance estimated from residuals; count that variance parameter.
        log_likelihood = float(-0.5 * n * (math.log(2.0 * math.pi * sse / n) + 1.0))
        likelihood_k = k + 1
    else:
        log_likelihood = None
        likelihood_k = k + 1
        warnings.append(
            "Information criteria are undefined for an empty or exact-zero-residual fit."
        )
    aic, aicc, bic = information_criteria(log_likelihood, likelihood_k, n, warnings)

    return MetricResult(
        values={
            "sse": finite_or_none(sse),
            "rmse": finite_or_none(rmse) if rmse is not None else None,
            "mae": finite_or_none(mae) if mae is not None else None,
            "r_squared": finite_or_none(r_squared) if r_squared is not None else None,
            "adjusted_r_squared": finite_or_none(adjusted_r_squared)
            if adjusted_r_squared is not None
            else None,
            "chi_square": finite_or_none(chi_square)
            if chi_square is not None
            else None,
            "aic": finite_or_none(aic) if aic is not None else None,
            "aicc": finite_or_none(aicc) if aicc is not None else None,
            "bic": finite_or_none(bic) if bic is not None else None,
        },
        warnings=warnings,
    )


###############################################################################
def pressure_factor(unit: str, pressure_basis: str) -> float:
    resolved = UnitRegistry.pressure_unit(unit)
    if pressure_basis == "relative":
        if resolved == "1":
            return 1.0
        if resolved == "%":
            return 0.01
        raise UnitConversionError(
            "Relative-pressure results can be displayed only as p/p0 or percent."
        )
    if resolved not in UnitRegistry.PRESSURE_TO_PA:
        raise UnitConversionError("A dimensional pressure display unit is required.")
    return UnitRegistry.PRESSURE_TO_PA[resolved]


###############################################################################
def parameter_unit(
    parameter: ParameterSpec,
    *,
    pressure_unit: str,
    uptake_unit: str,
    related_value: float | None = None,
) -> str:
    kind = parameter.unit_kind
    if kind == "pressure^-1":
        return f"{pressure_unit}⁻¹"
    if kind == "uptake":
        return uptake_unit
    if kind == "dimensionless":
        return "1"
    if kind == "energy^-2":
        return "mol²/J²"
    if kind == "uptake/pressure":
        return f"{uptake_unit}/{pressure_unit}"
    if kind == "pressure^-beta":
        exponent = related_value if related_value is not None else "β"
        return (
            f"{pressure_unit}^(-{exponent:g})"
            if isinstance(exponent, float)
            else f"{pressure_unit}^(-β)"
        )
    if kind == "freundlich":
        exponent = related_value if related_value is not None else "n"
        return (
            f"{uptake_unit}·{pressure_unit}^(-1/{exponent:g})"
            if isinstance(exponent, float)
            else f"{uptake_unit}·{pressure_unit}^(-1/n)"
        )
    return "1"


###############################################################################
def parameter_to_display(
    parameter: ParameterSpec,
    value: float,
    *,
    pressure_factor_value: float,
    uptake_factor_value: float,
    related_value: float | None,
) -> float:
    kind = parameter.unit_kind
    if kind == "pressure^-1":
        return value * pressure_factor_value
    if kind == "uptake":
        return value / uptake_factor_value
    if kind in {"dimensionless", "energy^-2"}:
        return value
    if kind == "uptake/pressure":
        return value * pressure_factor_value / uptake_factor_value
    if kind == "pressure^-beta":
        return value * pressure_factor_value ** float(related_value or 1.0)
    if kind == "freundlich":
        return (
            value
            * pressure_factor_value ** (1.0 / float(related_value or 1.0))
            / uptake_factor_value
        )
    return value


###############################################################################
def display_parameter_value(
    value: float | None,
    parameter: ParameterSpec,
    *,
    pressure_factor_value: float,
    uptake_factor_value: float,
    related_value: float,
) -> float | None:
    if value is None:
        return None
    return parameter_to_display(
        parameter,
        float(value),
        pressure_factor_value=pressure_factor_value,
        uptake_factor_value=uptake_factor_value,
        related_value=related_value,
    )


###############################################################################
def _fit_residual(
    parameters: np.ndarray,
    models: AdsorptionModels,
    model_name: str,
    pressure: np.ndarray,
    uptake: np.ndarray,
    temperature_k: float,
    pressure_basis: str,
    saturation_pressure_pa: float | None,
    sigma: np.ndarray | None,
) -> np.ndarray:
    predicted = models.evaluate(
        model_name,
        pressure,
        parameters,
        temperature_k=temperature_k,
        pressure_basis=pressure_basis,  # type: ignore[arg-type]
        saturation_pressure_pa=saturation_pressure_pa,
    )
    return _fit_residual_values(uptake, predicted, sigma)


###############################################################################
def _fit_residual_values(
    uptake: np.ndarray, predicted: np.ndarray, sigma: np.ndarray | None
) -> np.ndarray:
    values = uptake - predicted
    return values / sigma if sigma is not None else values


###############################################################################
class ModelSolver:
    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.models = AdsorptionModels()

    # -------------------------------------------------------------------------
    def initial_configuration(
        self,
        spec: ModelSpec,
        pressure: np.ndarray,
        uptake: np.ndarray,
        temperature_k: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positive_p = pressure[pressure > 0]
        p_min = float(np.min(positive_p)) if positive_p.size else 1.0
        p_median = float(np.median(positive_p)) if positive_p.size else 1.0
        p_max = float(np.max(positive_p)) if positive_p.size else 1.0
        q_max = max(float(np.max(uptake)), np.finfo(float).eps)
        q_median = max(float(np.median(uptake)), np.finfo(float).eps)

        initial: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        for parameter in spec.parameters:
            if parameter.unit_kind == "pressure^-1":
                guess = 1.0 / p_median
                low = max(parameter.lower, 1.0 / (p_max * 1e6))
                high = min(parameter.upper, 1e3 / p_min)
                if spec.key == "temkin":
                    low = max(low, 1.0 / p_min)
                    guess = max(guess, low * 1.05)
            elif parameter.unit_kind == "uptake":
                guess = q_max * (
                    0.55
                    if "qsat" in parameter.name and spec.key == "dual_site_langmuir"
                    else 1.1
                )
                low = max(parameter.lower, q_max * 1e-8)
                high = min(parameter.upper, q_max * 100.0)
            elif parameter.unit_kind == "freundlich":
                guess = q_median / (p_median**1.0)
                low = max(parameter.lower, guess * 1e-6)
                high = min(parameter.upper, guess * 1e6)
            elif parameter.unit_kind == "energy^-2":
                span = (
                    UnitRegistry.GAS_CONSTANT_J_MOL_K
                    * temperature_k
                    * max(1.0, math.log(max(p_max / p_min, 1.0)))
                )
                guess = 1.0 / (span * span)
                low = parameter.lower
                high = min(parameter.upper, max(guess * 1e6, parameter.lower * 10))
            elif parameter.name == "beta" and spec.key == "redlich_peterson":
                low = max(parameter.lower, 0.05)
                high = min(parameter.upper, 1.0)
                guess = 0.8
            else:
                guess, low, high = 1.0, parameter.lower, parameter.upper
            if high <= low:
                # The data-scale window misses the explicit interval: never widen
                # beyond the requested bounds, search them as given instead.
                low, high = parameter.lower, parameter.upper
            initial.append(float(np.clip(guess, low, high)))
            lower.append(low)
            upper.append(high)

        if spec.key == "dual_site_langmuir":
            initial[0] = min(upper[0], max(lower[0], 10.0 / p_median))
            initial[2] = min(upper[2], max(lower[2], 0.1 / p_median))
        return np.asarray(initial), np.asarray(lower), np.asarray(upper)

    # -------------------------------------------------------------------------
    @staticmethod
    def _bound_warnings(
        spec: ModelSpec,
        optimal: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> list[str]:
        """Flag fitted parameters that ended on a search bound."""
        warnings: list[str] = []
        for parameter, value, low, high in zip(
            spec.parameters, optimal, lower, upper, strict=True
        ):
            tolerance = 1e-6 * max(abs(high - low), np.finfo(float).tiny)
            if value - low <= tolerance or high - value <= tolerance:
                side = "lower" if value - low <= tolerance else "upper"
                warnings.append(
                    f"Parameter '{parameter.name}' reached its {side} bound; "
                    "the estimate is constrained by the search interval."
                )
        return warnings

    # -------------------------------------------------------------------------
    @staticmethod
    def _parameter_uncertainty(
        *,
        optimal: np.ndarray,
        jacobian: np.ndarray,
        rank: int,
        condition: float,
        sigma: np.ndarray | None,
        residuals: np.ndarray,
        observation_count: int,
    ) -> tuple[list[str], np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Return warnings, standard errors, and 95% CI bounds for the fit."""
        warnings: list[str] = []
        parameter_count = optimal.size
        if rank < parameter_count:
            warnings.append(
                "Parameter uncertainty is unavailable because the Jacobian is rank deficient."
            )
            return warnings, None, None, None
        if not math.isfinite(condition) or condition > CONDITION_WARNING:
            warnings.append(
                "The fit is poorly conditioned; parameter estimates may not be identifiable."
            )
            return warnings, None, None, None

        dof = observation_count - parameter_count
        covariance = None
        if sigma is not None:
            covariance = np.linalg.inv(jacobian.T @ jacobian)
        elif dof <= 0:
            warnings.append(
                "Parameter uncertainty is undefined because residual degrees of freedom are not positive."
            )
        else:
            variance = float(np.sum(residuals**2)) / dof
            covariance = np.linalg.inv(jacobian.T @ jacobian) * variance
        diagonal = np.array([-1.0]) if covariance is None else np.diag(covariance)
        if np.any(diagonal < 0) or not np.all(np.isfinite(diagonal)):
            warnings.append(
                "Parameter uncertainty is unavailable because covariance is invalid."
            )
            return warnings, None, None, None
        standard_errors = np.sqrt(diagonal)
        from scipy.stats import t as student_t  # deferred: scipy loads on first fit

        critical = float(student_t.ppf(0.975, dof))
        return (
            warnings,
            standard_errors,
            optimal - critical * standard_errors,
            optimal + critical * standard_errors,
        )

    # -------------------------------------------------------------------------
    def fit(
        self,
        *,
        spec: ModelSpec,
        pressure: np.ndarray,
        uptake: np.ndarray,
        uncertainty: np.ndarray | None,
        temperature_k: float,
        pressure_basis: str,
        saturation_pressure_pa: float | None,
        optimizer: str,
        max_evaluations: int,
        weighting: str = "unweighted",
        stop_event: Any | None = None,
    ) -> FitComputation:
        raise_if_cancelled(stop_event)
        parameter_count = len(spec.parameters)
        if pressure.size <= parameter_count:
            return self.failure(
                spec,
                pressure,
                f"{spec.name} has {parameter_count} fitted parameters but only "
                f"{pressure.size} observations; n must be greater than k.",
            )

        try:
            initial, lower, upper = self.initial_configuration(
                spec, pressure, uptake, temperature_k
            )
            if (
                np.any(lower >= upper)
                or np.any(initial < lower)
                or np.any(initial > upper)
            ):
                raise ValueError(
                    f"{spec.name} parameter bounds or initial values are invalid."
                )

            sigma = (
                uncertainty
                if uncertainty is not None
                and uncertainty.size == pressure.size
                and np.all(np.isfinite(uncertainty))
                and np.all(uncertainty > 0)
                else None
            )
            if weighting == "inverse_sigma" and sigma is None:
                raise ValueError(
                    "Inverse-sigma weighting requires a complete positive uncertainty series."
                )
            # Uncertainties shape the objective only when weighting is requested.
            fit_sigma = sigma if weighting == "inverse_sigma" else None

            starts = [initial]
            if parameter_count >= 3:
                starts.extend(
                    [
                        np.clip(initial * 0.5, lower, upper),
                        np.clip(initial * 2.0, lower, upper),
                    ]
                )
            from scipy.optimize import least_squares  # deferred: loads on first fit

            candidates = []
            for start in starts:
                raise_if_cancelled(stop_event)
                result = least_squares(
                    _fit_residual,
                    start,
                    args=(
                        self.models,
                        spec.key,
                        pressure,
                        uptake,
                        temperature_k,
                        pressure_basis,
                        saturation_pressure_pa,
                        fit_sigma,
                    ),
                    bounds=(lower, upper),
                    method=optimizer,
                    loss="linear",
                    x_scale="jac",
                    max_nfev=max_evaluations,
                    callback=(
                        lambda _result: (
                            raise_if_cancelled(stop_event)
                            if stop_event is not None
                            else None
                        )
                    ),
                )
                raise_if_cancelled(stop_event)
                predicted_candidate = self.models.evaluate(
                    spec.key,
                    pressure,
                    result.x,
                    temperature_k=temperature_k,
                    pressure_basis=pressure_basis,  # type: ignore[arg-type]
                    saturation_pressure_pa=saturation_pressure_pa,
                )
                candidate_objective = float(
                    np.sum(
                        _fit_residual_values(uptake, predicted_candidate, fit_sigma)
                        ** 2
                    )
                )
                candidates.append((candidate_objective, result, predicted_candidate))
            _, result, predicted = min(candidates, key=lambda item: item[0])

            if not result.success:
                return self.failure(
                    spec,
                    pressure,
                    f"Optimizer did not converge: {result.message}",
                    function_evaluations=int(result.nfev),
                )
            optimal = np.asarray(result.x, dtype=np.float64)
            jacobian = np.asarray(result.jac, dtype=np.float64)
            rank = int(np.linalg.matrix_rank(jacobian))
            condition = (
                float(np.linalg.cond(jacobian))
                if jacobian.size and rank > 0
                else math.inf
            )
            warnings, standard_errors, ci_low, ci_high = self._parameter_uncertainty(
                optimal=optimal,
                jacobian=jacobian,
                rank=rank,
                condition=condition,
                sigma=fit_sigma,
                residuals=uptake - predicted,
                observation_count=pressure.size,
            )
            warnings.extend(self._bound_warnings(spec, optimal, lower, upper))

            if spec.key == "dual_site_langmuir" and optimal[0] < optimal[2]:
                permutation = np.array([2, 3, 0, 1])
                optimal = optimal[permutation]
                if standard_errors is not None:
                    standard_errors = standard_errors[permutation]
                    ci_low = ci_low[permutation] if ci_low is not None else None
                    ci_high = ci_high[permutation] if ci_high is not None else None
                predicted = self.models.evaluate(
                    spec.key,
                    pressure,
                    optimal,
                    temperature_k=temperature_k,
                    pressure_basis=pressure_basis,  # type: ignore[arg-type]
                    saturation_pressure_pa=saturation_pressure_pa,
                )

            metrics = compute_metrics(
                uptake,
                predicted,
                parameter_count,
                uncertainty=sigma,
                weighting=weighting,
            )
            curve_pressure = self.curve_grid(pressure, spec)
            curve_uptake = self.models.evaluate(
                spec.key,
                curve_pressure,
                optimal,
                temperature_k=temperature_k,
                pressure_basis=pressure_basis,  # type: ignore[arg-type]
                saturation_pressure_pa=saturation_pressure_pa,
            )
            warnings.extend(metrics.warnings)
            status = "warning" if warnings else "success"
            return FitComputation(
                spec=spec,
                status=status,
                message=str(result.message),
                parameters=optimal,
                standard_errors=standard_errors,
                ci95_low=ci_low,
                ci95_high=ci_high,
                metrics=metrics,
                predicted=predicted,
                curve_pressure=curve_pressure,
                curve_uptake=curve_uptake,
                function_evaluations=int(result.nfev),
                jacobian_rank=rank,
                condition_number=finite_or_none(condition),
                warnings=warnings,
            )
        except (ValueError, UnitConversionError, FloatingPointError) as exc:
            return self.failure(spec, pressure, str(exc))

    # -------------------------------------------------------------------------
    @staticmethod
    def curve_grid(pressure: np.ndarray, spec: ModelSpec) -> np.ndarray:
        minimum = float(np.min(pressure))
        maximum = float(np.max(pressure))
        positive = pressure[pressure > 0]
        if maximum == minimum:
            return np.asarray([minimum])
        if positive.size and float(np.max(positive)) / float(np.min(positive)) >= 1000:
            grid = np.geomspace(
                float(np.min(positive)), float(np.max(positive)), CURVE_POINT_COUNT
            )
            if minimum == 0 and spec.key not in {"freundlich", "temkin"}:
                grid = np.insert(grid, 0, 0.0)
            return grid
        start = minimum
        if start == 0 and spec.key in {"freundlich", "temkin"}:
            start = float(np.min(positive))
        return np.linspace(start, maximum, CURVE_POINT_COUNT)

    # -------------------------------------------------------------------------
    @staticmethod
    def failure(
        spec: ModelSpec,
        pressure: np.ndarray,
        message: str,
        *,
        function_evaluations: int | None = None,
    ) -> FitComputation:
        return FitComputation(
            spec=spec,
            status="failed",
            message=message,
            parameters=None,
            standard_errors=None,
            ci95_low=None,
            ci95_high=None,
            metrics=MetricResult(
                values={
                    key: None
                    for key in (
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
                }
            ),
            predicted=np.asarray([], dtype=np.float64),
            curve_pressure=np.asarray([], dtype=np.float64),
            curve_uptake=np.asarray([], dtype=np.float64),
            function_evaluations=function_evaluations,
            jacobian_rank=None,
            condition_number=None,
            warnings=[message],
        )


###############################################################################
class FittingPipeline:
    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.solver = ModelSolver()

    # -------------------------------------------------------------------------
    @staticmethod
    def input_hash(series: dict[str, Any]) -> str:
        payload = {
            key: series[key]
            for key in (
                "dataset_id",
                "isotherm_id",
                "adsorbate_id",
                "observation_ids",
                "pressure",
                "uptake",
                "temperature_k",
                "pressure_basis",
                "saturation_pressure_pa",
                "uptake_stddev",
            )
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------------
    def run(
        self,
        series: dict[str, Any],
        request: ExperimentFitRequest,
        *,
        stop_event: Any | None = None,
    ) -> list[FitComputation]:
        raise_if_cancelled(stop_event)
        pressure = np.asarray(series["pressure"], dtype=np.float64)
        uptake = np.asarray(series["uptake"], dtype=np.float64)
        if pressure.ndim != 1 or uptake.ndim != 1 or pressure.size != uptake.size:
            raise ValueError(
                "Canonical pressure and uptake arrays must be aligned one-dimensional arrays of equal length."
            )
        if pressure.size < 2:
            raise ValueError("At least two observations are required for fitting.")
        if np.any(~np.isfinite(pressure)) or np.any(pressure < 0):
            raise ValueError(
                "Canonical pressure values must be finite and non-negative."
            )
        if np.any(~np.isfinite(uptake)) or np.any(uptake < 0):
            raise ValueError("Canonical uptake values must be finite and non-negative.")
        uncertainty_values = series.get("uptake_stddev") or []
        uncertainty = (
            np.asarray(uncertainty_values, dtype=np.float64)
            if uncertainty_values
            and all(value is not None for value in uncertainty_values)
            else None
        )

        resolved_specs: list[ModelSpec] = []
        seen: set[str] = set()
        for model in request.models:
            spec = self.solver.models.get_spec(model)
            if spec.key not in seen:
                seen.add(spec.key)
                resolved_specs.append(
                    with_bounds(spec, request.model_bounds.get(spec.key, {}))
                )
        computations = []
        for spec in resolved_specs:
            raise_if_cancelled(stop_event)
            computations.append(
                self.solver.fit(
                    spec=spec,
                    pressure=pressure,
                    uptake=uptake,
                    uncertainty=uncertainty,
                    temperature_k=float(series["temperature_k"]),
                    pressure_basis=series["pressure_basis"],
                    saturation_pressure_pa=series.get("saturation_pressure_pa"),
                    optimizer=request.optimizer,
                    max_evaluations=request.max_evaluations,
                    weighting=request.weighting,
                    stop_event=stop_event,
                )
            )
        successful = [
            item
            for item in computations
            if item.status != "failed" and item.metrics.values["sse"] is not None
        ]
        successful.sort(
            key=lambda item: (
                item.metrics.values["aicc"] is None,
                item.metrics.values["aicc"]
                if item.metrics.values["aicc"] is not None
                else item.metrics.values["aic"]
                if item.metrics.values["aic"] is not None
                else item.metrics.values["sse"],
            )
        )
        for rank, item in enumerate(successful, start=1):
            item.rank = rank
        return computations

    # -------------------------------------------------------------------------
    def catalog(
        self,
        *,
        enabled: dict[str, bool],
        bounds: dict[str, dict[str, ParameterBounds]],
    ) -> list[ModelDefinition]:
        """Describe every model with its effective canonical-unit default bounds."""
        definitions: list[ModelDefinition] = []
        for key in self.solver.models.model_names:
            model = self.solver.models.get_spec(key)
            stored = bounds.get(model.key, {})
            definitions.append(
                ModelDefinition(
                    key=model.key,
                    name=model.name,
                    equation_latex=model.equation_latex,
                    assumptions=model.assumptions,
                    pressure_requirement=model.pressure_requirement,
                    requires_temperature=model.requires_temperature,
                    reference=model.reference,
                    enabled=enabled.get(model.key, True),
                    parameters=[
                        ModelParameterDefinition(
                            name=parameter.name,
                            label=parameter.label,
                            lower=stored[parameter.name].lower
                            if parameter.name in stored
                            else parameter.lower,
                            upper=stored[parameter.name].upper
                            if parameter.name in stored
                            else parameter.upper,
                            canonical_lower=parameter.lower,
                            canonical_upper=parameter.upper,
                            unit=parameter_unit(
                                parameter,
                                pressure_unit="Pa",
                                uptake_unit="mol/kg",
                                related_value=1.0,
                            ),
                            customized=parameter.name in stored,
                        )
                        for parameter in model.parameters
                    ],
                )
            )
        return definitions


###############################################################################
def with_bounds(spec: ModelSpec, bounds: dict[str, ParameterBounds]) -> ModelSpec:
    """Return the spec with its canonical search bounds replaced by ``bounds``."""
    if not bounds:
        return spec
    return replace(
        spec,
        parameters=tuple(
            replace(
                parameter,
                lower=bounds[parameter.name].lower,
                upper=bounds[parameter.name].upper,
            )
            if parameter.name in bounds
            else parameter
            for parameter in spec.parameters
        ),
    )
