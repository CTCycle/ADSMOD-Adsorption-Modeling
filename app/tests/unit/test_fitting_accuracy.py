# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np

from server.domain.fitting import ParameterBounds
from server.services.modeling.fitting import FittingPipeline, ModelSolver, with_bounds
from server.services.modeling.models import MODEL_SPECS, langmuir

PRESSURE_PA = np.array([2e2, 1e3, 5e3, 2e4, 6e4, 1.5e5, 4e5, 1e6])


def _fit(spec, pressure, uptake, *, sigma=None, weighting="unweighted", basis="absolute"):
    return ModelSolver().fit(
        spec=spec,
        pressure=pressure,
        uptake=uptake,
        uncertainty=sigma,
        temperature_k=298.15,
        pressure_basis=basis,
        saturation_pressure_pa=None,
        optimizer="trf",
        max_evaluations=2000,
        weighting=weighting,
    )


###############################################################################
def test_weighting_mode_controls_objective_not_uncertainty_availability() -> None:
    # Heterogeneous misfit with strongly heterogeneous sigma separates the objectives.
    uptake = langmuir(PRESSURE_PA, 2e-5, 4.0) * np.array(
        [1.15, 0.9, 1.1, 0.95, 1.2, 0.85, 1.0, 1.1]
    )
    sigma = np.array([0.01, 0.02, 0.05, 0.1, 0.4, 0.8, 1.5, 3.0])
    spec = MODEL_SPECS["langmuir"]

    unweighted = _fit(spec, PRESSURE_PA, uptake, sigma=sigma)
    unweighted_no_sigma = _fit(spec, PRESSURE_PA, uptake)
    weighted = _fit(spec, PRESSURE_PA, uptake, sigma=sigma, weighting="inverse_sigma")

    np.testing.assert_allclose(
        unweighted.parameters, unweighted_no_sigma.parameters, rtol=1e-4
    )
    assert not np.allclose(unweighted.parameters, weighted.parameters, rtol=1e-3)
    assert unweighted.metrics.values["sse"] < weighted.metrics.values["sse"]
    assert weighted.metrics.values["chi_square"] < unweighted.metrics.values["chi_square"]


###############################################################################
def test_explicit_bounds_are_never_widened() -> None:
    spec = with_bounds(
        MODEL_SPECS["langmuir"], {"k": ParameterBounds(lower=0.1, upper=0.2)}
    )
    pressure = np.array([1.0, 10.0, 100.0, 1000.0])

    _, lower, upper = ModelSolver().initial_configuration(
        spec, pressure, np.array([0.5, 1.0, 1.5, 2.0]), 298.15
    )

    assert lower[0] >= 0.1 and upper[0] <= 0.2


###############################################################################
def test_default_bounds_recover_high_affinity_on_relative_pressure() -> None:
    relative = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.6, 0.9])
    uptake = langmuir(relative, 100.0, 3.0)

    result = _fit(MODEL_SPECS["langmuir"], relative, uptake, basis="relative")

    assert result.parameters is not None
    np.testing.assert_allclose(result.parameters, [100.0, 3.0], rtol=1e-3)


###############################################################################
def test_input_hash_covers_uncertainty_and_duplicate_policy() -> None:
    series = {
        "dataset_id": 1,
        "isotherm_id": 1,
        "component_id": 1,
        "observation_ids": [1, 2],
        "pressure": [1.0, 2.0],
        "uptake": [1.0, 2.0],
        "uptake_stddev": [0.1, 0.1],
        "duplicate_policy": "reject",
        "temperature_k": 298.15,
        "pressure_basis": "absolute",
        "saturation_pressure_pa": None,
    }
    base = FittingPipeline.input_hash(series)

    assert FittingPipeline.input_hash({**series, "uptake_stddev": [0.1, 0.2]}) != base
    assert FittingPipeline.input_hash({**series, "duplicate_policy": "average"}) != base


###############################################################################
def test_parameter_on_bound_is_reported() -> None:
    spec = with_bounds(
        MODEL_SPECS["langmuir"], {"qsat": ParameterBounds(lower=0.1, upper=1.0)}
    )
    uptake = langmuir(PRESSURE_PA, 2e-5, 4.0)

    result = _fit(spec, PRESSURE_PA, uptake)

    assert any("qsat" in warning and "bound" in warning for warning in result.warnings)
