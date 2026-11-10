# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""The one canonical experiment representation every adsorption source emits.

Uploads, NIST acquisition, and dataset edits all build `CanonicalExperiment`
records and pass them through `normalize_experiment` before persistence. An
experiment has one adsorbent and one or more species; every measurement is one
long-format row identified by (experiment, species, pressure) with the
canonical pressure and uptake columns. Every measurement also names its
branch (``adsorption`` or ``desorption``), so a hysteresis loop is stored in the
same long format and one pressure may hold one uptake per branch; the identity
is (experiment, species, branch, pressure). The number of species never
selects a different structure. An experiment has exactly one temperature
(`temperature_k`); measurements carry none, so a source whose rows disagree on
temperature must split them into separate experiments.

Pressure basis semantics (`Isotherm.pressure_basis`):

- ``absolute``: the system total pressure in Pa. All species at one
  equilibrium point share it; ``gas_mole_fraction`` (y) gives the species'
  partial pressure as y * P. With one species, total equals partial.
- ``partial``: each species' own partial pressure in Pa, as declared by the
  source. It is never derived from a total pressure.
- ``relative``: p/p0 of a single species (unit ``1``).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from server.common.identity import exact_key
from server.common.units import PressureBasis

DuplicatePolicy = Literal["reject", "average"]
ObservationBranch = Literal["adsorption", "desorption"]
BRANCH_ORDER: dict[str, int] = {"adsorption": 0, "desorption": 1}
BRANCH_LABELS: dict[str, ObservationBranch] = {
    "adsorption": "adsorption",
    "ads": "adsorption",
    "a": "adsorption",
    "desorption": "desorption",
    "des": "desorption",
    "d": "desorption",
}

# Quantization used by both duplicate detection and the database identity
# constraint: 1 µPa for absolute/partial pressures, 1e-9 for p/p0.
PRESSURE_KEY_SCALE: dict[str, float] = {"Pa": 1e6, "1": 1e9}
FRACTION_TOLERANCE = 1e-6


###############################################################################
class CanonicalizationError(ValueError):
    """An experiment that cannot be represented canonically, with a stable code."""

    def __init__(self, code: str, experiment: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.experiment = experiment


###############################################################################
class CanonicalSpecies(BaseModel):
    """Source identity of one adsorbate; resolution to the canonical species
    (curated catalog, InChIKey, aliases) happens at persistence."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1, max_length=255)
    inchi_key: str | None = None
    smiles: str | None = None
    molar_mass_g_mol: float | None = Field(default=None, gt=0)


###############################################################################
class CanonicalAdsorbent(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1, max_length=255)


###############################################################################
class CanonicalObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adsorbate: str = Field(min_length=1)
    source_row: int | None = None
    pressure_original: float = Field(allow_inf_nan=False)
    pressure_original_unit: str
    pressure_canonical: float = Field(ge=0, allow_inf_nan=False)
    pressure_canonical_unit: Literal["Pa", "1"]
    branch: ObservationBranch = "adsorption"
    uptake_original: float = Field(allow_inf_nan=False)
    uptake_original_unit: str
    uptake_mol_kg: float = Field(ge=0, allow_inf_nan=False)
    uptake_stddev_mol_kg: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    gas_mole_fraction: float | None = Field(default=None, ge=0, le=1)
    conversion_metadata: dict[str, Any] = Field(default_factory=dict)
    extra_metadata: dict[str, Any] = Field(default_factory=dict)


###############################################################################
class CanonicalExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    external_key: str = Field(min_length=1, max_length=255)
    name: str = Field(min_length=1, max_length=255)
    adsorbent: CanonicalAdsorbent
    adsorbates: list[CanonicalSpecies] = Field(min_length=1)
    temperature_original: float
    temperature_original_unit: str
    temperature_k: float = Field(gt=0)
    pressure_basis: PressureBasis
    saturation_pressure_pa: float | None = Field(default=None, gt=0)
    conditions: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    observations: list[CanonicalObservation]


###############################################################################
def pressure_key(value: float, unit: str) -> int:
    """Quantized pressure identity; equal keys mean the same measured pressure."""
    if unit not in PRESSURE_KEY_SCALE:
        raise ValueError(f"Unsupported canonical pressure unit '{unit}'.")
    if not math.isfinite(value) or value < 0:
        raise ValueError("Canonical pressure must be finite and non-negative.")
    return int(round(value * PRESSURE_KEY_SCALE[unit]))


###############################################################################
def species_key(name: str) -> str:
    return exact_key(name)


###############################################################################
def _collapse(items: list[CanonicalObservation]) -> CanonicalObservation:
    """Average replicate measurements of one identity into one observation."""
    first = items[0]
    sigmas = [item.uptake_stddev_mol_kg for item in items]
    sigma = (
        math.sqrt(sum(value * value for value in sigmas)) / len(sigmas)  # type: ignore[operator]
        if all(value is not None for value in sigmas)
        else None
    )
    fractions = [item.gas_mole_fraction for item in items]
    fraction = (
        sum(fractions) / len(fractions)  # type: ignore[arg-type]
        if all(value is not None for value in fractions)
        else None
    )
    count = len(items)
    return first.model_copy(
        update={
            "uptake_original": sum(item.uptake_original for item in items) / count,
            "uptake_mol_kg": sum(item.uptake_mol_kg for item in items) / count,
            "uptake_stddev_mol_kg": sigma,
            "gas_mole_fraction": fraction,
            "extra_metadata": {
                **first.extra_metadata,
                "replicates": {
                    "count": count,
                    "source_rows": [item.source_row for item in items],
                    "uptake_mol_kg": [item.uptake_mol_kg for item in items],
                },
            },
        }
    )


###############################################################################
def _validated(record: CanonicalExperiment | Mapping[str, Any]) -> CanonicalExperiment:
    if isinstance(record, CanonicalExperiment):
        return record
    try:
        return CanonicalExperiment.model_validate(dict(record))
    except ValidationError as exc:
        name = str(record.get("name") or record.get("external_key") or "experiment")
        detail = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()[:5]
        )
        raise CanonicalizationError(
            "invalid_experiment", name, f"Experiment '{name}' is invalid: {detail}"
        ) from exc


###############################################################################
def _species_positions(experiment: CanonicalExperiment) -> dict[str, int]:
    positions: dict[str, int] = {}
    for position, species in enumerate(experiment.adsorbates):
        key = species_key(species.name)
        if key in positions:
            raise CanonicalizationError(
                "duplicate_species",
                experiment.name,
                f"Experiment '{experiment.name}' lists species '{species.name}' "
                "more than once.",
            )
        positions[key] = position
    if experiment.pressure_basis == "relative" and len(positions) > 1:
        raise CanonicalizationError(
            "relative_pressure_multi_species",
            experiment.name,
            f"Experiment '{experiment.name}' uses relative pressure with several "
            "species; p/p0 is defined for one species only.",
        )
    return positions


###############################################################################
def _group_by_identity(
    experiment: CanonicalExperiment, positions: dict[str, int]
) -> dict[tuple[int, int, int], list[CanonicalObservation]]:
    name = experiment.name
    expected_unit = "1" if experiment.pressure_basis == "relative" else "Pa"
    grouped: dict[tuple[int, int, int], list[CanonicalObservation]] = {}
    for observation in experiment.observations:
        position = positions.get(species_key(observation.adsorbate))
        if position is None:
            raise CanonicalizationError(
                "undeclared_species",
                name,
                f"Experiment '{name}' has a measurement for undeclared species "
                f"'{observation.adsorbate}'.",
            )
        if observation.pressure_canonical_unit != expected_unit:
            raise CanonicalizationError(
                "pressure_unit_mismatch",
                name,
                f"Experiment '{name}' mixes pressure units with its "
                f"{experiment.pressure_basis} pressure basis.",
            )
        if len(positions) == 1 and observation.gas_mole_fraction not in (None, 1.0):
            raise CanonicalizationError(
                "single_species_fraction",
                name,
                f"Experiment '{name}' has one species, so its gas mole fraction "
                "must be 1 or omitted.",
            )
        key = (
            position,
            BRANCH_ORDER[observation.branch],
            pressure_key(
                observation.pressure_canonical, observation.pressure_canonical_unit
            ),
        )
        grouped.setdefault(key, []).append(observation)
    return grouped


###############################################################################
def _check_coverage(
    experiment: CanonicalExperiment,
    collapsed: dict[tuple[int, int, int], CanonicalObservation],
) -> None:
    name = experiment.name
    adsorption = BRANCH_ORDER["adsorption"]
    counts = Counter((position, branch) for position, branch, _ in collapsed)
    for position, species in enumerate(experiment.adsorbates):
        if counts[(position, adsorption)] < 2:
            desorption = counts[(position, BRANCH_ORDER["desorption"])]
            raise CanonicalizationError(
                "insufficient_observations",
                name,
                f"Species '{species.name}' in experiment '{name}' needs at least "
                "two distinct pressures on its adsorption branch"
                + (
                    f" (it has {desorption} desorption point(s), which cannot "
                    "be stored without an adsorption branch)."
                    if desorption
                    else "."
                ),
            )
    if len(experiment.adsorbates) < 2 or experiment.pressure_basis != "absolute":
        return
    # Species measured at one equilibrium point share its total pressure and
    # branch, so their fractions are summed per (branch, pressure).
    totals: Counter[tuple[int, int]] = Counter()
    for (_, branch, key), observation in collapsed.items():
        if observation.gas_mole_fraction is not None:
            totals[(branch, key)] += observation.gas_mole_fraction
    if any(total > 1 + FRACTION_TOLERANCE for total in totals.values()):
        raise CanonicalizationError(
            "gas_fraction_sum",
            name,
            f"Gas mole fractions of experiment '{name}' add up to more than 1 at "
            "one or more pressures.",
        )


###############################################################################
def normalize_experiment(
    record: CanonicalExperiment | Mapping[str, Any],
    duplicates: DuplicatePolicy = "reject",
) -> CanonicalExperiment:
    """Validate one experiment and return it in canonical, deterministic order.

    Observations are ordered by species position, branch (adsorption first),
    then pressure. Repeated (species, branch, pressure) identities raise with
    ``reject`` and are averaged into one observation with ``average``; they are
    never silently overwritten. Each species needs at least two adsorption
    pressures; desorption points are optional.
    The function is idempotent, so persistence can re-apply it as a guard.
    """
    experiment = _validated(record)
    grouped = _group_by_identity(experiment, _species_positions(experiment))
    repeated = sorted(key for key, items in grouped.items() if len(items) > 1)
    if repeated and duplicates == "reject":
        position, _, _ = repeated[0]
        first = grouped[repeated[0]][0]
        raise CanonicalizationError(
            "duplicate_measurement",
            experiment.name,
            f"Experiment '{experiment.name}' has {len(repeated)} repeated "
            f"pressure(s) (first: {experiment.adsorbates[position].name} at "
            f"{first.pressure_canonical:g} {first.pressure_canonical_unit} on "
            f"the {first.branch} branch). Each species may have one uptake per "
            "branch and pressure; average the replicates or correct the source "
            "rows.",
        )
    collapsed = {
        key: items[0] if len(items) == 1 else _collapse(items)
        for key, items in sorted(grouped.items())
    }
    _check_coverage(experiment, collapsed)
    provenance = dict(experiment.provenance)
    averaged = bool(repeated) or provenance.get("duplicates") == "averaged"
    provenance["duplicates"] = "averaged" if averaged else "none"
    return experiment.model_copy(
        update={"observations": list(collapsed.values()), "provenance": provenance}
    )


###############################################################################
def parse_branch(value: object) -> ObservationBranch:
    """Read a source branch label: adsorption/desorption, ads/des or a/d."""
    text = str(value if value is not None else "").strip().casefold()
    if text not in BRANCH_LABELS:
        raise ValueError(
            f"Branch '{value}' is not recognized; use adsorption or desorption "
            "(ads/des or a/d)."
        )
    return BRANCH_LABELS[text]


###############################################################################
def assign_branches(record: Mapping[str, Any]) -> dict[str, Any]:
    """Label the observations of a source that carries no branch marker.

    The rule is applied per species in source order, which must be the order
    of acquisition:

    - every point up to the first point at the species' highest pressure
      (inclusive) is ``adsorption``;
    - no later point may rise in pressure: a point below its predecessor is
      ``desorption``, and a point at its predecessor's pressure joins that
      point's branch, so a vertical step stays a repeated measurement that
      `normalize_experiment` rejects or averages.

    A pressure rise after the desorption branch has started (a second cycle or
    an unordered source) cannot be classified and raises
    `CanonicalizationError` with code ``ambiguous_branch`` instead of guessing.
    Explicit ``branch`` values are replaced, so callers apply this only to
    unlabelled sources. Provenance records that the branches were inferred.
    """
    name = str(record.get("name") or record.get("external_key") or "experiment")
    observations = [dict(item) for item in record.get("observations") or []]
    by_species: dict[str, list[dict[str, Any]]] = {}
    for observation in observations:
        by_species.setdefault(
            species_key(str(observation.get("adsorbate", ""))), []
        ).append(observation)
    for points in by_species.values():
        try:
            keys = [
                pressure_key(
                    float(point["pressure_canonical"]),
                    str(point["pressure_canonical_unit"]),
                )
                for point in points
            ]
        except (KeyError, TypeError, ValueError):
            # Invalid points are reported by `normalize_experiment`.
            continue
        peak = keys.index(max(keys))
        branch = "adsorption"
        for index, point in enumerate(points):
            if index > peak:
                if keys[index] > keys[index - 1]:
                    raise CanonicalizationError(
                        "ambiguous_branch",
                        name,
                        f"Species '{point.get('adsorbate')}' in experiment "
                        f"'{name}' rises in pressure again after its desorption "
                        f"branch started (source row {point.get('source_row')}), "
                        "so the branch of each point cannot be inferred. Label "
                        "the branch explicitly or split the cycles into "
                        "separate experiments.",
                    )
                if keys[index] < keys[index - 1]:
                    branch = "desorption"
            point["branch"] = branch
    return {
        **dict(record),
        "observations": observations,
        "provenance": {
            **dict(record.get("provenance") or {}),
            "branch_assignment": "inferred",
        },
    }
