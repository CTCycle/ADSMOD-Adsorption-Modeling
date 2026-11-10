# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""The one canonical experiment representation every adsorption source emits.

Uploads, NIST acquisition, and dataset edits all build `CanonicalExperiment`
records and pass them through `normalize_experiment` before persistence. An
experiment has one adsorbent and one or more species; every measurement is one
long-format row identified by (experiment, species, pressure) with the
canonical pressure and uptake columns. The number of species never selects a
different structure. An experiment has exactly one temperature
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
) -> dict[tuple[int, int], list[CanonicalObservation]]:
    name = experiment.name
    expected_unit = "1" if experiment.pressure_basis == "relative" else "Pa"
    grouped: dict[tuple[int, int], list[CanonicalObservation]] = {}
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
            pressure_key(
                observation.pressure_canonical, observation.pressure_canonical_unit
            ),
        )
        grouped.setdefault(key, []).append(observation)
    return grouped


###############################################################################
def _check_coverage(
    experiment: CanonicalExperiment,
    collapsed: dict[tuple[int, int], CanonicalObservation],
) -> None:
    name = experiment.name
    counts = Counter(position for position, _ in collapsed)
    for position, species in enumerate(experiment.adsorbates):
        if counts[position] < 2:
            raise CanonicalizationError(
                "insufficient_observations",
                name,
                f"Species '{species.name}' in experiment '{name}' needs at least "
                "two distinct pressures.",
            )
    if len(experiment.adsorbates) < 2 or experiment.pressure_basis != "absolute":
        return
    totals: Counter[int] = Counter()
    for (_, key), observation in collapsed.items():
        if observation.gas_mole_fraction is not None:
            totals[key] += observation.gas_mole_fraction
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

    Observations are ordered by species position, then pressure. Repeated
    (species, pressure) identities raise with ``reject`` and are averaged into
    one observation with ``average``; they are never silently overwritten.
    The function is idempotent, so persistence can re-apply it as a guard.
    """
    experiment = _validated(record)
    grouped = _group_by_identity(experiment, _species_positions(experiment))
    repeated = sorted(key for key, items in grouped.items() if len(items) > 1)
    if repeated and duplicates == "reject":
        position, _ = repeated[0]
        first = grouped[repeated[0]][0]
        raise CanonicalizationError(
            "duplicate_measurement",
            experiment.name,
            f"Experiment '{experiment.name}' has {len(repeated)} repeated "
            f"pressure(s) (first: {experiment.adsorbates[position].name} at "
            f"{first.pressure_canonical:g} {first.pressure_canonical_unit}). "
            "Each species may have one uptake per pressure; average the "
            "replicates or correct the source rows.",
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
