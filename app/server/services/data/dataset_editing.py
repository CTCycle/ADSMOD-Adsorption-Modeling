# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""Validate dataset edits and recompute canonical values before persisting.

Edits follow an overwrite model: the plan built here replaces existing values
in place, and `DatasetEditRepository.apply` writes it in a single transaction.
Unit conversion reuses the importer's `UnitCatalog`, so edited values obey the
same rules as imported ones.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from server.common.units import (
    KNOWN_MOLAR_MASSES_G_MOL,
    ConvertedValue,
    UnitCatalog,
    UnitConversionError,
)
from server.domain.canonical import FRACTION_TOLERANCE, pressure_key
from server.domain.datasets import (
    DatasetEditIssue,
    DatasetUpdateRequest,
    ExperimentUpdate,
    ObservationUpdate,
)
from server.repositories.dataset_edits import DatasetEditRepository
from server.repositories.units import UnitRepository


###############################################################################
class DatasetEditValidationError(ValueError):
    def __init__(self, issues: list[DatasetEditIssue]) -> None:
        super().__init__(
            f"{len(issues)} change(s) could not be saved. Correct the highlighted "
            "values and try again."
        )
        self.issues = issues


###############################################################################
@dataclass
class EditPlan:
    dataset: dict[str, Any] = field(default_factory=dict)
    experiments: dict[int, dict[str, Any]] = field(default_factory=dict)
    # (experiment id, species position) -> new adsorbate id
    components: dict[tuple[int, int], int] = field(default_factory=dict)
    observations: dict[int, dict[str, Any]] = field(default_factory=dict)
    edited_experiment_ids: set[int] = field(default_factory=set)
    issues: list[DatasetEditIssue] = field(default_factory=list)

    # -------------------------------------------------------------------------
    def issue(
        self,
        scope: str,
        field_name: str,
        message: str,
        experiment_id: int | None = None,
        observation_id: int | None = None,
    ) -> None:
        self.issues.append(
            DatasetEditIssue(
                scope=scope,  # type: ignore[arg-type]
                field=field_name,
                message=message,
                experiment_id=experiment_id,
                observation_id=observation_id,
            )
        )

    # -------------------------------------------------------------------------
    def as_repository_plan(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "experiments": self.experiments,
            "components": self.components,
            "observations": self.observations,
            "edited_experiment_ids": self.edited_experiment_ids,
        }


###############################################################################
@dataclass
class ExperimentContext:
    """Resolved experiment state after applying the requested experiment changes."""

    current: dict[str, Any]
    pressure_basis: str
    pressure_unit: str | None
    uptake_unit: str | None
    # current adsorbate id -> adsorbate id after the edit
    species_map: dict[int, int]
    recompute_all: bool


###############################################################################
def is_finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


###############################################################################
class DatasetEditService:
    # -------------------------------------------------------------------------
    def __init__(self, edits: DatasetEditRepository, units: UnitRepository) -> None:
        self.edits = edits
        self.units = units

    # -------------------------------------------------------------------------
    def apply(self, dataset_id: int, request: DatasetUpdateRequest) -> None:
        plan = self.build_plan(dataset_id, request)
        if plan.issues:
            raise DatasetEditValidationError(plan.issues)
        self.edits.apply(dataset_id, plan.as_repository_plan())

    # -------------------------------------------------------------------------
    def build_plan(self, dataset_id: int, request: DatasetUpdateRequest) -> EditPlan:
        plan = EditPlan()
        self._dataset_fields(plan, request)
        snapshot = self.edits.snapshot(
            dataset_id,
            [item.id for item in request.experiments],
            [item.id for item in request.observations],
        )
        experiments: dict[int, dict[str, Any]] = snapshot["experiments"]
        owners: dict[int, int] = snapshot["observation_owners"]
        for item in request.experiments:
            if item.id not in experiments:
                plan.issue(
                    "experiment",
                    "id",
                    "This experiment is not part of the dataset.",
                    item.id,
                )
        for item in request.observations:
            if item.id not in owners:
                plan.issue(
                    "observation",
                    "id",
                    "This measurement is not part of the dataset.",
                    observation_id=item.id,
                )
        materials, species = self.edits.reference_entities(
            [item.adsorbent_id for item in request.experiments if item.adsorbent_id],
            self._species_ids(request.experiments, experiments.values()),
        )
        catalog = UnitCatalog(self.units.custom_units())
        experiment_updates = {item.id: item for item in request.experiments}
        observation_updates: dict[int, dict[int, ObservationUpdate]] = {}
        for item in request.observations:
            if item.id in owners:
                observation_updates.setdefault(owners[item.id], {})[item.id] = item
        for experiment_id in sorted(experiments):
            self._experiment(
                plan,
                experiments[experiment_id],
                experiment_updates.get(experiment_id),
                observation_updates.get(experiment_id, {}),
                materials,
                species,
                catalog,
            )
        return plan

    # -------------------------------------------------------------------------
    @staticmethod
    def _species_ids(
        updates: list[ExperimentUpdate], current: Iterable[dict[str, Any]]
    ) -> list[int]:
        ids = [
            component.adsorbate_id
            for update in updates
            for component in update.components
            if component.adsorbate_id
        ]
        ids.extend(
            component["adsorbate_id"]
            for experiment in current
            for component in experiment["components"]
        )
        return ids

    # -------------------------------------------------------------------------
    @staticmethod
    def _dataset_fields(plan: EditPlan, request: DatasetUpdateRequest) -> None:
        if request.name is not None:
            if not request.name.strip():
                plan.issue("dataset", "name", "The dataset name cannot be empty.")
            else:
                plan.dataset["name"] = request.name
        if request.description is not None:
            plan.dataset["description"] = request.description
        if request.tags is not None:
            plan.dataset["tags"] = request.tags

    # -------------------------------------------------------------------------
    def _experiment(
        self,
        plan: EditPlan,
        current: dict[str, Any],
        update: ExperimentUpdate | None,
        observation_updates: dict[int, ObservationUpdate],
        materials: set[int],
        species: dict[int, tuple[str, float | None]],
        catalog: UnitCatalog,
    ) -> None:
        experiment_id = int(current["id"])
        issue_count = len(plan.issues)
        row: dict[str, Any] = {}
        components: dict[int, int] = {}
        if update is not None:
            row = self._experiment_fields(plan, current, update, materials, catalog)
            components = self._component_changes(plan, current, update, species)
        context = self._context(current, update, row, components)
        observations = self._observations(
            plan, context, observation_updates, species, catalog
        )
        self._check_measurements(plan, context, observations)
        if len(plan.issues) > issue_count:
            return
        if row or components or observations:
            plan.experiments[experiment_id] = row
            plan.components.update(
                {
                    (experiment_id, position): value
                    for position, value in components.items()
                }
            )
            plan.observations.update(observations)
            plan.edited_experiment_ids.add(experiment_id)

    # -------------------------------------------------------------------------
    @staticmethod
    def _experiment_fields(
        plan: EditPlan,
        current: dict[str, Any],
        update: ExperimentUpdate,
        materials: set[int],
        catalog: UnitCatalog,
    ) -> dict[str, Any]:
        experiment_id = int(current["id"])
        row: dict[str, Any] = {}
        if update.name is not None:
            if update.name.strip():
                row["name"] = update.name.strip()
            else:
                plan.issue(
                    "experiment",
                    "name",
                    "The experiment name cannot be empty.",
                    experiment_id,
                )
        if update.adsorbent_id is not None:
            if update.adsorbent_id in materials:
                row["adsorbent_id"] = update.adsorbent_id
            else:
                plan.issue(
                    "experiment",
                    "adsorbent_id",
                    "Select an existing material.",
                    experiment_id,
                )
        if update.pressure_basis is not None:
            row["pressure_basis"] = update.pressure_basis
        row.update(
            DatasetEditService._temperature_fields(plan, current, update, catalog)
        )
        if "saturation_pressure_pa" in update.model_fields_set:
            value = update.saturation_pressure_pa
            if value is None or (is_finite(value) and value > 0):
                row["saturation_pressure_pa"] = value
            else:
                plan.issue(
                    "experiment",
                    "saturation_pressure_pa",
                    "Saturation pressure must be a positive number or empty.",
                    experiment_id,
                )
        DatasetEditService._check_units(plan, experiment_id, update, catalog)
        return row

    # -------------------------------------------------------------------------
    @staticmethod
    def _temperature_fields(
        plan: EditPlan,
        current: dict[str, Any],
        update: ExperimentUpdate,
        catalog: UnitCatalog,
    ) -> dict[str, Any]:
        if update.temperature_original is None and not update.temperature_original_unit:
            return {}
        value = (
            update.temperature_original
            if update.temperature_original is not None
            else current["temperature_original"]
        )
        unit = update.temperature_original_unit or current["temperature_original_unit"]
        try:
            if not is_finite(value):
                raise UnitConversionError("Temperature must be a finite number.")
            temperature = catalog.convert_temperature(float(value), unit)
            if temperature.canonical_value <= 0:
                raise UnitConversionError("Temperature must be above absolute zero.")
        except UnitConversionError as exc:
            plan.issue(
                "experiment", "temperature_original", str(exc), int(current["id"])
            )
            return {}
        return {
            "temperature_original": temperature.original_value,
            "temperature_original_unit": temperature.original_unit,
            "temperature_k": temperature.canonical_value,
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def _check_units(
        plan: EditPlan,
        experiment_id: int,
        update: ExperimentUpdate,
        catalog: UnitCatalog,
    ) -> None:
        for quantity, unit in (
            ("pressure", update.pressure_unit),
            ("uptake", update.uptake_unit),
        ):
            if unit is None:
                continue
            try:
                catalog.check(quantity, unit)
            except UnitConversionError as exc:
                plan.issue("experiment", f"{quantity}_unit", str(exc), experiment_id)

    # -------------------------------------------------------------------------
    @staticmethod
    def _component_changes(
        plan: EditPlan,
        current: dict[str, Any],
        update: ExperimentUpdate,
        species: dict[int, tuple[str, float | None]],
    ) -> dict[int, int]:
        """Return position -> new adsorbate id for each requested species change."""
        experiment_id = int(current["id"])
        existing = {
            item["position"]: item["adsorbate_id"] for item in current["components"]
        }
        changes: dict[int, int] = {}
        for component in update.components:
            if component.position not in existing:
                plan.issue(
                    "experiment",
                    "components",
                    "Species cannot be added or removed while editing.",
                    experiment_id,
                )
            elif component.adsorbate_id not in species:
                plan.issue(
                    "experiment",
                    "components",
                    "Select an existing species.",
                    experiment_id,
                )
            elif component.adsorbate_id != existing[component.position]:
                changes[component.position] = component.adsorbate_id
        merged = [changes.get(position, value) for position, value in existing.items()]
        if len(set(merged)) != len(merged):
            plan.issue(
                "experiment",
                "components",
                "Each species can appear only once in an experiment.",
                experiment_id,
            )
        return changes

    # -------------------------------------------------------------------------
    @staticmethod
    def _context(
        current: dict[str, Any],
        update: ExperimentUpdate | None,
        row: dict[str, Any],
        components: dict[int, int],
    ) -> ExperimentContext:
        species_map = {
            item["adsorbate_id"]: components.get(item["position"], item["adsorbate_id"])
            for item in current["components"]
        }
        species_changed = bool(components)
        pressure_unit = update.pressure_unit if update else None
        uptake_unit = update.uptake_unit if update else None
        return ExperimentContext(
            current=current,
            pressure_basis=row.get("pressure_basis", current["pressure_basis"]),
            pressure_unit=pressure_unit,
            uptake_unit=uptake_unit,
            species_map=species_map,
            recompute_all=bool(
                pressure_unit
                or uptake_unit
                or species_changed
                or "pressure_basis" in row
            ),
        )

    # -------------------------------------------------------------------------
    def _observations(
        self,
        plan: EditPlan,
        context: ExperimentContext,
        updates: dict[int, ObservationUpdate],
        species: dict[int, tuple[str, float | None]],
        catalog: UnitCatalog,
    ) -> dict[int, dict[str, Any]]:
        rows: dict[int, dict[str, Any]] = {}
        for current in context.current["observations"]:
            update = updates.get(current["id"])
            if update is None and not context.recompute_all:
                continue
            row = self._observation(plan, context, current, update, species, catalog)
            if row is not None:
                rows[current["id"]] = row
        return rows

    # -------------------------------------------------------------------------
    @staticmethod
    def _observation(
        plan: EditPlan,
        context: ExperimentContext,
        current: dict[str, Any],
        update: ObservationUpdate | None,
        species: dict[int, tuple[str, float | None]],
        catalog: UnitCatalog,
    ) -> dict[str, Any] | None:
        experiment_id = int(context.current["id"])
        observation_id = int(current["id"])
        sent = update.model_fields_set if update else set()
        values: dict[str, float] = {}
        for name in ("pressure_original", "uptake_original"):
            value = getattr(update, name) if name in sent else current[name]
            if not is_finite(value) or value < 0:
                plan.issue(
                    "observation",
                    name,
                    "Enter a finite value of zero or more.",
                    experiment_id,
                    observation_id,
                )
            else:
                values[name] = float(value)
        row: dict[str, Any] = {}
        if "uptake_stddev_mol_kg" in sent:
            sigma = update.uptake_stddev_mol_kg if update else None
            if sigma is None or (is_finite(sigma) and sigma > 0):
                row["uptake_stddev_mol_kg"] = sigma
            else:
                plan.issue(
                    "observation",
                    "uptake_stddev_mol_kg",
                    "Uncertainty must be a positive number or empty.",
                    experiment_id,
                    observation_id,
                )
        if "gas_mole_fraction" in sent:
            fraction = update.gas_mole_fraction if update else None
            if fraction is None or (is_finite(fraction) and 0 <= fraction <= 1):
                row["gas_mole_fraction"] = fraction
            else:
                plan.issue(
                    "observation",
                    "gas_mole_fraction",
                    "Gas mole fraction must be between 0 and 1 or empty.",
                    experiment_id,
                    observation_id,
                )
        if len(values) < 2:
            return None
        name, molar_mass = species.get(
            context.species_map[current["adsorbate_id"]], ("", None)
        )
        if molar_mass is None:
            molar_mass = KNOWN_MOLAR_MASSES_G_MOL.get(name.casefold())
        pressure = DatasetEditService._convert(
            plan,
            "pressure_original",
            experiment_id,
            observation_id,
            lambda: catalog.convert_pressure(
                values["pressure_original"],
                context.pressure_unit or current["pressure_original_unit"],
                context.pressure_basis,  # type: ignore[arg-type]
            ),
        )
        uptake = DatasetEditService._convert(
            plan,
            "uptake_original",
            experiment_id,
            observation_id,
            lambda: catalog.convert_uptake(
                values["uptake_original"],
                context.uptake_unit or current["uptake_original_unit"],
                molar_mass,
            ),
        )
        if pressure is None or uptake is None:
            return None
        return {
            **row,
            "pressure_original": pressure.original_value,
            "pressure_original_unit": pressure.original_unit,
            "pressure_canonical": pressure.canonical_value,
            "pressure_canonical_unit": pressure.canonical_unit,
            "pressure_key": pressure_key(
                pressure.canonical_value, pressure.canonical_unit
            ),
            "uptake_original": uptake.original_value,
            "uptake_original_unit": uptake.original_unit,
            "uptake_mol_kg": uptake.canonical_value,
            "conversion_metadata": {
                **current["conversion_metadata"],
                "pressure_rule": pressure.rule,
                "uptake_rule": uptake.rule,
            },
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def _convert(
        plan: EditPlan,
        field_name: str,
        experiment_id: int,
        observation_id: int,
        conversion: Any,
    ) -> ConvertedValue | None:
        try:
            return conversion()
        except UnitConversionError as exc:
            plan.issue(
                "observation", field_name, str(exc), experiment_id, observation_id
            )
            return None

    # -------------------------------------------------------------------------
    @staticmethod
    def _check_measurements(
        plan: EditPlan,
        context: ExperimentContext,
        changed: dict[int, dict[str, Any]],
    ) -> None:
        """Apply the canonical identity and composition rules to the edited state."""
        experiment_id = int(context.current["id"])
        multi_species = len(context.species_map) > 1
        if context.pressure_basis == "relative" and multi_species:
            plan.issue(
                "experiment",
                "pressure_basis",
                "Relative pressure (p/p0) applies to single-species experiments only.",
                experiment_id,
            )
        if not changed:
            return
        seen: set[tuple[int, int]] = set()
        totals: dict[int, float] = {}
        for current in context.current["observations"]:
            state = {**current, **changed.get(current["id"], {})}
            key = pressure_key(
                state["pressure_canonical"], state["pressure_canonical_unit"]
            )
            identity = (context.species_map[current["adsorbate_id"]], key)
            if identity in seen:
                plan.issue(
                    "observation",
                    "pressure_original",
                    "This species already has a measurement at this pressure.",
                    experiment_id,
                    int(current["id"]),
                )
            seen.add(identity)
            fraction = state.get("gas_mole_fraction")
            if fraction is None:
                continue
            if not multi_species and fraction != 1:
                plan.issue(
                    "observation",
                    "gas_mole_fraction",
                    "A single-species experiment has a gas mole fraction of 1 or none.",
                    experiment_id,
                    int(current["id"]),
                )
            totals[key] = totals.get(key, 0.0) + fraction
        if multi_species and any(
            total > 1 + FRACTION_TOLERANCE for total in totals.values()
        ):
            plan.issue(
                "experiment",
                "gas_mole_fraction",
                "Gas mole fractions at one pressure add up to more than 1.",
                experiment_id,
            )
