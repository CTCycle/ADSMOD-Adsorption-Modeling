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
from server.domain.datasets import (
    DatasetEditIssue,
    DatasetUpdateRequest,
    ExperimentUpdate,
    ObservationUpdate,
)
from server.repositories.dataset_edits import DatasetEditRepository
from server.repositories.units import UnitRepository

MOLE_FRACTION_TOLERANCE = 0.01


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
    components: dict[int, dict[str, Any]] = field(default_factory=dict)
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
    species_by_component: dict[int, int]
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
                plan.issue("experiment", "id", "This experiment is not part of the dataset.", item.id)
        for item in request.observations:
            if item.id not in owners:
                plan.issue(
                    "observation", "id", "This measurement is not part of the dataset.",
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
        components: dict[int, dict[str, Any]] = {}
        if update is not None:
            row = self._experiment_fields(plan, current, update, materials, catalog)
            components = self._component_changes(plan, current, update, species)
        context = self._context(current, update, row, components)
        observations = self._observations(
            plan, context, observation_updates, species, catalog
        )
        self._check_duplicates(plan, context, observations)
        if len(plan.issues) > issue_count:
            return
        if row or components or observations:
            plan.experiments[experiment_id] = row
            plan.components.update(components)
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
                    "experiment", "name", "The experiment name cannot be empty.",
                    experiment_id,
                )
        if update.adsorbent_id is not None:
            if update.adsorbent_id in materials:
                row["adsorbent_id"] = update.adsorbent_id
            else:
                plan.issue(
                    "experiment", "adsorbent_id", "Select an existing material.",
                    experiment_id,
                )
        if update.pressure_basis is not None:
            row["pressure_basis"] = update.pressure_basis
        row.update(DatasetEditService._temperature_fields(plan, current, update, catalog))
        if "saturation_pressure_pa" in update.model_fields_set:
            value = update.saturation_pressure_pa
            if value is None or (is_finite(value) and value > 0):
                row["saturation_pressure_pa"] = value
            else:
                plan.issue(
                    "experiment", "saturation_pressure_pa",
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
            plan.issue("experiment", "temperature_original", str(exc), int(current["id"]))
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
    ) -> dict[int, dict[str, Any]]:
        experiment_id = int(current["id"])
        existing = {item["id"]: item for item in current["components"]}
        changes: dict[int, dict[str, Any]] = {}
        for component in update.components:
            if component.id not in existing:
                plan.issue(
                    "experiment", "components",
                    "Species components cannot be added or removed while editing.",
                    experiment_id,
                )
                continue
            values: dict[str, Any] = {}
            if component.adsorbate_id is not None:
                if component.adsorbate_id in species:
                    values["adsorbate_id"] = component.adsorbate_id
                else:
                    plan.issue("experiment", "components", "Select an existing species.", experiment_id)
            if "mole_fraction" in component.model_fields_set:
                fraction = component.mole_fraction
                if fraction is None or (is_finite(fraction) and 0 <= fraction <= 1):
                    values["mole_fraction"] = fraction
                else:
                    plan.issue(
                        "experiment", "components",
                        "Mole fractions must be between 0 and 1.", experiment_id,
                    )
            if values:
                changes[component.id] = values
        merged = [
            {**item, **changes.get(item["id"], {})} for item in current["components"]
        ]
        if len({item["adsorbate_id"] for item in merged}) != len(merged):
            plan.issue(
                "experiment", "components",
                "Each species can appear only once in an experiment.", experiment_id,
            )
        fractions = [item["mole_fraction"] for item in merged]
        if (
            changes
            and len(merged) > 1
            and all(value is not None for value in fractions)
            and abs(sum(fractions) - 1.0) > MOLE_FRACTION_TOLERANCE
        ):
            plan.issue(
                "experiment", "components",
                "Mixture mole fractions must add up to 1.", experiment_id,
            )
        return changes

    # -------------------------------------------------------------------------
    @staticmethod
    def _context(
        current: dict[str, Any],
        update: ExperimentUpdate | None,
        row: dict[str, Any],
        components: dict[int, dict[str, Any]],
    ) -> ExperimentContext:
        species_by_component = {
            item["id"]: components.get(item["id"], {}).get(
                "adsorbate_id", item["adsorbate_id"]
            )
            for item in current["components"]
        }
        species_changed = any("adsorbate_id" in values for values in components.values())
        pressure_unit = update.pressure_unit if update else None
        uptake_unit = update.uptake_unit if update else None
        return ExperimentContext(
            current=current,
            pressure_basis=row.get("pressure_basis", current["pressure_basis"]),
            pressure_unit=pressure_unit,
            uptake_unit=uptake_unit,
            species_by_component=species_by_component,
            recompute_all=bool(
                pressure_unit or uptake_unit or species_changed or "pressure_basis" in row
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
                    "observation", name, "Enter a finite value of zero or more.",
                    experiment_id, observation_id,
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
                    "observation", "uptake_stddev_mol_kg",
                    "Uncertainty must be a positive number or empty.",
                    experiment_id, observation_id,
                )
        if len(values) < 2:
            return None
        name, molar_mass = species.get(
            context.species_by_component[current["component_id"]], ("", None)
        )
        if molar_mass is None:
            molar_mass = KNOWN_MOLAR_MASSES_G_MOL.get(name.casefold())
        pressure = DatasetEditService._convert(
            plan, "pressure_original", experiment_id, observation_id,
            lambda: catalog.convert_pressure(
                values["pressure_original"],
                context.pressure_unit or current["pressure_original_unit"],
                context.pressure_basis,  # type: ignore[arg-type]
            ),
        )
        uptake = DatasetEditService._convert(
            plan, "uptake_original", experiment_id, observation_id,
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
            plan.issue("observation", field_name, str(exc), experiment_id, observation_id)
            return None

    # -------------------------------------------------------------------------
    @staticmethod
    def _check_duplicates(
        plan: EditPlan,
        context: ExperimentContext,
        changed: dict[int, dict[str, Any]],
    ) -> None:
        if not changed or context.current["duplicate_policy"] != "reject":
            return
        seen: set[tuple[int, float]] = set()
        for current in context.current["observations"]:
            pressure = changed.get(current["id"], current)["pressure_canonical"]
            key = (current["component_id"], float(pressure))
            if key in seen:
                plan.issue(
                    "observation", "pressure_original",
                    "Another measurement of this experiment already uses this pressure.",
                    int(context.current["id"]), int(current["id"]),
                )
            seen.add(key)
