# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""Persistence for in-place dataset edits and restoring experiments from source."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from server.repositories.database.manager import DatabaseManager
from server.repositories.datasets import (
    isotherm_fields,
    resolve_adsorbent,
    write_experiment_content,
)
from server.repositories.identity import IdentityResolver
from server.repositories.schemas.models import (
    Adsorbate,
    Adsorbent,
    Dataset,
    Isotherm,
    IsothermComponent,
    Observation,
    utc_now,
)
from server.repositories.schemas.types import normalize_identity


###############################################################################
class DatasetEditRepository:
    # -------------------------------------------------------------------------
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    # -------------------------------------------------------------------------
    def snapshot(
        self,
        dataset_id: int,
        experiment_ids: Iterable[int],
        observation_ids: Iterable[int],
    ) -> dict[str, Any]:
        """Load the current state of every experiment touched by an edit.

        Observations are resolved to their owning experiments, and every
        observation of a touched experiment is returned so experiment-level
        changes (units, species, basis) can be recomputed consistently.
        """
        requested_observations = list(set(observation_ids))
        with self.database.session_factory() as session:
            dataset = session.get(Dataset, dataset_id)
            if dataset is None:
                raise LookupError(f"Dataset {dataset_id} does not exist.")
            owners: dict[int, int] = {}
            if requested_observations:
                owners = {
                    int(observation_id): int(isotherm_id)
                    for observation_id, isotherm_id in session.execute(
                        select(Observation.id, Observation.isotherm_id)
                        .join(Isotherm, Isotherm.id == Observation.isotherm_id)
                        .where(
                            Observation.id.in_(requested_observations),
                            Isotherm.dataset_id == dataset_id,
                        )
                    )
                }
            touched = set(experiment_ids) | set(owners.values())
            isotherms = (
                list(
                    session.scalars(
                        select(Isotherm)
                        .where(
                            Isotherm.id.in_(touched), Isotherm.dataset_id == dataset_id
                        )
                        .options(
                            selectinload(Isotherm.components).selectinload(
                                IsothermComponent.adsorbate
                            ),
                            selectinload(Isotherm.observations),
                        )
                    ).unique()
                )
                if touched
                else []
            )
            return {
                "dataset": {"id": dataset.id, "name": dataset.name},
                "observation_owners": owners,
                "experiments": {
                    isotherm.id: {
                        "id": isotherm.id,
                        "name": isotherm.name,
                        "adsorbent_id": isotherm.adsorbent_id,
                        "temperature_original": isotherm.temperature_original,
                        "temperature_original_unit": isotherm.temperature_original_unit,
                        "temperature_k": isotherm.temperature_k,
                        "pressure_basis": isotherm.pressure_basis,
                        "saturation_pressure_pa": isotherm.saturation_pressure_pa,
                        "components": [
                            {
                                "position": component.position,
                                "adsorbate_id": component.adsorbate_id,
                            }
                            for component in sorted(
                                isotherm.components, key=lambda item: item.position
                            )
                        ],
                        "observations": [
                            {
                                "id": item.id,
                                "adsorbate_id": item.adsorbate_id,
                                "branch": item.branch,
                                "pressure_original": item.pressure_original,
                                "pressure_original_unit": item.pressure_original_unit,
                                "pressure_canonical": item.pressure_canonical,
                                "pressure_canonical_unit": item.pressure_canonical_unit,
                                "gas_mole_fraction": item.gas_mole_fraction,
                                "uptake_original": item.uptake_original,
                                "uptake_original_unit": item.uptake_original_unit,
                                "uptake_stddev_mol_kg": item.uptake_stddev_mol_kg,
                                "conversion_metadata": dict(
                                    item.conversion_metadata or {}
                                ),
                            }
                            for item in sorted(
                                isotherm.observations, key=lambda item: item.id
                            )
                        ],
                    }
                    for isotherm in isotherms
                },
            }

    # -------------------------------------------------------------------------
    def reference_entities(
        self, adsorbent_ids: Iterable[int], adsorbate_ids: Iterable[int]
    ) -> tuple[set[int], dict[int, tuple[str, float | None]]]:
        """Return existing material ids and species (name, molar mass) by id."""
        material_ids = list(set(adsorbent_ids))
        species_ids = list(set(adsorbate_ids))
        with self.database.session_factory() as session:
            materials = (
                set(
                    session.scalars(
                        select(Adsorbent.id).where(Adsorbent.id.in_(material_ids))
                    )
                )
                if material_ids
                else set()
            )
            species = (
                {
                    int(identifier): (str(name), molar_mass)
                    for identifier, name, molar_mass in session.execute(
                        select(
                            Adsorbate.id, Adsorbate.name, Adsorbate.molar_mass_g_mol
                        ).where(Adsorbate.id.in_(species_ids))
                    )
                }
                if species_ids
                else {}
            )
            return materials, species

    # -------------------------------------------------------------------------
    def apply(self, dataset_id: int, plan: dict[str, Any]) -> None:
        """Write a validated edit plan atomically; any failure rolls back all rows."""
        experiment_rows: dict[int, dict[str, Any]] = plan["experiments"]
        component_rows: dict[tuple[int, int], int] = plan["components"]
        observation_rows: dict[int, dict[str, Any]] = plan["observations"]
        edited_ids: set[int] = set(plan["edited_experiment_ids"])
        now = utc_now()
        with self.database.transaction() as session:
            dataset = session.get(Dataset, dataset_id)
            if dataset is None:
                raise LookupError(f"Dataset {dataset_id} does not exist.")
            self._apply_dataset_fields(session, dataset, plan["dataset"])
            owned = set(
                session.scalars(
                    select(Isotherm.id).where(
                        Isotherm.id.in_(edited_ids), Isotherm.dataset_id == dataset_id
                    )
                )
            )
            if owned != edited_ids:
                raise LookupError(
                    "Edited experiments no longer belong to this dataset."
                )
            if experiment_rows or edited_ids:
                session.execute(
                    update(Isotherm),
                    [
                        {
                            "id": isotherm_id,
                            **experiment_rows.get(isotherm_id, {}),
                            "edited_at": now,
                        }
                        for isotherm_id in sorted(edited_ids)
                    ],
                )
            # Measurements follow a species change through the ON UPDATE
            # CASCADE of their (isotherm_id, adsorbate_id) foreign key.
            for (isotherm_id, position), adsorbate_id in sorted(component_rows.items()):
                session.execute(
                    update(IsothermComponent)
                    .where(
                        IsothermComponent.isotherm_id == isotherm_id,
                        IsothermComponent.position == position,
                    )
                    .values(adsorbate_id=adsorbate_id)
                )
            if observation_rows:
                session.execute(
                    update(Observation),
                    [{"id": key, **values} for key, values in observation_rows.items()],
                )
            dataset.updated_at = now
            try:
                session.flush()
            except IntegrityError as exc:
                raise ValueError(
                    "The edit would give a species two measurements on the same "
                    "branch at the same pressure or list a species twice; nothing "
                    "was saved."
                ) from exc

    # -------------------------------------------------------------------------
    @staticmethod
    def _apply_dataset_fields(
        session: Any, dataset: Dataset, values: dict[str, Any]
    ) -> None:
        if "name" in values:
            normalized = normalize_identity(values["name"])
            conflict = session.scalar(
                select(Dataset.id).where(
                    Dataset.normalized_name == normalized, Dataset.id != dataset.id
                )
            )
            if conflict is not None:
                raise ValueError(f"A dataset named '{values['name']}' already exists.")
            dataset.name = values["name"].strip()
            dataset.normalized_name = normalized
        if "description" in values:
            dataset.description = values["description"].strip()
        if "tags" in values:
            dataset.tags = sorted(
                {tag.strip() for tag in values["tags"] if tag.strip()}
            )

    # -------------------------------------------------------------------------
    def edited_external_keys(self, dataset_id: int) -> set[str]:
        with self.database.session_factory() as session:
            return {
                str(key).casefold()
                for key in session.scalars(
                    select(Isotherm.external_key).where(
                        Isotherm.dataset_id == dataset_id,
                        Isotherm.edited_at.is_not(None),
                    )
                )
            }

    # -------------------------------------------------------------------------
    def edited_count(self, dataset_id: int) -> int:
        with self.database.session_factory() as session:
            return int(
                session.scalar(
                    select(func.count(Isotherm.id)).where(
                        Isotherm.dataset_id == dataset_id,
                        Isotherm.edited_at.is_not(None),
                    )
                )
                or 0
            )

    # -------------------------------------------------------------------------
    def replace_from_source(self, dataset_id: int, record: dict[str, Any]) -> bool:
        """Overwrite one experiment with its source content, keeping its id.

        Keeping the id preserves fitting history; `edited_at` is cleared because
        the experiment matches its source again.
        """
        with self.database.transaction() as session:
            isotherm = session.scalar(
                select(Isotherm).where(
                    Isotherm.dataset_id == dataset_id,
                    func.lower(Isotherm.external_key)
                    == str(record["external_key"]).casefold(),
                )
            )
            if isotherm is None:
                return False
            session.execute(
                delete(Observation).where(Observation.isotherm_id == isotherm.id)
            )
            session.execute(
                delete(IsothermComponent).where(
                    IsothermComponent.isotherm_id == isotherm.id
                )
            )
            resolver = IdentityResolver(
                session, dataset_id=dataset_id, alias_origin="nist"
            )
            adsorbent = resolve_adsorbent(session, resolver, record["adsorbent"])
            for key, value in isotherm_fields(record, adsorbent.id).items():
                setattr(isotherm, key, value)
            isotherm.edited_at = None
            session.flush()
            write_experiment_content(session, isotherm, record, resolver)
            dataset = session.get(Dataset, dataset_id)
            if dataset is not None:
                dataset.updated_at = utc_now()
            session.flush()
            return True
