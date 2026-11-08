# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import case, delete, func, or_, select
from sqlalchemy.orm import Session, selectinload

from server.repositories.database.manager import DatabaseManager
from server.repositories.provenance import provenance_entries, references_for_records
from server.repositories.schemas.models import (
    Adsorbate,
    Adsorbent,
    Dataset,
    DatasetImport,
    FittingSession,
    Isotherm,
    IsothermComponent,
    Observation,
    utc_now,
)
from server.repositories.schemas.types import normalize_identity

SUMMARY_NAME_LIMIT = 3


###############################################################################
def stable_material_key(kind: str, name: str, external: str | None = None) -> str:
    identity = normalize_identity(external or name)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
    return f"{kind}:{digest}"


###############################################################################
def resolve_adsorbent(
    session: Session, cache: dict[str, Adsorbent], record: dict[str, Any]
) -> Adsorbent:
    external = record.get("external_identifier")
    key = record.get("key") or stable_material_key("adsorbent", record["name"], external)
    adsorbent = cache.get(key) or session.scalar(
        select(Adsorbent).where(Adsorbent.key == key)
    )
    if adsorbent is None and external:
        adsorbent = session.scalar(
            select(Adsorbent).where(Adsorbent.external_identifier == external)
        )
    if adsorbent is None:
        adsorbent = Adsorbent(
            key=key, name=record["name"].strip(), external_identifier=external
        )
        session.add(adsorbent)
        session.flush()
    cache[key] = adsorbent
    return adsorbent


###############################################################################
def resolve_adsorbate(
    session: Session, cache: dict[str, Adsorbate], record: dict[str, Any]
) -> Adsorbate:
    inchi_key = record.get("inchi_key")
    key = record.get("key") or stable_material_key("adsorbate", record["name"], inchi_key)
    adsorbate = cache.get(key) or session.scalar(
        select(Adsorbate).where(Adsorbate.key == key)
    )
    if adsorbate is None and inchi_key:
        adsorbate = session.scalar(
            select(Adsorbate).where(Adsorbate.inchi_key == inchi_key)
        )
    if adsorbate is None:
        adsorbate = Adsorbate(
            key=key,
            name=record["name"].strip(),
            inchi_key=inchi_key,
            inchi=record.get("inchi"),
            formula=record.get("formula"),
            molar_mass_g_mol=record.get("molar_mass_g_mol"),
            smiles=record.get("smiles"),
        )
        session.add(adsorbate)
        session.flush()
    elif record.get("smiles") and not adsorbate.smiles:
        adsorbate.smiles = record["smiles"]
    cache[key] = adsorbate
    return adsorbate


###############################################################################
def write_experiment_content(
    session: Session,
    isotherm: Isotherm,
    record: dict[str, Any],
    adsorbates_by_key: dict[str, Adsorbate],
) -> None:
    """Insert the components and observations of one canonical experiment."""
    component_by_name: dict[str, IsothermComponent] = {}
    for position, adsorbate_record in enumerate(record["adsorbates"], start=1):
        adsorbate = resolve_adsorbate(session, adsorbates_by_key, adsorbate_record)
        component = IsothermComponent(
            isotherm_id=isotherm.id,
            position=position,
            adsorbate_id=adsorbate.id,
            mole_fraction=adsorbate_record.get("mole_fraction"),
        )
        session.add(component)
        session.flush()
        component_by_name[normalize_identity(adsorbate_record["name"])] = component

    for observation in record["observations"]:
        component = component_by_name[normalize_identity(observation["adsorbate"])]
        session.add(
            Observation(
                isotherm_id=isotherm.id,
                component_id=component.id,
                sequence_index=observation["sequence_index"],
                source_row=observation.get("source_row"),
                pressure_original=observation["pressure_original"],
                pressure_original_unit=observation["pressure_original_unit"],
                pressure_canonical=observation["pressure_canonical"],
                pressure_canonical_unit=observation["pressure_canonical_unit"],
                uptake_original=observation["uptake_original"],
                uptake_original_unit=observation["uptake_original_unit"],
                uptake_mol_kg=observation["uptake_mol_kg"],
                uptake_stddev_mol_kg=observation.get("uptake_stddev_mol_kg"),
                conversion_metadata=observation.get("conversion_metadata", {}),
                extra_metadata=observation.get("extra_metadata", {}),
            )
        )


###############################################################################
def isotherm_fields(record: dict[str, Any], adsorbent_id: int) -> dict[str, Any]:
    return {
        "name": record["name"],
        "adsorbent_id": adsorbent_id,
        "temperature_original": record["temperature_original"],
        "temperature_original_unit": record["temperature_original_unit"],
        "temperature_k": record["temperature_k"],
        "pressure_basis": record["pressure_basis"],
        "duplicate_policy": record.get("duplicate_policy", "reject"),
        "saturation_pressure_pa": record.get("saturation_pressure_pa"),
        "conditions": record.get("conditions", {}),
        "provenance": record.get("provenance", {}),
    }


###############################################################################
def iso_or_none(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


###############################################################################
class DatasetRepository:

    # -------------------------------------------------------------------------
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    # -------------------------------------------------------------------------
    def list_summaries(self, dataset_id: int | None = None) -> list[dict[str, Any]]:
        statement = (
            select(
                Dataset,
                func.count(func.distinct(Isotherm.id)).label("experiment_count"),
                func.count(Observation.id).label("observation_count"),
                func.count(
                    func.distinct(
                        case((Isotherm.edited_at.is_not(None), Isotherm.id))
                    )
                ).label("edited_experiment_count"),
            )
            .outerjoin(Isotherm, Isotherm.dataset_id == Dataset.id)
            .outerjoin(Observation, Observation.isotherm_id == Isotherm.id)
            .group_by(Dataset.id)
            .order_by(Dataset.created_at.desc(), Dataset.id.desc())
        )
        if dataset_id is not None:
            statement = statement.where(Dataset.id == dataset_id)
        with self.database.session_factory() as session:
            rows = session.execute(statement).all()
            ids = [dataset.id for dataset, *_ in rows]
            materials = self._ranked_names(
                session,
                select(
                    Isotherm.dataset_id.label("dataset_id"),
                    Adsorbent.name.label("name"),
                    func.count(Isotherm.id).label("uses"),
                )
                .join(Adsorbent, Adsorbent.id == Isotherm.adsorbent_id)
                .where(Isotherm.dataset_id.in_(ids))
                .group_by(Isotherm.dataset_id, Adsorbent.id, Adsorbent.name),
            )
            species = self._ranked_names(
                session,
                select(
                    Isotherm.dataset_id.label("dataset_id"),
                    Adsorbate.name.label("name"),
                    func.count(Isotherm.id).label("uses"),
                )
                .join(IsothermComponent, IsothermComponent.isotherm_id == Isotherm.id)
                .join(Adsorbate, Adsorbate.id == IsothermComponent.adsorbate_id)
                .where(Isotherm.dataset_id.in_(ids))
                .group_by(Isotherm.dataset_id, Adsorbate.id, Adsorbate.name),
            )
            return [
                {
                    "id": dataset.id,
                    "name": dataset.name,
                    "source": dataset.source,
                    "created_at": dataset.created_at.isoformat(),
                    "last_update": dataset.updated_at.isoformat(),
                    "experiment_count": int(experiment_count or 0),
                    "observation_count": int(observation_count or 0),
                    "edited_experiment_count": int(edited_count or 0),
                    "material_count": materials.get(dataset.id, (0, []))[0],
                    "material_names": materials.get(dataset.id, (0, []))[1],
                    "species_count": species.get(dataset.id, (0, []))[0],
                    "species_names": species.get(dataset.id, (0, []))[1],
                    "tags": list(dataset.tags or []),
                    "description": dataset.description or "",
                }
                for dataset, experiment_count, observation_count, edited_count in rows
            ]

    # -------------------------------------------------------------------------
    @staticmethod
    def _ranked_names(
        session: Session, grouped: Any
    ) -> dict[int, tuple[int, list[str]]]:
        """Return (distinct count, most used names) per dataset in one query."""
        usage = grouped.subquery()
        ranked = select(
            usage.c.dataset_id,
            usage.c.name,
            func.row_number()
            .over(
                partition_by=usage.c.dataset_id,
                order_by=(usage.c.uses.desc(), usage.c.name),
            )
            .label("rank"),
            func.count().over(partition_by=usage.c.dataset_id).label("total"),
        ).subquery()
        result: dict[int, tuple[int, list[str]]] = {}
        for dataset_id, name, _, total in session.execute(
            select(ranked).where(ranked.c.rank <= SUMMARY_NAME_LIMIT).order_by(
                ranked.c.dataset_id, ranked.c.rank
            )
        ):
            count, names = result.get(int(dataset_id), (int(total), []))
            result[int(dataset_id)] = (count, [*names, str(name)])
        return result

    # -------------------------------------------------------------------------
    def summary(self, dataset_id: int) -> dict[str, Any]:
        rows = self.list_summaries(dataset_id)
        if not rows:
            raise LookupError(f"Dataset {dataset_id} does not exist.")
        return rows[0]

    # -------------------------------------------------------------------------
    def detail(self, dataset_id: int) -> dict[str, Any]:
        summary = self.summary(dataset_id)
        with self.database.session_factory() as session:
            dataset = session.get(Dataset, dataset_id)
            if dataset is None:
                raise LookupError(f"Dataset {dataset_id} does not exist.")
            latest = session.scalar(
                select(DatasetImport)
                .where(DatasetImport.dataset_id == dataset_id)
                .order_by(DatasetImport.created_at.desc(), DatasetImport.id.desc())
            )
            return {
                "dataset": summary,
                "provenance": dict(dataset.provenance or {}),
                "latest_import": (
                    {
                        "original_filename": latest.original_filename,
                        "source_structure": latest.source_structure,
                        "parser_version": latest.parser_version,
                        "created_at": latest.created_at.isoformat(),
                    }
                    if latest is not None
                    else None
                ),
            }

    # -------------------------------------------------------------------------
    def delete_impact(self, dataset_id: int) -> dict[str, Any]:
        summary = self.summary(dataset_id)
        with self.database.session_factory() as session:
            runs = session.scalar(
                select(func.count(FittingSession.id)).where(
                    FittingSession.dataset_id == dataset_id
                )
            )
        return {
            "dataset_id": dataset_id,
            "experiment_count": summary["experiment_count"],
            "observation_count": summary["observation_count"],
            "fitting_session_count": int(runs or 0),
        }

    # -------------------------------------------------------------------------
    def import_inspection(self, dataset_id: int) -> dict[str, Any]:
        statement = (
            select(DatasetImport)
            .where(DatasetImport.dataset_id == dataset_id)
            .order_by(DatasetImport.created_at.desc(), DatasetImport.id.desc())
        )
        with self.database.session_factory() as session:
            if session.get(Dataset, dataset_id) is None:
                raise LookupError(f"Dataset {dataset_id} does not exist.")
            record = session.scalar(statement)
            if record is None:
                raise LookupError(f"Dataset {dataset_id} has no import manifest.")
            return {
                "dataset_id": dataset_id,
                "original_filename": record.original_filename,
                "source_sha256": record.source_sha256,
                "source_structure": record.source_structure,
                "parser_version": record.parser_version,
                "mapping": dict(record.column_mapping),
                "validation": dict(record.validation_result),
                "warnings": list(record.warnings or []),
            }

    # -------------------------------------------------------------------------
    def observation_frame(self, dataset_id: int) -> pd.DataFrame:
        """Return one canonical row per persisted adsorption observation."""
        statement = (
            select(
                Dataset.id.label("dataset_id"),
                Dataset.name.label("dataset_name"),
                Isotherm.id.label("isotherm_id"),
                Isotherm.external_key.label("experiment"),
                Isotherm.temperature_k.label("temperature"),
                Adsorbent.name.label("adsorbent_name"),
                Adsorbate.name.label("adsorbate_name"),
                Adsorbate.molar_mass_g_mol.label("adsorbate_molecular_weight"),
                Adsorbate.smiles.label("adsorbate_SMILE"),
                Observation.component_id,
                Observation.pressure_canonical_unit,
                Observation.pressure_canonical.label("pressure"),
                Observation.uptake_mol_kg.label("adsorbed_amount"),
            )
            .join(Isotherm, Isotherm.dataset_id == Dataset.id)
            .join(Adsorbent, Adsorbent.id == Isotherm.adsorbent_id)
            .join(Observation, Observation.isotherm_id == Isotherm.id)
            .join(IsothermComponent, IsothermComponent.id == Observation.component_id)
            .join(Adsorbate, Adsorbate.id == IsothermComponent.adsorbate_id)
            .where(Dataset.id == dataset_id)
            .order_by(Isotherm.id, Observation.sequence_index, Observation.id)
        )
        with self.database.session_factory() as session:
            if session.get(Dataset, dataset_id) is None:
                raise LookupError(f"Dataset {dataset_id} does not exist.")
            rows = session.execute(statement).mappings()
            return pd.DataFrame([dict(row) for row in rows])

    # -------------------------------------------------------------------------
    def delete(self, dataset_id: int) -> None:
        with self.database.transaction() as session:
            result = session.execute(delete(Dataset).where(Dataset.id == dataset_id))
            if not result.rowcount:
                raise LookupError(f"Dataset {dataset_id} does not exist.")

    # -------------------------------------------------------------------------
    def experiments(
        self,
        dataset_id: int,
        *,
        offset: int = 0,
        limit: int = 50,
        query_text: str | None = None,
        adsorbent_id: int | None = None,
        adsorbate_id: int | None = None,
        temperature_min_k: float | None = None,
        temperature_max_k: float | None = None,
        edited_only: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        filtered = select(Isotherm.id).where(Isotherm.dataset_id == dataset_id)
        if query_text:
            pattern = f"%{query_text.strip()}%"
            filtered = filtered.where(
                or_(
                    Isotherm.name.ilike(pattern),
                    Isotherm.external_key.ilike(pattern),
                    Isotherm.adsorbent_id.in_(
                        select(Adsorbent.id).where(Adsorbent.name.ilike(pattern))
                    ),
                    Isotherm.id.in_(
                        select(IsothermComponent.isotherm_id)
                        .join(Adsorbate, Adsorbate.id == IsothermComponent.adsorbate_id)
                        .where(Adsorbate.name.ilike(pattern))
                    ),
                )
            )
        if adsorbent_id is not None:
            filtered = filtered.where(Isotherm.adsorbent_id == adsorbent_id)
        if adsorbate_id is not None:
            filtered = filtered.where(
                Isotherm.id.in_(
                    select(IsothermComponent.isotherm_id).where(
                        IsothermComponent.adsorbate_id == adsorbate_id
                    )
                )
            )
        if temperature_min_k is not None:
            filtered = filtered.where(Isotherm.temperature_k >= temperature_min_k)
        if temperature_max_k is not None:
            filtered = filtered.where(Isotherm.temperature_k <= temperature_max_k)
        if edited_only:
            filtered = filtered.where(Isotherm.edited_at.is_not(None))
        with self.database.session_factory() as session:
            if session.get(Dataset, dataset_id) is None:
                raise LookupError(f"Dataset {dataset_id} does not exist.")
            total = int(
                session.scalar(select(func.count()).select_from(filtered.subquery()))
                or 0
            )
            isotherms = list(
                session.scalars(
                    select(Isotherm)
                    .where(Isotherm.id.in_(filtered))
                    .options(
                        selectinload(Isotherm.adsorbent),
                        selectinload(Isotherm.components).selectinload(
                            IsothermComponent.adsorbate
                        ),
                    )
                    .order_by(Isotherm.name, Isotherm.id)
                    .offset(offset)
                    .limit(limit)
                ).unique()
            )
            return self._experiment_payloads(session, isotherms), total

    # -------------------------------------------------------------------------
    def experiment(self, dataset_id: int, experiment_id: int) -> dict[str, Any]:
        with self.database.session_factory() as session:
            isotherm = session.scalar(
                select(Isotherm)
                .where(Isotherm.id == experiment_id, Isotherm.dataset_id == dataset_id)
                .options(
                    selectinload(Isotherm.adsorbent),
                    selectinload(Isotherm.components).selectinload(
                        IsothermComponent.adsorbate
                    ),
                )
            )
            if isotherm is None:
                raise LookupError(
                    f"Experiment {experiment_id} does not belong to dataset {dataset_id}."
                )
            return self._experiment_payloads(session, [isotherm])[0]

    # -------------------------------------------------------------------------
    def experiment_detail(self, dataset_id: int, experiment_id: int) -> dict[str, Any]:
        summary = self.experiment(dataset_id, experiment_id)
        with self.database.session_factory() as session:
            row = session.execute(
                select(Isotherm.conditions, Dataset.name)
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .where(Isotherm.id == experiment_id)
            ).one()
            entries = provenance_entries(session, "isotherm", [experiment_id]).get(
                experiment_id, []
            )
            references = references_for_records(
                session, [record_id for record_id, _ in entries]
            )
            return {
                "dataset_name": row.name,
                "experiment": summary,
                "conditions": dict(row.conditions or {}),
                "provenance": [
                    {
                        "source": entry["source"],
                        "external_id": entry["external_id"],
                        "source_url": entry["source_url"],
                        "doi": [
                            doi
                            for doi in [references.get(record_id, {}).get("doi")]
                            if doi
                        ],
                    }
                    for record_id, entry in entries
                ],
            }

    # -------------------------------------------------------------------------
    @staticmethod
    def _experiment_payloads(
        session: Session, isotherms: list[Isotherm]
    ) -> list[dict[str, Any]]:
        ids = [isotherm.id for isotherm in isotherms]
        stats = {
            row[0]: row[1:]
            for row in session.execute(
                select(
                    Observation.isotherm_id,
                    func.count(Observation.id),
                    func.min(Observation.pressure_canonical),
                    func.max(Observation.pressure_canonical),
                    func.min(Observation.pressure_canonical_unit),
                    func.min(Observation.pressure_original_unit),
                    func.min(Observation.uptake_original_unit),
                )
                .where(Observation.isotherm_id.in_(ids))
                .group_by(Observation.isotherm_id)
            )
        } if ids else {}
        output: list[dict[str, Any]] = []
        for isotherm in isotherms:
            count, pressure_min, pressure_max, canonical_unit, pressure_unit, uptake_unit = (
                stats.get(isotherm.id, (0, None, None, None, None, None))
            )
            components = sorted(isotherm.components, key=lambda item: item.position)
            reason = None
            if len(components) != 1:
                reason = "The available theoretical models are single-component models."
            elif count < 2:
                reason = "At least two observations are required."
            output.append(
                {
                    "id": isotherm.id,
                    "dataset_id": isotherm.dataset_id,
                    "external_key": isotherm.external_key,
                    "name": isotherm.name,
                    "adsorbent_id": isotherm.adsorbent_id,
                    "adsorbent": isotherm.adsorbent.name,
                    "adsorbates": [item.adsorbate.name for item in components],
                    "components": [
                        {
                            "id": item.id,
                            "position": item.position,
                            "adsorbate_id": item.adsorbate_id,
                            "name": item.adsorbate.name,
                            "mole_fraction": item.mole_fraction,
                        }
                        for item in components
                    ],
                    "temperature_k": isotherm.temperature_k,
                    "temperature_original": isotherm.temperature_original,
                    "temperature_original_unit": isotherm.temperature_original_unit,
                    "pressure_basis": isotherm.pressure_basis,
                    "saturation_pressure_pa": isotherm.saturation_pressure_pa,
                    "duplicate_policy": isotherm.duplicate_policy,
                    "observation_count": count,
                    "pressure_min": pressure_min,
                    "pressure_max": pressure_max,
                    "pressure_canonical_unit": canonical_unit,
                    "pressure_original_unit": pressure_unit,
                    "uptake_original_unit": uptake_unit,
                    "edited_at": iso_or_none(isotherm.edited_at),
                    "fitting_eligible": reason is None,
                    "ineligibility_reason": reason,
                }
            )
        return output

    # -------------------------------------------------------------------------
    def observations(
        self,
        dataset_id: int,
        isotherm_id: int,
        offset: int = 0,
        limit: int = 100,
        order: str = "pressure",
    ) -> tuple[list[dict[str, Any]], int]:
        base = (
            select(Observation)
            .join(Isotherm, Isotherm.id == Observation.isotherm_id)
            .where(
                Isotherm.dataset_id == dataset_id,
                Observation.isotherm_id == isotherm_id,
            )
        )
        ordering = (
            (Observation.sequence_index, Observation.component_id, Observation.id)
            if order == "sequence"
            else (
                Observation.pressure_canonical,
                Observation.sequence_index,
                Observation.id,
            )
        )
        with self.database.session_factory() as session:
            total = int(
                session.scalar(select(func.count()).select_from(base.subquery())) or 0
            )
            rows = list(
                session.scalars(base.order_by(*ordering).offset(offset).limit(limit))
            )
            if not rows and session.get(Isotherm, isotherm_id) is None:
                raise LookupError(f"Isotherm {isotherm_id} does not exist.")
            return (
                [
                    {
                        "id": row.id,
                        "component_id": row.component_id,
                        "sequence_index": row.sequence_index,
                        "source_row": row.source_row,
                        "pressure_original": row.pressure_original,
                        "pressure_original_unit": row.pressure_original_unit,
                        "pressure_canonical": row.pressure_canonical,
                        "pressure_canonical_unit": row.pressure_canonical_unit,
                        "uptake_original": row.uptake_original,
                        "uptake_original_unit": row.uptake_original_unit,
                        "uptake_mol_kg": row.uptake_mol_kg,
                        "uptake_stddev_mol_kg": row.uptake_stddev_mol_kg,
                    }
                    for row in rows
                ],
                total,
            )

    # -------------------------------------------------------------------------
    def experiment_identities(self, dataset_id: int) -> tuple[str, list[tuple[int, str]]]:
        """Return the dataset name and every experiment ``(id, name)`` it owns."""
        with self.database.session_factory() as session:
            dataset = session.get(Dataset, dataset_id)
            if dataset is None:
                raise LookupError(f"Dataset {dataset_id} does not exist.")
            rows = session.execute(
                select(Isotherm.id, Isotherm.name)
                .where(Isotherm.dataset_id == dataset_id)
                .order_by(Isotherm.name, Isotherm.id)
            ).all()
            return dataset.name, [(int(row.id), str(row.name)) for row in rows]

    # -------------------------------------------------------------------------
    def fitting_series(self, dataset_id: int, isotherm_id: int) -> dict[str, Any]:
        statement = (
            select(Isotherm)
            .where(Isotherm.id == isotherm_id, Isotherm.dataset_id == dataset_id)
            .options(
                selectinload(Isotherm.dataset),
                selectinload(Isotherm.adsorbent),
                selectinload(Isotherm.components).selectinload(
                    IsothermComponent.adsorbate
                ),
                selectinload(Isotherm.observations),
            )
        )
        with self.database.session_factory() as session:
            isotherm = session.scalar(statement)
            if isotherm is None:
                raise LookupError(
                    f"Isotherm {isotherm_id} does not belong to dataset {dataset_id}."
                )
            components = sorted(isotherm.components, key=lambda item: item.position)
            if len(components) != 1:
                raise ValueError(
                    "The available theoretical models require a single-component isotherm."
                )
            component = components[0]
            observations = sorted(
                (
                    item
                    for item in isotherm.observations
                    if item.component_id == component.id
                ),
                key=lambda item: (
                    item.pressure_canonical,
                    item.sequence_index,
                    item.id,
                ),
            )
            if len(observations) < 2:
                raise ValueError(
                    "At least two canonical observations are required for fitting."
                )
            if isotherm.duplicate_policy == "average":
                grouped: dict[float, list[Observation]] = {}
                for item in observations:
                    grouped.setdefault(float(item.pressure_canonical), []).append(item)
                averaged: list[tuple[float, float, float | None]] = []
                source_ids: list[int] = []
                for pressure, group in sorted(grouped.items()):
                    source_ids.extend(item.id for item in group)
                    values = [float(item.uptake_mol_kg) for item in group]
                    sigmas = [item.uptake_stddev_mol_kg for item in group]
                    sigma = (
                        float(np.sqrt(np.sum(np.square(sigmas))) / len(sigmas))
                        if sigmas
                        and all(value is not None and value > 0 for value in sigmas)
                        else None
                    )
                    averaged.append((pressure, float(np.mean(values)), sigma))
                pressure_values = [item[0] for item in averaged]
                uptake_values = [item[1] for item in averaged]
                sigma_values = [item[2] for item in averaged]
            else:
                source_ids = [item.id for item in observations]
                pressure_values = [item.pressure_canonical for item in observations]
                uptake_values = [item.uptake_mol_kg for item in observations]
                sigma_values = [item.uptake_stddev_mol_kg for item in observations]
            return {
                "dataset_id": isotherm.dataset.id,
                "dataset_name": isotherm.dataset.name,
                "isotherm_id": isotherm.id,
                "isotherm_name": isotherm.name,
                "component_id": component.id,
                "adsorbent": isotherm.adsorbent.name,
                "adsorbate": component.adsorbate.name,
                "adsorbate_molar_mass_g_mol": component.adsorbate.molar_mass_g_mol,
                "temperature_k": isotherm.temperature_k,
                "pressure_basis": isotherm.pressure_basis,
                "saturation_pressure_pa": isotherm.saturation_pressure_pa,
                "observation_ids": source_ids,
                "pressure": pressure_values,
                "uptake": uptake_values,
                "uptake_stddev": sigma_values,
                "duplicate_policy": isotherm.duplicate_policy,
            }

    # -------------------------------------------------------------------------
    def persist_canonical(
        self,
        *,
        name: str,
        source: str,
        provenance: dict[str, Any],
        experiments: list[dict[str, Any]],
        import_manifest: dict[str, Any] | None = None,
        dataset_id: int | None = None,
    ) -> int:
        normalized_name = normalize_identity(name)
        with self.database.transaction() as session:
            if dataset_id is None:
                if (
                    session.scalar(
                        select(Dataset.id).where(
                            Dataset.normalized_name == normalized_name
                        )
                    )
                    is not None
                ):
                    raise ValueError(f"A dataset named '{name}' already exists.")
                dataset = Dataset(
                    name=name.strip(),
                    normalized_name=normalized_name,
                    source=source,
                    provenance=provenance,
                )
                session.add(dataset)
                session.flush()
            else:
                dataset = session.get(Dataset, dataset_id)
                if dataset is None or dataset.source != source:
                    raise LookupError(
                        f"Canonical {source} dataset {dataset_id} does not exist."
                    )
                if experiments:
                    dataset.updated_at = utc_now()

            if import_manifest is not None:
                session.add(DatasetImport(dataset_id=dataset.id, **import_manifest))

            adsorbates_by_key: dict[str, Adsorbate] = {}
            adsorbents_by_key: dict[str, Adsorbent] = {}
            for record in experiments:
                adsorbent = resolve_adsorbent(
                    session, adsorbents_by_key, record["adsorbent"]
                )
                isotherm = Isotherm(
                    dataset_id=dataset.id,
                    external_key=record["external_key"],
                    **isotherm_fields(record, adsorbent.id),
                )
                session.add(isotherm)
                session.flush()
                write_experiment_content(session, isotherm, record, adsorbates_by_key)
            session.flush()
            return dataset.id
