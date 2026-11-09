# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sqlalchemy import case, delete, func, or_, select
from sqlalchemy.orm import Session

from server.repositories.database.bulk import upsert_records
from server.repositories.database.manager import DatabaseManager
from server.repositories.provenance import (
    counts_by,
    provenance_entries,
    public_adsorbate_ids,
    public_adsorbent_ids,
)
from server.repositories.schemas.models import (
    Adsorbate,
    Adsorbent,
    Dataset,
    Isotherm,
    IsothermComponent,
)
from server.repositories.schemas.public_data import (
    CONFORMER_PROPERTY,
    AdsorbateSynonym,
    ChemicalProperty,
    DataSource,
    SourceRecord,
    Structure,
)
from server.repositories.schemas.types import normalize_identity

PUBCHEM_HIDDEN_PROPERTIES = {
    "preferred_name",
    "connectivity_smiles",
    "pubchem_3d_available",
}
PUBCHEM_COMPOUND_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"


###############################################################################
class EntityInUseError(RuntimeError):
    """Raised when deleting a material or species that experiments reference."""


###############################################################################
class MaterialRepository:

    # -------------------------------------------------------------------------
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    # -------------------------------------------------------------------------
    def _upsert_materials(
        self, model: Any, records: Iterable[dict[str, Any]], identity: str
    ) -> int:
        records = list(records)
        table = model.__table__
        with self.database.transaction() as session:
            self._reconcile_keys(session, model, records, identity)
            rows = [
                {key: value for key, value in row.items() if key in table.c}
                | {
                    "normalized_name": normalize_identity(
                        str(row.get("name", row["key"]))
                    )
                }
                for row in records
            ]
            return upsert_records(
                session, table, rows, ["key"], keep_existing_on_null=True
            )

    # -------------------------------------------------------------------------
    @staticmethod
    def _reconcile_keys(
        session: Session, model: Any, records: list[dict[str, Any]], identity: str
    ) -> None:
        """Point records at the row already owning their unique external identity.

        The identity column is unique besides ``key``; a row stored under another
        key (or repeated in the batch) would otherwise violate that constraint.
        Keys are rewritten in place so callers can still resolve ids by record key.
        """
        column = getattr(model, identity)
        values = {record[identity] for record in records if record.get(identity)}
        owners: dict[str, str] = {}
        if values:
            owners = {
                str(value): str(key)
                for value, key in session.execute(
                    select(column, model.key).where(column.in_(list(values)))
                )
            }
        for record in records:
            value = record.get(identity)
            if value:
                record["key"] = owners.setdefault(str(value), record["key"])

    # -------------------------------------------------------------------------
    def upsert_adsorbates(self, records: Iterable[dict[str, Any]]) -> int:
        return self._upsert_materials(Adsorbate, records, "inchi_key")

    # -------------------------------------------------------------------------
    def upsert_adsorbents(self, records: Iterable[dict[str, Any]]) -> int:
        return self._upsert_materials(Adsorbent, records, "external_identifier")

    # -------------------------------------------------------------------------
    def adsorbate_ids(self, keys: Iterable[str]) -> dict[str, int]:
        with self.database.session_factory() as session:
            rows = session.execute(
                select(Adsorbate.key, Adsorbate.id).where(Adsorbate.key.in_(list(keys)))
            ).all()
            return {key: identifier for key, identifier in rows}

    # -------------------------------------------------------------------------
    def adsorbent_ids(self, keys: Iterable[str]) -> dict[str, int]:
        with self.database.session_factory() as session:
            rows = session.execute(
                select(Adsorbent.key, Adsorbent.id).where(Adsorbent.key.in_(list(keys)))
            ).all()
            return {key: identifier for key, identifier in rows}

    # -------------------------------------------------------------------------
    def list_materials(
        self,
        *,
        page: int,
        page_size: int,
        query_text: str | None = None,
        formula: str | None = None,
        source: str | None = None,
        has_structure: bool | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        with self.database.session_factory() as session:
            query = select(Adsorbent)
            if query_text:
                query = query.where(Adsorbent.name.ilike(f"%{query_text.strip()}%"))
            if formula:
                query = query.where(Adsorbent.formula.ilike(f"%{formula.strip()}%"))
            if source:
                ids = public_adsorbent_ids(source).subquery()
                query = query.where(Adsorbent.id.in_(select(ids.c.entity_id)))
            if has_structure is not None:
                structured = select(Structure.adsorbent_id).where(
                    Structure.adsorbent_id.is_not(None)
                )
                query = query.where(
                    Adsorbent.id.in_(structured)
                    if has_structure
                    else Adsorbent.id.not_in(structured)
                )
            total = int(
                session.scalar(select(func.count()).select_from(query.subquery())) or 0
            )
            rows = session.scalars(
                query.order_by(Adsorbent.name, Adsorbent.id)
                .offset((page - 1) * page_size)
                .limit(page_size)
            ).all()
            return self._material_views(session, list(rows)), total

    # -------------------------------------------------------------------------
    @staticmethod
    def _material_views(
        session: Session, items: list[Adsorbent]
    ) -> list[dict[str, Any]]:
        row_ids = [item.id for item in items]
        structure_counts = counts_by(
            session, Structure.adsorbent_id, Structure.id, row_ids
        )
        isotherm_counts = counts_by(session, Isotherm.adsorbent_id, Isotherm.id, row_ids)
        provenance = provenance_entries(session, "adsorbent", row_ids)
        return [
            {
                "id": item.id,
                "name": item.name,
                "formula": item.formula,
                "molar_mass_g_mol": item.molar_mass_g_mol,
                "smiles": item.smiles,
                "structure_count": structure_counts.get(item.id, 0),
                "isotherm_count": isotherm_counts.get(item.id, 0),
                "external_identifiers": [
                    identifier for _, identifier in provenance.get(item.id, [])
                ],
            }
            for item in items
        ]

    # -------------------------------------------------------------------------
    def get_material(self, adsorbent_id: int) -> dict[str, Any]:
        with self.database.session_factory() as session:
            item = session.get(Adsorbent, adsorbent_id)
            if item is None:
                raise LookupError(f"Material {adsorbent_id} was not found.")
            view = self._material_views(session, [item])[0]
            view["datasets"] = self._dataset_usage(
                session,
                select(Isotherm.dataset_id, Isotherm.id).where(
                    Isotherm.adsorbent_id == adsorbent_id
                ),
            )
            return view

    # -------------------------------------------------------------------------
    def list_species(
        self,
        *,
        page: int,
        page_size: int,
        query_text: str | None = None,
        formula: str | None = None,
        source: str | None = None,
        molecular_weight_min: float | None = None,
        molecular_weight_max: float | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        with self.database.session_factory() as session:
            query = select(Adsorbate)
            if query_text:
                normalized = normalize_identity(query_text)
                synonym_ids = select(AdsorbateSynonym.adsorbate_id).where(
                    AdsorbateSynonym.normalized_synonym.ilike(f"%{normalized}%")
                )
                query = query.where(
                    or_(
                        Adsorbate.name.ilike(f"%{query_text.strip()}%"),
                        Adsorbate.id.in_(synonym_ids),
                    )
                )
            if formula:
                query = query.where(Adsorbate.formula.ilike(f"%{formula.strip()}%"))
            if molecular_weight_min is not None:
                query = query.where(Adsorbate.molar_mass_g_mol >= molecular_weight_min)
            if molecular_weight_max is not None:
                query = query.where(Adsorbate.molar_mass_g_mol <= molecular_weight_max)
            if source:
                ids = public_adsorbate_ids(source).subquery()
                query = query.where(Adsorbate.id.in_(select(ids.c.entity_id)))
            total = int(
                session.scalar(select(func.count()).select_from(query.subquery())) or 0
            )
            rows = session.scalars(
                query.order_by(Adsorbate.name, Adsorbate.id)
                .offset((page - 1) * page_size)
                .limit(page_size)
            ).all()
            return self._species_views(session, list(rows)), total

    # -------------------------------------------------------------------------
    def get_species(self, adsorbate_id: int) -> dict[str, Any]:
        with self.database.session_factory() as session:
            item = session.get(Adsorbate, adsorbate_id)
            if item is None:
                raise LookupError(f"Species {adsorbate_id} was not found.")
            view = self._species_views(session, [item])[0]
            view["datasets"] = self._dataset_usage(
                session,
                select(Isotherm.dataset_id, Isotherm.id)
                .join(IsothermComponent, IsothermComponent.isotherm_id == Isotherm.id)
                .where(IsothermComponent.adsorbate_id == adsorbate_id),
            )
            return view

    # -------------------------------------------------------------------------
    @staticmethod
    def _dataset_usage(session: Session, experiments: Any) -> list[dict[str, Any]]:
        usage = experiments.subquery()
        rows = session.execute(
            select(Dataset.id, Dataset.name, func.count(func.distinct(usage.c.id)))
            .join(usage, usage.c.dataset_id == Dataset.id)
            .group_by(Dataset.id, Dataset.name)
            .order_by(Dataset.name)
        ).all()
        return [
            {"id": int(dataset_id), "name": name, "experiment_count": int(count)}
            for dataset_id, name, count in rows
        ]

    # -------------------------------------------------------------------------
    def _species_views(
        self, session: Session, items: list[Adsorbate]
    ) -> list[dict[str, Any]]:
        if not items:
            return []
        item_ids = [item.id for item in items]
        provenance = provenance_entries(session, "adsorbate", item_ids)
        properties_by_adsorbate: dict[int, list[tuple[Any, str]]] = {}
        cached_conformers: set[int] = set()
        # The cached 3D conformer is a large text value: its row marks the cache, its content is not loaded.
        value_text = case(
            (ChemicalProperty.key == CONFORMER_PROPERTY, None),
            else_=ChemicalProperty.value_text,
        ).label("value_text")
        for row in session.execute(
            select(
                ChemicalProperty.adsorbate_id,
                ChemicalProperty.key,
                ChemicalProperty.value_number,
                value_text,
                ChemicalProperty.unit,
                DataSource.key.label("source_key"),
            )
            .join(SourceRecord, SourceRecord.id == ChemicalProperty.source_record_id)
            .join(DataSource, DataSource.id == SourceRecord.source_id)
            .where(ChemicalProperty.adsorbate_id.in_(item_ids))
            .order_by(
                ChemicalProperty.adsorbate_id, ChemicalProperty.key, ChemicalProperty.id
            )
        ):
            if row.key == CONFORMER_PROPERTY:
                cached_conformers.add(int(row.adsorbate_id))
                continue
            properties_by_adsorbate.setdefault(int(row.adsorbate_id), []).append(
                (row, str(row.source_key))
            )
        ranked_synonyms = (
            select(
                AdsorbateSynonym.adsorbate_id.label("adsorbate_id"),
                AdsorbateSynonym.synonym.label("synonym"),
                func.row_number()
                .over(
                    partition_by=AdsorbateSynonym.adsorbate_id,
                    order_by=AdsorbateSynonym.synonym,
                )
                .label("row_number"),
            )
            .where(AdsorbateSynonym.adsorbate_id.in_(item_ids))
            .subquery()
        )
        synonyms_by_adsorbate: dict[int, list[str]] = {}
        for adsorbate_id, synonym in session.execute(
            select(ranked_synonyms.c.adsorbate_id, ranked_synonyms.c.synonym)
            .where(ranked_synonyms.c.row_number <= 100)
            .order_by(ranked_synonyms.c.adsorbate_id, ranked_synonyms.c.synonym)
        ):
            synonyms_by_adsorbate.setdefault(int(adsorbate_id), []).append(str(synonym))
        isotherm_counts = counts_by(
            session,
            IsothermComponent.adsorbate_id,
            IsothermComponent.isotherm_id,
            item_ids,
        )
        return [
            {
                **self._species_payload(
                    item,
                    [identifier for _, identifier in provenance.get(item.id, [])],
                    properties_by_adsorbate.get(item.id, []),
                    synonyms_by_adsorbate.get(item.id, []),
                ),
                "isotherm_count": isotherm_counts.get(item.id, 0),
                "conformer_3d_cached": item.id in cached_conformers,
            }
            for item in items
        ]

    # -------------------------------------------------------------------------
    @staticmethod
    def _species_payload(
        item: Adsorbate,
        identifiers: list[dict[str, Any]],
        properties: list[tuple[Any, str]],
        synonyms: list[str],
    ) -> dict[str, Any]:
        property_map = {prop.key: prop for prop, _ in properties}
        pubchem = next(
            (identifier for identifier in identifiers if identifier["source"] == "pubchem"),
            None,
        )
        cid = pubchem["external_id"] if pubchem else None
        preferred = property_map.get("preferred_name")
        connectivity = property_map.get("connectivity_smiles")
        has_3d = property_map.get("pubchem_3d_available") is not None
        return {
            "id": item.id,
            "name": item.name,
            "preferred_name": preferred.value_text if preferred else None,
            "formula": item.formula,
            "molecular_weight": item.molar_mass_g_mol,
            "inchi": item.inchi,
            "inchi_key": item.inchi_key,
            "connectivity_smiles": connectivity.value_text if connectivity else None,
            "smiles": item.smiles,
            "pubchem_cid": cid,
            "synonyms": synonyms,
            "properties": [
                {
                    "key": prop.key,
                    "value_number": prop.value_number,
                    "value_text": prop.value_text,
                    "unit": prop.unit,
                    "source": source_key,
                }
                for prop, source_key in properties
                if prop.key not in PUBCHEM_HIDDEN_PROPERTIES
            ],
            "external_identifiers": identifiers,
            "structure_2d_url": (
                f"{PUBCHEM_COMPOUND_URL}{cid}/PNG?image_size=300x300" if cid else None
            ),
            "conformer_3d_url": (
                f"{PUBCHEM_COMPOUND_URL}{cid}/record/SDF?record_type=3d"
                if cid and has_3d
                else None
            ),
            "retrieved_at": max(
                (
                    identifier["retrieved_at"]
                    for identifier in identifiers
                    if identifier["retrieved_at"]
                ),
                default=None,
            ),
        }

    # -------------------------------------------------------------------------
    def delete_material(self, adsorbent_id: int) -> None:
        with self.database.transaction() as session:
            if session.get(Adsorbent, adsorbent_id) is None:
                raise LookupError(f"Material {adsorbent_id} was not found.")
            used = int(
                session.scalar(
                    select(func.count(Isotherm.id)).where(
                        Isotherm.adsorbent_id == adsorbent_id
                    )
                )
                or 0
            )
            if used:
                raise EntityInUseError(
                    f"This material is used by {used} experiment(s). Delete those "
                    "datasets or assign another material to the experiments first."
                )
            session.execute(delete(Adsorbent).where(Adsorbent.id == adsorbent_id))

    # -------------------------------------------------------------------------
    def delete_species(self, adsorbate_id: int) -> None:
        with self.database.transaction() as session:
            if session.get(Adsorbate, adsorbate_id) is None:
                raise LookupError(f"Species {adsorbate_id} was not found.")
            used = int(
                session.scalar(
                    select(
                        func.count(func.distinct(IsothermComponent.isotherm_id))
                    ).where(IsothermComponent.adsorbate_id == adsorbate_id)
                )
                or 0
            )
            if used:
                raise EntityInUseError(
                    f"This species is used by {used} experiment(s). Delete those "
                    "datasets or assign another species to the experiments first."
                )
            session.execute(delete(Adsorbate).where(Adsorbate.id == adsorbate_id))
