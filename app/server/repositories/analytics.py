# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from server.repositories.database.manager import DatabaseManager
from server.repositories.schemas.models import (
    Adsorbate,
    Adsorbent,
    Dataset,
    Isotherm,
    IsothermComponent,
    Observation,
)

FACET_LIMIT = 50
TOP_ENTITY_LIMIT = 10

###############################################################################
class AnalyticsRepository:
    """Read-only cross-dataset queries over canonical isotherm records."""

    # -------------------------------------------------------------------------
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    # -------------------------------------------------------------------------
    @staticmethod
    def _matching_isotherms(
        filters: dict[str, Any], *, exclude: str | None = None
    ) -> Select[Any]:
        """Isotherm ids matching the filter, optionally ignoring one facet."""
        query = select(Isotherm.id).join(Dataset, Dataset.id == Isotherm.dataset_id)
        if filters.get("dataset_sources") and exclude != "sources":
            query = query.where(Dataset.source.in_(filters["dataset_sources"]))
        if filters.get("dataset_ids") and exclude != "datasets":
            query = query.where(Isotherm.dataset_id.in_(filters["dataset_ids"]))
        if filters.get("adsorbent_ids") and exclude != "materials":
            query = query.where(Isotherm.adsorbent_id.in_(filters["adsorbent_ids"]))
        if filters.get("isotherm_ids"):
            query = query.where(Isotherm.id.in_(filters["isotherm_ids"]))
        if filters.get("adsorbate_ids") and exclude != "adsorbates":
            query = query.where(
                Isotherm.id.in_(
                    select(IsothermComponent.isotherm_id).where(
                        IsothermComponent.adsorbate_id.in_(filters["adsorbate_ids"])
                    )
                )
            )
        if filters.get("temperature_min_k") is not None:
            query = query.where(Isotherm.temperature_k >= filters["temperature_min_k"])
        if filters.get("temperature_max_k") is not None:
            query = query.where(Isotherm.temperature_k <= filters["temperature_max_k"])
        if filters.get("pressure_basis") and exclude != "bases":
            query = query.where(Isotherm.pressure_basis == filters["pressure_basis"])
        if not filters.get("include_mixtures"):
            query = query.where(
                Isotherm.id.in_(
                    select(IsothermComponent.isotherm_id)
                    .group_by(IsothermComponent.isotherm_id)
                    .having(func.count(IsothermComponent.id) == 1)
                )
            )
        return query

    # -------------------------------------------------------------------------
    def selection_page(
        self, filters: dict[str, Any], *, page: int, page_size: int
    ) -> tuple[list[dict[str, Any]], int]:
        matching = self._matching_isotherms(filters).subquery()
        with self.database.session_factory() as session:
            total = int(session.scalar(select(func.count()).select_from(matching)) or 0)
            rows = session.execute(
                select(
                    Isotherm.id,
                    Isotherm.name,
                    Isotherm.dataset_id,
                    Dataset.name.label("dataset"),
                    Dataset.source.label("dataset_source"),
                    Isotherm.adsorbent_id.label("material_id"),
                    Adsorbent.name.label("material"),
                    Isotherm.temperature_k,
                    Isotherm.pressure_basis,
                )
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .join(Adsorbent, Adsorbent.id == Isotherm.adsorbent_id)
                .where(Isotherm.id.in_(select(matching.c.id)))
                .order_by(Adsorbent.name, Isotherm.temperature_k, Isotherm.id)
                .offset((page - 1) * page_size)
                .limit(page_size)
            ).all()
            isotherm_ids = [int(row.id) for row in rows]
            adsorbates = self._adsorbates_by_isotherm(session, isotherm_ids)
            point_counts = dict(
                session.execute(
                    select(Observation.isotherm_id, func.count(Observation.id))
                    .where(Observation.isotherm_id.in_(isotherm_ids))
                    .group_by(Observation.isotherm_id)
                ).all()
            )
        items = [
            {
                **row._asdict(),
                "adsorbates": adsorbates.get(int(row.id), []),
                "point_count": int(point_counts.get(row.id, 0)),
            }
            for row in rows
        ]
        return items, total

    # -------------------------------------------------------------------------
    @staticmethod
    def _adsorbates_by_isotherm(
        session: Session, isotherm_ids: list[int]
    ) -> dict[int, list[dict[str, Any]]]:
        grouped: dict[int, list[dict[str, Any]]] = {}
        if not isotherm_ids:
            return grouped
        for isotherm_id, adsorbate_id, name in session.execute(
            select(IsothermComponent.isotherm_id, Adsorbate.id, Adsorbate.name)
            .join(Adsorbate, Adsorbate.id == IsothermComponent.adsorbate_id)
            .where(IsothermComponent.isotherm_id.in_(isotherm_ids))
            .order_by(IsothermComponent.isotherm_id, IsothermComponent.position)
        ):
            grouped.setdefault(int(isotherm_id), []).append(
                {"id": int(adsorbate_id), "name": str(name)}
            )
        return grouped

    # -------------------------------------------------------------------------
    def facets(
        self,
        filters: dict[str, Any],
        *,
        material_search: str | None = None,
        adsorbate_search: str | None = None,
    ) -> dict[str, Any]:
        """Facet counts; each facet ignores its own selection so siblings stay visible."""

        def scope(exclude: str) -> Any:
            return select(self._matching_isotherms(filters, exclude=exclude).subquery().c.id)

        materials = (
            select(Adsorbent.id, Adsorbent.name, func.count(Isotherm.id).label("count"))
            .join(Isotherm, Isotherm.adsorbent_id == Adsorbent.id)
            .where(Isotherm.id.in_(scope("materials")))
            .group_by(Adsorbent.id, Adsorbent.name)
        )
        if material_search and material_search.strip():
            materials = materials.where(Adsorbent.name.ilike(f"%{material_search.strip()}%"))
        adsorbates = (
            select(
                Adsorbate.id,
                Adsorbate.name,
                func.count(func.distinct(IsothermComponent.isotherm_id)).label("count"),
            )
            .join(IsothermComponent, IsothermComponent.adsorbate_id == Adsorbate.id)
            .where(IsothermComponent.isotherm_id.in_(scope("adsorbates")))
            .group_by(Adsorbate.id, Adsorbate.name)
        )
        if adsorbate_search and adsorbate_search.strip():
            adsorbates = adsorbates.where(Adsorbate.name.ilike(f"%{adsorbate_search.strip()}%"))
        datasets = (
            select(Dataset.id, Dataset.name, func.count(Isotherm.id).label("count"))
            .join(Isotherm, Isotherm.dataset_id == Dataset.id)
            .where(Isotherm.id.in_(scope("datasets")))
            .group_by(Dataset.id, Dataset.name)
        )
        with self.database.session_factory() as session:
            # Selected entries sort first so the client can always label them.
            facet_rows = {
                key: [
                    {"id": int(row.id), "name": str(row.name), "count": int(row.count)}
                    for row in session.execute(
                        query.order_by(
                            query.selected_columns[0].in_(filters.get(selected) or [-1]).desc(),
                            func.count().desc(),
                            query.selected_columns[1],
                        ).limit(FACET_LIMIT)
                    )
                ]
                for key, selected, query in (
                    ("materials", "adsorbent_ids", materials),
                    ("adsorbates", "adsorbate_ids", adsorbates),
                    ("datasets", "dataset_ids", datasets),
                )
            }
            sources = dict(
                session.execute(
                    select(Dataset.source, func.count(Isotherm.id))
                    .join(Isotherm, Isotherm.dataset_id == Dataset.id)
                    .where(Isotherm.id.in_(scope("sources")))
                    .group_by(Dataset.source)
                ).all()
            )
            bases = dict(
                session.execute(
                    select(Isotherm.pressure_basis, func.count(Isotherm.id))
                    .where(Isotherm.id.in_(scope("bases")))
                    .group_by(Isotherm.pressure_basis)
                ).all()
            )
            temperature = session.execute(
                select(func.min(Isotherm.temperature_k), func.max(Isotherm.temperature_k))
                .where(Isotherm.id.in_(scope("none")))
            ).one()
        return {
            **facet_rows,
            "sources": {str(key): int(value) for key, value in sources.items()},
            "pressure_bases": {str(key): int(value) for key, value in bases.items()},
            "temperature_min_k": temperature[0],
            "temperature_max_k": temperature[1],
        }

    # -------------------------------------------------------------------------
    def series_metadata(self, filters: dict[str, Any], *, limit: int) -> tuple[list[dict[str, Any]], int]:
        """One row per (isotherm, component) with its point count, capped at ``limit``."""
        matching = select(self._matching_isotherms(filters).subquery().c.id)
        point_counts = (
            select(
                Observation.component_id.label("component_id"),
                func.count(Observation.id).label("point_count"),
            )
            .group_by(Observation.component_id)
            .subquery()
        )
        query = (
            select(
                Isotherm.id.label("isotherm_id"),
                IsothermComponent.id.label("component_id"),
                Isotherm.name.label("isotherm"),
                Isotherm.dataset_id,
                Dataset.name.label("dataset"),
                Dataset.source.label("dataset_source"),
                Isotherm.adsorbent_id.label("material_id"),
                Adsorbent.name.label("material"),
                Adsorbate.id.label("adsorbate_id"),
                Adsorbate.name.label("adsorbate"),
                Adsorbate.molar_mass_g_mol,
                IsothermComponent.mole_fraction,
                Isotherm.temperature_k,
                Isotherm.pressure_basis,
                func.coalesce(point_counts.c.point_count, 0).label("point_count"),
            )
            .join(Isotherm, Isotherm.id == IsothermComponent.isotherm_id)
            .join(Dataset, Dataset.id == Isotherm.dataset_id)
            .join(Adsorbent, Adsorbent.id == Isotherm.adsorbent_id)
            .join(Adsorbate, Adsorbate.id == IsothermComponent.adsorbate_id)
            .outerjoin(point_counts, point_counts.c.component_id == IsothermComponent.id)
            .where(IsothermComponent.isotherm_id.in_(matching))
        )
        if filters.get("adsorbate_ids"):
            query = query.where(IsothermComponent.adsorbate_id.in_(filters["adsorbate_ids"]))
        with self.database.session_factory() as session:
            total = int(session.scalar(select(func.count()).select_from(query.subquery())) or 0)
            rows = session.execute(
                query.order_by(
                    Adsorbent.name,
                    Adsorbate.name,
                    Isotherm.temperature_k,
                    Isotherm.id,
                    IsothermComponent.position,
                ).limit(limit)
            ).all()
        return [row._asdict() for row in rows], total

    # -------------------------------------------------------------------------
    def component_observations(
        self, component_ids: list[int]
    ) -> dict[int, list[dict[str, Any]]]:
        grouped: dict[int, list[dict[str, Any]]] = {}
        if not component_ids:
            return grouped
        with self.database.session_factory() as session:
            for row in session.execute(
                select(
                    Observation.component_id,
                    Observation.id.label("observation_id"),
                    Observation.pressure_canonical.label("pressure"),
                    Observation.uptake_mol_kg.label("uptake"),
                    Observation.uptake_stddev_mol_kg.label("uptake_stddev"),
                )
                .where(Observation.component_id.in_(component_ids))
                .order_by(Observation.component_id, Observation.sequence_index)
            ):
                grouped.setdefault(int(row.component_id), []).append(
                    {
                        "observation_id": int(row.observation_id),
                        "pressure": float(row.pressure),
                        "uptake": float(row.uptake),
                        "uptake_stddev": row.uptake_stddev,
                    }
                )
        return grouped

    # -------------------------------------------------------------------------
    def distribution_values(self, filters: dict[str, Any], variable: str) -> list[float]:
        """Raw values of one variable over the matching isotherms."""
        matching = select(self._matching_isotherms(filters).subquery().c.id)
        if variable == "temperature_k":
            query: Select[Any] = select(Isotherm.temperature_k).where(Isotherm.id.in_(matching))
        elif variable == "points_per_isotherm":
            query = (
                select(func.count(Observation.id))
                .where(Observation.isotherm_id.in_(matching))
                .group_by(Observation.isotherm_id)
            )
        else:
            column = (
                Observation.pressure_canonical
                if variable == "pressure"
                else Observation.uptake_mol_kg
            )
            query = select(column).where(Observation.isotherm_id.in_(matching))
            if filters.get("adsorbate_ids"):
                query = query.join(
                    IsothermComponent, IsothermComponent.id == Observation.component_id
                ).where(IsothermComponent.adsorbate_id.in_(filters["adsorbate_ids"]))
        with self.database.session_factory() as session:
            return [float(value) for value in session.scalars(query)]

    # -------------------------------------------------------------------------
    def overview(self) -> dict[str, Any]:
        with self.database.session_factory() as session:
            counts = session.execute(
                select(
                    select(func.count(Dataset.id)).scalar_subquery().label("datasets"),
                    select(func.count(Isotherm.id)).scalar_subquery().label("isotherms"),
                    select(func.count(Observation.id)).scalar_subquery().label("observations"),
                    select(func.count(func.distinct(Isotherm.adsorbent_id)))
                    .scalar_subquery()
                    .label("materials"),
                    select(func.count(func.distinct(IsothermComponent.adsorbate_id)))
                    .scalar_subquery()
                    .label("adsorbates"),
                    select(func.count())
                    .select_from(
                        select(IsothermComponent.isotherm_id)
                        .group_by(IsothermComponent.isotherm_id)
                        .having(func.count(IsothermComponent.id) > 1)
                        .subquery()
                    )
                    .scalar_subquery()
                    .label("mixture_isotherms"),
                )
            ).one()
            by_source = dict(
                session.execute(
                    select(Dataset.source, func.count(Isotherm.id))
                    .join(Isotherm, Isotherm.dataset_id == Dataset.id)
                    .group_by(Dataset.source)
                ).all()
            )
            isotherm_spans = {
                row.pressure_basis: row
                for row in session.execute(
                    select(
                        Isotherm.pressure_basis,
                        func.count(Isotherm.id).label("isotherms"),
                        func.min(Isotherm.temperature_k).label("temperature_min_k"),
                        func.max(Isotherm.temperature_k).label("temperature_max_k"),
                    ).group_by(Isotherm.pressure_basis)
                )
            }
            observation_spans = {
                row.pressure_basis: row
                for row in session.execute(
                    select(
                        Isotherm.pressure_basis,
                        func.count(Observation.id).label("observations"),
                        func.min(Observation.pressure_canonical).label("pressure_min"),
                        func.max(Observation.pressure_canonical).label("pressure_max"),
                        func.min(Observation.uptake_mol_kg).label("uptake_min_mol_kg"),
                        func.max(Observation.uptake_mol_kg).label("uptake_max_mol_kg"),
                    )
                    .join(Isotherm, Isotherm.id == Observation.isotherm_id)
                    .group_by(Isotherm.pressure_basis)
                )
            }
            top_materials = session.execute(
                select(Adsorbent.id, Adsorbent.name, func.count(Isotherm.id).label("count"))
                .join(Isotherm, Isotherm.adsorbent_id == Adsorbent.id)
                .group_by(Adsorbent.id, Adsorbent.name)
                .order_by(func.count(Isotherm.id).desc(), Adsorbent.name)
                .limit(TOP_ENTITY_LIMIT)
            ).all()
            top_adsorbates = session.execute(
                select(
                    Adsorbate.id,
                    Adsorbate.name,
                    func.count(func.distinct(IsothermComponent.isotherm_id)).label("count"),
                )
                .join(IsothermComponent, IsothermComponent.adsorbate_id == Adsorbate.id)
                .group_by(Adsorbate.id, Adsorbate.name)
                .order_by(
                    func.count(func.distinct(IsothermComponent.isotherm_id)).desc(),
                    Adsorbate.name,
                )
                .limit(TOP_ENTITY_LIMIT)
            ).all()
        spans = []
        for basis, isotherm_row in isotherm_spans.items():
            observation_row = observation_spans.get(basis)
            spans.append(
                {
                    "pressure_basis": basis,
                    "pressure_unit": "1" if basis == "relative" else "Pa",
                    "isotherms": int(isotherm_row.isotherms),
                    "temperature_min_k": isotherm_row.temperature_min_k,
                    "temperature_max_k": isotherm_row.temperature_max_k,
                    "observations": int(observation_row.observations) if observation_row else 0,
                    "pressure_min": observation_row.pressure_min if observation_row else None,
                    "pressure_max": observation_row.pressure_max if observation_row else None,
                    "uptake_min_mol_kg": (
                        observation_row.uptake_min_mol_kg if observation_row else None
                    ),
                    "uptake_max_mol_kg": (
                        observation_row.uptake_max_mol_kg if observation_row else None
                    ),
                }
            )
        return {
            **{key: int(value or 0) for key, value in counts._asdict().items()},
            "isotherms_by_source": {str(key): int(value) for key, value in by_source.items()},
            "pressure_bases": sorted(spans, key=lambda span: span["pressure_basis"]),
            "top_materials": [row._asdict() for row in top_materials],
            "top_adsorbates": [row._asdict() for row in top_adsorbates],
        }


__all__ = ["AnalyticsRepository"]
