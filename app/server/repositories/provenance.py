# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""Shared read helpers for provider provenance and batched entity counts."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select, union
from sqlalchemy.orm import Session

from server.repositories.schemas.models import Isotherm, IsothermComponent
from server.repositories.schemas.public_data import (
    AdsorbateSourceRecord,
    AdsorbentSourceRecord,
    DataSource,
    IsothermSourceRecord,
    Reference,
    SourceRecord,
    SourceRecordReference,
    StructureSourceRecord,
)

# `source` filter value selecting records backed by any provider.
ANY_PROVIDER = "any"

LINK_COLUMNS = {
    "adsorbate": AdsorbateSourceRecord.adsorbate_id,
    "adsorbent": AdsorbentSourceRecord.adsorbent_id,
    "isotherm": IsothermSourceRecord.isotherm_id,
    "structure": StructureSourceRecord.structure_id,
}
LINK_MODELS = {
    "adsorbate": AdsorbateSourceRecord,
    "adsorbent": AdsorbentSourceRecord,
    "isotherm": IsothermSourceRecord,
    "structure": StructureSourceRecord,
}


###############################################################################
def provenance_entries(
    session: Session, entity: str, entity_ids: list[int]
) -> dict[int, list[tuple[int, dict[str, Any]]]]:
    """Return provider records per entity, newest retrieval first."""
    if not entity_ids:
        return {}
    if entity not in LINK_MODELS:
        raise ValueError(f"Unsupported provenance entity: {entity}")
    link_model = LINK_MODELS[entity]
    id_column = LINK_COLUMNS[entity]
    rows = session.execute(
        select(
            id_column.label("entity_id"),
            SourceRecord.id.label("source_record_id"),
            DataSource.key.label("source_key"),
            SourceRecord.external_id,
            SourceRecord.source_url,
            SourceRecord.retrieved_at,
            SourceRecord.source_version,
        )
        .join(SourceRecord, SourceRecord.source_id == DataSource.id)
        .join(link_model, link_model.source_record_id == SourceRecord.id)
        .where(id_column.in_(entity_ids))
        .order_by(id_column, SourceRecord.retrieved_at.desc(), SourceRecord.id.desc())
    ).all()
    result: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for row in rows:
        result.setdefault(int(row.entity_id), []).append(
            (
                int(row.source_record_id),
                {
                    "source": row.source_key,
                    "external_id": row.external_id,
                    "source_url": row.source_url,
                    "retrieved_at": row.retrieved_at,
                    "source_version": row.source_version,
                },
            )
        )
    return result


###############################################################################
def external_identifiers(
    session: Session, entity: str, entity_id: int
) -> list[dict[str, Any]]:
    return [
        identifier
        for _, identifier in provenance_entries(session, entity, [entity_id]).get(
            entity_id, []
        )
    ]


###############################################################################
def references_for_records(
    session: Session, source_record_ids: list[int]
) -> dict[int, dict[str, str | None]]:
    """Return the first literature reference attached to each source record."""
    if not source_record_ids:
        return {}
    rows = session.execute(
        select(SourceRecordReference.source_record_id, Reference.doi, Reference.title)
        .join(Reference, SourceRecordReference.reference_id == Reference.id)
        .where(SourceRecordReference.source_record_id.in_(source_record_ids))
        .order_by(SourceRecordReference.source_record_id, Reference.id)
    ).all()
    result: dict[int, dict[str, str | None]] = {}
    for row in rows:
        result.setdefault(
            int(row.source_record_id), {"doi": row.doi, "title": row.title}
        )
    return result


###############################################################################
def counts_by(
    session: Session, group_column: Any, count_column: Any, ids: list[int]
) -> dict[int, int]:
    if not ids:
        return {}
    return {
        int(group_id): int(count)
        for group_id, count in session.execute(
            select(group_column, func.count(count_column))
            .where(group_column.in_(ids))
            .group_by(group_column)
        )
        if group_id is not None
    }


###############################################################################
def public_adsorbate_ids(source: str | None = None) -> Any:
    """Chemicals with a provider record, or used by a provider-backed isotherm."""
    linked = (
        select(AdsorbateSourceRecord.adsorbate_id.label("entity_id"))
        .join(SourceRecord, SourceRecord.id == AdsorbateSourceRecord.source_record_id)
        .join(DataSource, DataSource.id == SourceRecord.source_id)
    )
    measured = (
        select(IsothermComponent.adsorbate_id.label("entity_id"))
        .join(
            IsothermSourceRecord,
            IsothermSourceRecord.isotherm_id == IsothermComponent.isotherm_id,
        )
        .join(SourceRecord, SourceRecord.id == IsothermSourceRecord.source_record_id)
        .join(DataSource, DataSource.id == SourceRecord.source_id)
    )
    if source and source != ANY_PROVIDER:
        linked = linked.where(DataSource.key == source)
        measured = measured.where(DataSource.key == source)
    return union(linked, measured)


###############################################################################
def public_adsorbent_ids(source: str | None = None) -> Any:
    """Materials with a provider record, or measured in a provider-backed isotherm."""
    linked = (
        select(AdsorbentSourceRecord.adsorbent_id.label("entity_id"))
        .join(SourceRecord, SourceRecord.id == AdsorbentSourceRecord.source_record_id)
        .join(DataSource, DataSource.id == SourceRecord.source_id)
    )
    measured = (
        select(Isotherm.adsorbent_id.label("entity_id"))
        .join(IsothermSourceRecord, IsothermSourceRecord.isotherm_id == Isotherm.id)
        .join(SourceRecord, SourceRecord.id == IsothermSourceRecord.source_record_id)
        .join(DataSource, DataSource.id == SourceRecord.source_id)
    )
    if source and source != ANY_PROVIDER:
        linked = linked.where(DataSource.key == source)
        measured = measured.where(DataSource.key == source)
    return union(linked, measured)
