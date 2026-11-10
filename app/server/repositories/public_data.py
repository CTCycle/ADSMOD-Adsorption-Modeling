# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import String, cast, delete, func, literal, or_, select, update
from sqlalchemy.orm import Session

from server.repositories.database.manager import DatabaseManager
from server.common.identity import STATUS_RESOLVED, stable_material_key
from server.repositories.identity import (
    IdentityConflictError,
    add_alias,
    curated_adsorbate,
    raise_review,
)
from server.repositories.schemas.models import (
    Adsorbate,
    Adsorbent,
    Isotherm,
)
from server.repositories.schemas.public_data import (
    CONFORMER_PROPERTY,
    AdsorbateSourceRecord,
    AdsorbateSynonym,
    AdsorbentSourceRecord,
    ChemicalProperty,
    DataSource,
    IsothermSourceRecord,
    PublicDataSyncFailure,
    PublicDataSyncRun,
    Reference,
    SourceRecord,
    SourceRecordReference,
    Structure,
    StructureAtom,
    StructureSourceRecord,
)
from server.repositories.provenance import (
    provenance_entries,
    public_adsorbate_ids,
    public_adsorbent_ids,
    references_for_records,
)
from server.repositories.schemas.types import normalize_identity

ACTIVE_SYNC_STATUSES = ("queued", "running")
# Probes that read provider state without acquiring records; they are never
# listed as acquisitions or counted in acquisition statistics.
NON_ACQUISITION_OPERATIONS = ("health_check", "nist_index")
TERMINAL_SYNC_STATUSES = (
    "completed",
    "partially_completed",
    "failed",
    "cancelled",
    "interrupted",
)

###############################################################################
def normalized_sync_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Return the request identity used to detect duplicate active sync runs."""
    normalized = {
        key: value for key, value in parameters.items() if key != "trigger"
    }
    for key in ("cod_ids", "identifiers"):
        values = normalized.get(key)
        if isinstance(values, (list, tuple)):
            normalized[key] = sorted(str(item) for item in values)
    return normalized

###############################################################################
class PublicDataRepository:

    # -------------------------------------------------------------------------
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    # -------------------------------------------------------------------------
    def ensure_sources(self, definitions: list[dict[str, Any]]) -> None:
        with self.database.transaction() as session:
            for definition in definitions:
                source = session.scalar(
                    select(DataSource).where(DataSource.key == definition["key"])
                )
                if source is None:
                    session.add(DataSource(**definition))
                    continue
                for key, value in definition.items():
                    if key != "key":
                        setattr(source, key, value)

    # -------------------------------------------------------------------------
    @staticmethod
    def _source(session: Session, key: str) -> DataSource:
        source = session.scalar(select(DataSource).where(DataSource.key == key))
        if source is None:
            raise LookupError(f"Public data source {key!r} is not registered.")
        return source

    # -------------------------------------------------------------------------
    @staticmethod
    def _record(
        session: Session,
        *,
        source: DataSource,
        record_type: str,
        external_id: str,
        source_url: str | None = None,
        source_version: str | None = None,
        raw_metadata: dict[str, Any] | None = None,
    ) -> SourceRecord:
        record = session.scalar(
            select(SourceRecord).where(
                SourceRecord.source_id == source.id,
                SourceRecord.record_type == record_type,
                SourceRecord.external_id == external_id,
            )
        )
        if record is None:
            record = SourceRecord(
                source_id=source.id,
                record_type=record_type,
                external_id=external_id,
                source_url=source_url,
                source_version=source_version,
                retrieved_at=datetime.now(timezone.utc),
                raw_metadata=raw_metadata or {},
            )
            session.add(record)
            session.flush()
        else:
            record.source_url = source_url or record.source_url
            record.source_version = source_version or record.source_version
            record.retrieved_at = datetime.now(timezone.utc)
            if raw_metadata is not None:
                record.raw_metadata = raw_metadata
        return record

    # -------------------------------------------------------------------------
    def source_rows(self) -> list[dict[str, Any]]:
        with self.database.session_factory() as session:
            rows = session.execute(
                select(DataSource, func.count(SourceRecord.id))
                .outerjoin(SourceRecord, SourceRecord.source_id == DataSource.id)
                .group_by(DataSource.id)
                .order_by(DataSource.name)
            ).all()
            return [
                {
                    "key": source.key,
                    "name": source.name,
                    "description": source.description,
                    "homepage_url": source.homepage_url,
                    "license_name": source.license_name,
                    "license_url": source.license_url,
                    "terms_url": source.terms_url,
                    "capabilities": list(source.capabilities),
                    "record_count": int(count or 0),
                    "health_status": source.health_status,
                    "health_detail": source.health_detail,
                    "last_health_checked_at": source.last_health_checked_at,
                    "last_successful_sync_at": source.last_successful_sync_at,
                }
                for source, count in rows
            ]

    # -------------------------------------------------------------------------
    def overview_counts(self) -> dict[str, int]:
        with self.database.session_factory() as session:
            row = session.execute(
                select(
                    # Only provider-backed entities count as public data.
                    select(func.count(func.distinct(IsothermSourceRecord.isotherm_id)))
                    .scalar_subquery()
                    .label("adsorption_records"),
                    select(func.count())
                    .select_from(public_adsorbent_ids().subquery())
                    .scalar_subquery()
                    .label("materials"),
                    select(func.count())
                    .select_from(public_adsorbate_ids().subquery())
                    .scalar_subquery()
                    .label("chemicals"),
                    select(func.count(func.distinct(StructureSourceRecord.structure_id)))
                    .scalar_subquery()
                    .label("structures"),
                    select(func.count(Adsorbate.id))
                    .scalar_subquery()
                    .label("species_total"),
                    select(
                        func.count(func.distinct(AdsorbateSourceRecord.adsorbate_id))
                    )
                    .join(
                        SourceRecord,
                        SourceRecord.id == AdsorbateSourceRecord.source_record_id,
                    )
                    .join(DataSource, DataSource.id == SourceRecord.source_id)
                    .where(DataSource.key == "pubchem")
                    .scalar_subquery()
                    .label("species_enriched"),
                    select(func.count(Adsorbent.id))
                    .scalar_subquery()
                    .label("materials_total"),
                    select(func.count(func.distinct(Structure.adsorbent_id)))
                    .where(Structure.adsorbent_id.is_not(None))
                    .scalar_subquery()
                    .label("materials_with_structure"),
                    select(func.count(PublicDataSyncRun.id))
                    .where(
                        PublicDataSyncRun.status.in_(ACTIVE_SYNC_STATUSES),
                        PublicDataSyncRun.operation.not_in(
                            NON_ACQUISITION_OPERATIONS
                        ),
                    )
                    .scalar_subquery()
                    .label("active_sync_runs"),
                    select(func.count(PublicDataSyncRun.id))
                    .where(
                        PublicDataSyncRun.status.in_(
                            ("failed", "partially_completed")
                        ),
                        PublicDataSyncRun.operation.not_in(
                            NON_ACQUISITION_OPERATIONS
                        ),
                    )
                    .scalar_subquery()
                    .label("failed_sync_runs"),
                    select(func.max(DataSource.last_successful_sync_at))
                    .scalar_subquery()
                    .label("latest_successful_sync_at"),
                )
            ).one()
            return {
                "adsorption_records": int(row.adsorption_records or 0),
                "materials": int(row.materials or 0),
                "chemicals": int(row.chemicals or 0),
                "structures": int(row.structures or 0),
                "species_total": int(row.species_total or 0),
                "species_enriched": int(row.species_enriched or 0),
                "materials_total": int(row.materials_total or 0),
                "materials_with_structure": int(row.materials_with_structure or 0),
                "active_sync_runs": int(row.active_sync_runs or 0),
                "failed_sync_runs": int(row.failed_sync_runs or 0),
                "latest_successful_sync_at": row.latest_successful_sync_at,
            }

    # -------------------------------------------------------------------------
    def update_source_health(
        self, source_key: str, *, status: str, detail: str | None, checked_at: datetime
    ) -> None:
        with self.database.transaction() as session:
            source = self._source(session, source_key)
            source.health_status = status
            source.health_detail = detail[:2000] if detail else None
            source.last_health_checked_at = checked_at

    # -------------------------------------------------------------------------
    def create_sync_run(
        self,
        *,
        source_key: str,
        operation: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = normalized_sync_parameters(parameters)
        with self.database.transaction() as session:
            self._source(session, source_key)
            active_runs = session.scalars(
                select(PublicDataSyncRun).where(
                    PublicDataSyncRun.source_key == source_key,
                    PublicDataSyncRun.operation == operation,
                    PublicDataSyncRun.status.in_(ACTIVE_SYNC_STATUSES),
                )
            ).all()
            for active_run in active_runs:
                if (
                    normalized_sync_parameters(dict(active_run.parameters))
                    == normalized
                ):
                    return self._sync_run_payload(active_run)
            run = PublicDataSyncRun(
                id=str(uuid4()),
                source_key=source_key,
                operation=operation,
                parameters=parameters,
                status="queued",
            )
            session.add(run)
            session.flush()
            return self._sync_run_payload(run)

    # -------------------------------------------------------------------------
    def resume_sync_run(self, run_id: str) -> dict[str, Any]:
        with self.database.transaction() as session:
            run = session.get(PublicDataSyncRun, run_id)
            if run is None:
                raise LookupError(
                    f"Public-data sync run {run_id!r} was not found."
                )
            if run.status != "interrupted":
                raise ValueError("Only interrupted sync runs can be resumed.")
            normalized = normalized_sync_parameters(dict(run.parameters))
            active_runs = session.scalars(
                select(PublicDataSyncRun).where(
                    PublicDataSyncRun.source_key == run.source_key,
                    PublicDataSyncRun.operation == run.operation,
                    PublicDataSyncRun.status.in_(ACTIVE_SYNC_STATUSES),
                    PublicDataSyncRun.id != run_id,
                )
            ).all()
            if any(
                normalized_sync_parameters(dict(active_run.parameters))
                == normalized
                for active_run in active_runs
            ):
                raise ValueError(
                    "An identical acquisition is already queued or running."
                )
            now = datetime.now(timezone.utc)
            run.status = "queued"
            run.started_at = None
            run.finished_at = None
            run.cancel_requested = False
            run.error_detail = None
            run.updated_at = now
            session.flush()
            return self._sync_run_payload(run)

    # -------------------------------------------------------------------------
    @staticmethod
    def _sync_run_payload(run: PublicDataSyncRun) -> dict[str, Any]:
        return {
            "id": run.id,
            "source_key": run.source_key,
            "operation": run.operation,
            "parameters": dict(run.parameters),
            "status": run.status,
            "progress": float(run.progress),
            "processed_count": int(run.processed_count),
            "imported_count": int(run.imported_count),
            "skipped_count": int(run.skipped_count),
            "failed_count": int(run.failed_count),
            "checkpoint": dict(run.checkpoint),
            "cancel_requested": bool(run.cancel_requested),
            "error_detail": run.error_detail,
            "created_at": run.created_at,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
        }

    # -------------------------------------------------------------------------
    def claim_next_sync_run(self) -> dict[str, Any] | None:
        with self.database.transaction() as session:
            run = session.scalar(
                select(PublicDataSyncRun)
                .where(PublicDataSyncRun.status == "queued")
                .order_by(PublicDataSyncRun.created_at, PublicDataSyncRun.id)
                .limit(1)
            )
            if run is None:
                return None
            now = datetime.now(timezone.utc)
            result = session.execute(
                update(PublicDataSyncRun)
                .where(
                    PublicDataSyncRun.id == run.id,
                    PublicDataSyncRun.status == "queued",
                )
                .values(status="running", started_at=now, updated_at=now)
            )
            if result.rowcount != 1:
                return None
            session.refresh(run)
            return self._sync_run_payload(run)

    # -------------------------------------------------------------------------
    def get_sync_run(self, run_id: str) -> dict[str, Any]:
        with self.database.session_factory() as session:
            run = session.get(PublicDataSyncRun, run_id)
            if run is None:
                raise LookupError(f"Public-data sync run {run_id!r} was not found.")
            return self._sync_run_payload(run)

    # -------------------------------------------------------------------------
    def list_sync_runs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        source_key: str | None = None,
        status_filter: str | None = None,
        status_scope: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        with self.database.session_factory() as session:
            query = select(PublicDataSyncRun)
            if source_key:
                query = query.where(PublicDataSyncRun.source_key == source_key)
            if status_filter:
                query = query.where(PublicDataSyncRun.status == status_filter)
            if status_scope == "active":
                query = query.where(
                    PublicDataSyncRun.status.in_(ACTIVE_SYNC_STATUSES),
                    PublicDataSyncRun.operation.not_in(NON_ACQUISITION_OPERATIONS),
                )
            elif status_scope == "history":
                query = query.where(
                    PublicDataSyncRun.status.in_(TERMINAL_SYNC_STATUSES),
                    PublicDataSyncRun.operation.not_in(NON_ACQUISITION_OPERATIONS),
                )
            elif status_scope is not None:
                raise ValueError(f"Unsupported sync-run scope: {status_scope!r}")
            total = int(
                session.scalar(select(func.count()).select_from(query.subquery())) or 0
            )
            rows = session.scalars(
                query.order_by(PublicDataSyncRun.created_at.desc(), PublicDataSyncRun.id)
                .offset(offset)
                .limit(limit)
            ).all()
            return [self._sync_run_payload(run) for run in rows], total

    # -------------------------------------------------------------------------
    def latest_nist_availability(self) -> dict[str, dict[str, Any]]:
        """Return the newest remote record count each NIST collection reported."""
        with self.database.session_factory() as session:
            runs = session.scalars(
                select(PublicDataSyncRun)
                .where(
                    PublicDataSyncRun.source_key == "nist",
                    PublicDataSyncRun.operation.in_(("nist_index", "nist_category")),
                    PublicDataSyncRun.finished_at.is_not(None),
                )
                .order_by(PublicDataSyncRun.finished_at.desc())
                .limit(200)
            ).all()
        availability: dict[str, dict[str, Any]] = {}
        for run in runs:
            category = run.checkpoint.get("category") or run.parameters.get("category")
            count = run.checkpoint.get("available_count")
            if category in availability or count is None:
                continue
            availability[str(category)] = {
                "available_count": int(count),
                "indexed_at": run.finished_at,
            }
        return availability

    # -------------------------------------------------------------------------
    def update_sync_run(self, run_id: str, **values: Any) -> None:
        allowed = {
            "status",
            "progress",
            "processed_count",
            "imported_count",
            "skipped_count",
            "failed_count",
            "checkpoint",
            "cancel_requested",
            "error_detail",
            "started_at",
            "finished_at",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unsupported sync-run fields: {', '.join(sorted(unknown))}")
        values["updated_at"] = datetime.now(timezone.utc)
        with self.database.transaction() as session:
            result = session.execute(
                update(PublicDataSyncRun)
                .where(PublicDataSyncRun.id == run_id)
                .values(**values)
            )
            if result.rowcount != 1:
                raise LookupError(f"Public-data sync run {run_id!r} was not found.")

    # -------------------------------------------------------------------------
    def request_sync_cancel(self, run_id: str) -> dict[str, Any]:
        with self.database.transaction() as session:
            run = session.get(PublicDataSyncRun, run_id)
            if run is None:
                raise LookupError(f"Public-data sync run {run_id!r} was not found.")
            if run.status == "queued":
                run.status = "cancelled"
                run.finished_at = datetime.now(timezone.utc)
            elif run.status == "running":
                run.cancel_requested = True
            return self._sync_run_payload(run)

    # -------------------------------------------------------------------------
    def sync_cancel_requested(self, run_id: str) -> bool:
        with self.database.session_factory() as session:
            run = session.get(PublicDataSyncRun, run_id)
            return run is None or bool(run.cancel_requested)

    # -------------------------------------------------------------------------
    def mark_unfinished_sync_runs(self, *, resume: bool) -> int:
        now = datetime.now(timezone.utc)
        with self.database.transaction() as session:
            query = update(PublicDataSyncRun).where(
                PublicDataSyncRun.status.in_(("queued", "running"))
            )
            if resume:
                result = session.execute(
                    query.values(
                        status="queued",
                        started_at=None,
                        finished_at=None,
                        cancel_requested=False,
                        updated_at=now,
                    )
                )
            else:
                result = session.execute(
                    query.values(
                        status="interrupted",
                        error_detail="Application stopped before the run completed.",
                        finished_at=now,
                        updated_at=now,
                    )
                )
            return int(result.rowcount or 0)

    # -------------------------------------------------------------------------
    def add_sync_failure(
        self,
        *,
        sync_run_id: str,
        source_key: str,
        reason: str,
        external_id: str | None = None,
        retry_eligible_at: datetime | None = None,
    ) -> None:
        with self.database.transaction() as session:
            session.add(
                PublicDataSyncFailure(
                    sync_run_id=sync_run_id,
                    source_key=source_key,
                    external_id=external_id,
                    reason=reason[:2000],
                    retry_eligible_at=retry_eligible_at,
                )
            )

    # -------------------------------------------------------------------------
    def list_sync_failures(
        self, run_id: str, *, limit: int = 100, offset: int = 0
    ) -> tuple[list[dict[str, Any]], int]:
        with self.database.session_factory() as session:
            query = select(PublicDataSyncFailure).where(
                PublicDataSyncFailure.sync_run_id == run_id
            )
            total = int(
                session.scalar(select(func.count()).select_from(query.subquery())) or 0
            )
            rows = session.scalars(
                query.order_by(PublicDataSyncFailure.id).offset(offset).limit(limit)
            ).all()
            return (
                [
                    {
                        "id": row.id,
                        "source_key": row.source_key,
                        "external_id": row.external_id,
                        "reason": row.reason,
                        "retry_eligible_at": row.retry_eligible_at,
                        "created_at": row.created_at,
                    }
                    for row in rows
                ],
                total,
            )

    # -------------------------------------------------------------------------
    def mark_source_sync_success(self, source_key: str, when: datetime) -> None:
        with self.database.transaction() as session:
            self._source(session, source_key).last_successful_sync_at = when

    # -------------------------------------------------------------------------
    def link_adsorbate_record(
        self,
        *,
        source_key: str,
        adsorbate_id: int,
        external_id: str,
        source_url: str | None = None,
        raw_metadata: dict[str, Any] | None = None,
        synonyms: list[str] | None = None,
    ) -> None:
        with self.database.transaction() as session:
            source = self._source(session, source_key)
            record = self._record(
                session,
                source=source,
                record_type="chemical",
                external_id=external_id,
                source_url=source_url,
                raw_metadata=raw_metadata,
            )
            link = session.get(AdsorbateSourceRecord, record.id)
            if link is not None and link.adsorbate_id != adsorbate_id:
                raise ValueError(
                    f"{source_key}:{external_id} is already linked to another adsorbate."
                )
            if link is None:
                session.add(
                    AdsorbateSourceRecord(
                        source_record_id=record.id, adsorbate_id=adsorbate_id
                    )
                )
            for synonym in synonyms or []:
                add_alias(session, "adsorbate", adsorbate_id, synonym, source_key, record.id)

    # -------------------------------------------------------------------------
    def link_adsorbent_record(
        self,
        *,
        source_key: str,
        adsorbent_id: int,
        external_id: str,
        source_url: str | None = None,
        raw_metadata: dict[str, Any] | None = None,
        synonyms: list[str] | None = None,
    ) -> None:
        with self.database.transaction() as session:
            source = self._source(session, source_key)
            record = self._record(
                session,
                source=source,
                record_type="material",
                external_id=external_id,
                source_url=source_url,
                raw_metadata=raw_metadata,
            )
            link = session.get(AdsorbentSourceRecord, record.id)
            if link is not None and link.adsorbent_id != adsorbent_id:
                raise ValueError(
                    f"{source_key}:{external_id} is already linked to another material."
                )
            if link is None:
                session.add(
                    AdsorbentSourceRecord(
                        source_record_id=record.id, adsorbent_id=adsorbent_id
                    )
                )
            for synonym in synonyms or []:
                add_alias(session, "adsorbent", adsorbent_id, synonym, source_key, record.id)

    # -------------------------------------------------------------------------
    def link_isotherm_record(
        self,
        *,
        source_key: str,
        dataset_id: int,
        external_id: str,
        source_url: str | None = None,
        raw_metadata: dict[str, Any] | None = None,
        doi: str | None = None,
        session: Session | None = None,
    ) -> None:
        """Link a stored isotherm to its source record.

        Pass `session` to join the caller's transaction.
        """
        if session is not None:
            self._link_isotherm_record(
                session, source_key, dataset_id, external_id, source_url, raw_metadata, doi
            )
            return
        with self.database.transaction() as own_session:
            self._link_isotherm_record(
                own_session, source_key, dataset_id, external_id, source_url, raw_metadata, doi
            )

    # -------------------------------------------------------------------------
    def _link_isotherm_record(
        self,
        session: Session,
        source_key: str,
        dataset_id: int,
        external_id: str,
        source_url: str | None,
        raw_metadata: dict[str, Any] | None,
        doi: str | None,
    ) -> None:
        isotherm_id = session.scalar(
            select(Isotherm.id).where(
                Isotherm.dataset_id == dataset_id,
                Isotherm.external_key == external_id,
            )
        )
        if isotherm_id is None:
            raise LookupError(f"Isotherm {external_id!r} was not persisted.")
        source = self._source(session, source_key)
        record = self._record(
            session,
            source=source,
            record_type="adsorption",
            external_id=external_id,
            source_url=source_url,
            raw_metadata=raw_metadata,
        )
        link = session.get(IsothermSourceRecord, record.id)
        if link is not None and link.isotherm_id != isotherm_id:
            raise ValueError(
                f"{source_key}:{external_id} is already linked to another isotherm."
            )
        if link is None:
            session.add(
                IsothermSourceRecord(
                    source_record_id=record.id, isotherm_id=isotherm_id
                )
            )
        self._attach_reference(session, record, doi=doi)

    # -------------------------------------------------------------------------
    @staticmethod
    def _attach_reference(
        session: Session,
        record: SourceRecord,
        *,
        doi: object,
        title: str | None = None,
        journal: str | None = None,
        year: int | None = None,
    ) -> None:
        normalized_doi = str(doi or "").strip() or None
        if normalized_doi is None:
            return
        reference = session.scalar(
            select(Reference).where(
                func.lower(Reference.doi) == normalized_doi.lower()
            )
        )
        if reference is None:
            reference = Reference(
                doi=normalized_doi,
                title=title,
                journal=journal,
                year=year,
                url=f"https://doi.org/{normalized_doi}",
            )
            session.add(reference)
            session.flush()
        else:
            reference.title = reference.title or title
            reference.journal = reference.journal or journal
            reference.year = reference.year or year
        if session.get(SourceRecordReference, (record.id, reference.id)) is None:
            session.add(
                SourceRecordReference(
                    source_record_id=record.id, reference_id=reference.id
                )
            )

    # -------------------------------------------------------------------------
    def upsert_pubchem_compound(
        self, payload: dict[str, Any], adsorbate_id: int | None = None
    ) -> int:
        """Persist one PubChem compound, optionally onto an existing adsorbate.

        An explicit ``adsorbate_id`` attaches the CID to that chemical; a
        conflicting InChIKey or an existing CID link elsewhere is rejected.
        """
        cid = str(payload["cid"])
        inchi_key = str(payload.get("inchi_key") or "").strip() or None
        name = str(payload.get("name") or payload.get("preferred_name") or cid).strip()
        with self.database.transaction() as session:
            source = self._source(session, "pubchem")
            record = self._record(
                session,
                source=source,
                record_type="chemical",
                external_id=cid,
                source_url=payload.get("source_url"),
                raw_metadata=payload.get("raw_metadata") or {},
            )
            existing_link = session.get(AdsorbateSourceRecord, record.id)
            adsorbate = self._pubchem_target(
                session,
                cid=cid,
                inchi_key=inchi_key,
                existing_link=existing_link,
                adsorbate_id=adsorbate_id,
            )
            if adsorbate is None:
                adsorbate = Adsorbate(
                    key=stable_material_key("adsorbate", name, inchi_key or f"pubchem:{cid}"),
                    name=name,
                    inchi_key=inchi_key,
                )
                session.add(adsorbate)
                session.flush()
            elif inchi_key and adsorbate.inchi_key not in (None, inchi_key):
                raise ValueError(
                    "PubChem identity conflicts with the canonical adsorbate InChIKey; "
                    "automatic merging was refused."
                )

            adsorbate.identity_status = STATUS_RESOLVED
            # Enrichment keeps the canonical display name chosen by its source.
            adsorbate.inchi_key = inchi_key or adsorbate.inchi_key
            adsorbate.inchi = payload.get("inchi") or adsorbate.inchi
            adsorbate.formula = payload.get("formula") or adsorbate.formula
            adsorbate.molar_mass_g_mol = (
                payload.get("molecular_weight")
                if payload.get("molecular_weight") is not None
                else adsorbate.molar_mass_g_mol
            )
            adsorbate.smiles = payload.get("smiles") or adsorbate.smiles
            if existing_link is None:
                session.add(
                    AdsorbateSourceRecord(
                        source_record_id=record.id, adsorbate_id=adsorbate.id
                    )
                )

            session.execute(
                delete(AdsorbateSynonym).where(
                    AdsorbateSynonym.source_record_id == record.id
                )
            )
            seen: set[str] = set()
            for synonym in payload.get("synonyms") or []:
                value = str(synonym).strip()
                normalized = normalize_identity(value) if value else ""
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                session.add(
                    AdsorbateSynonym(
                        adsorbate_id=adsorbate.id,
                        source_record_id=record.id,
                        synonym=value,
                        normalized_synonym=normalized,
                    )
                )

            session.execute(
                delete(ChemicalProperty).where(
                    ChemicalProperty.source_record_id == record.id
                )
            )
            property_rows = self._pubchem_property_rows(payload)
            for key, value_number, value_text, unit in property_rows:
                session.add(
                    ChemicalProperty(
                        adsorbate_id=adsorbate.id,
                        source_record_id=record.id,
                        key=key,
                        value_number=value_number,
                        value_text=value_text,
                        unit=unit,
                    )
                )
            session.flush()
            return adsorbate.id

    # -------------------------------------------------------------------------
    @staticmethod
    def _pubchem_property_rows(
        payload: dict[str, Any],
    ) -> list[tuple[str, float | None, str | None, str | None]]:
        property_rows: list[tuple[str, float | None, str | None, str | None]] = []
        preferred_name = payload.get("preferred_name")
        if preferred_name:
            property_rows.append(("preferred_name", None, str(preferred_name), None))
        connectivity_smiles = payload.get("connectivity_smiles")
        if connectivity_smiles:
            property_rows.append(
                ("connectivity_smiles", None, str(connectivity_smiles), None)
            )
        if payload.get("conformer_3d_url"):
            property_rows.append(("pubchem_3d_available", None, "true", None))
        units = {
            "tpsa_angstrom2": "Å²",
            "exact_mass_da": "Da",
            "monoisotopic_mass_da": "Da",
        }
        for key, value in (payload.get("descriptors") or {}).items():
            property_rows.append((str(key), float(value), None, units.get(str(key))))
        for element, value in (payload.get("elemental_composition") or {}).items():
            property_rows.append((f"element:{element}", float(value), None, "atoms"))
        return property_rows

    # -------------------------------------------------------------------------
    @staticmethod
    def _pubchem_target(
        session: Session,
        *,
        cid: str,
        inchi_key: str | None,
        existing_link: AdsorbateSourceRecord | None,
        adsorbate_id: int | None,
    ) -> Adsorbate | None:
        if adsorbate_id is None:
            if existing_link is not None:
                return session.get(Adsorbate, existing_link.adsorbate_id)
            if inchi_key:
                return session.scalar(
                    select(Adsorbate).where(Adsorbate.inchi_key == inchi_key)
                )
            return None
        adsorbate = session.get(Adsorbate, adsorbate_id)
        if adsorbate is None:
            raise LookupError(f"Chemical {adsorbate_id} does not exist.")
        if existing_link is not None and existing_link.adsorbate_id != adsorbate_id:
            raise IdentityConflictError(
                f"PubChem CID {cid} is already linked to another chemical; "
                "a review was opened.",
                "adsorbate",
                adsorbate_id,
                int(existing_link.adsorbate_id),
                adsorbate.name,
            )
        owner_id = (
            session.scalar(
                select(Adsorbate.id).where(
                    Adsorbate.inchi_key == inchi_key, Adsorbate.id != adsorbate_id
                )
            )
            if inchi_key
            else None
        )
        if owner_id is not None:
            raise IdentityConflictError(
                f"PubChem CID {cid} matches another chemical's InChIKey; "
                "automatic merging was refused and a review was opened.",
                "adsorbate",
                adsorbate_id,
                int(owner_id),
                adsorbate.name,
            )
        return adsorbate

    # -------------------------------------------------------------------------
    def find_local_pubchem_adsorbate(self, query: str) -> int | None:
        value = query.strip()
        if not value:
            return None
        with self.database.session_factory() as session:
            if value.isdigit():
                return session.scalar(
                    select(AdsorbateSourceRecord.adsorbate_id)
                    .join(SourceRecord, SourceRecord.id == AdsorbateSourceRecord.source_record_id)
                    .join(DataSource, DataSource.id == SourceRecord.source_id)
                    .where(
                        DataSource.key == "pubchem",
                        SourceRecord.record_type == "chemical",
                        SourceRecord.external_id == value,
                    )
                )
            if len(value) == 27 and value[14] == "-" and value[25] == "-":
                return session.scalar(
                    select(Adsorbate.id).where(
                        Adsorbate.inchi_key == value.upper()
                    )
                )
            # A name is reused only when PubChem itself listed it as a synonym
            # of exactly one stored compound.
            matches = session.scalars(
                select(AdsorbateSynonym.adsorbate_id)
                .join(
                    SourceRecord,
                    SourceRecord.id == AdsorbateSynonym.source_record_id,
                )
                .join(DataSource, DataSource.id == SourceRecord.source_id)
                .where(
                    DataSource.key == "pubchem",
                    AdsorbateSynonym.normalized_synonym == normalize_identity(value),
                )
                .distinct()
                .limit(2)
            ).all()
            if len(matches) == 1:
                return int(matches[0])
            curated = curated_adsorbate(value)
            if curated is None:
                return None
            return session.scalar(
                select(Adsorbate.id).where(Adsorbate.inchi_key == curated.inchi_key)
            )

    # -------------------------------------------------------------------------
    def open_identity_review(self, **fields: Any) -> None:
        with self.database.transaction() as session:
            raise_review(session, **fields)

    # -------------------------------------------------------------------------
    def pubchem_enrichment_candidates(
        self, limit: int | None, fraction: float = 1.0
    ) -> list[dict[str, Any]]:
        """Return chemicals without PubChem identity that may still resolve.

        Chemicals whose earlier lookup failed permanently (no retry time) are
        excluded so repeated enrichment runs make progress. `fraction` selects
        the leading share of the eligible chemicals.
        """
        with self.database.session_factory() as session:
            linked = (
                select(AdsorbateSourceRecord.adsorbate_id)
                .join(SourceRecord, SourceRecord.id == AdsorbateSourceRecord.source_record_id)
                .join(DataSource, DataSource.id == SourceRecord.source_id)
                .where(DataSource.key == "pubchem")
            )
            unresolvable = select(PublicDataSyncFailure.external_id).where(
                PublicDataSyncFailure.source_key == "pubchem",
                PublicDataSyncFailure.retry_eligible_at.is_(None),
                PublicDataSyncFailure.external_id.like("adsorbate:%"),
            )
            rows = session.execute(
                select(
                    Adsorbate.id,
                    Adsorbate.name,
                    Adsorbate.inchi_key,
                    Adsorbate.formula,
                    Adsorbate.smiles,
                )
                .where(
                    Adsorbate.id.not_in(linked),
                    (literal("adsorbate:") + cast(Adsorbate.id, String)).not_in(
                        unresolvable
                    ),
                )
                .order_by(Adsorbate.id)
            ).all()
            selected = rows[: self._fraction_count(len(rows), fraction, limit)]
            return [
                {
                    "id": int(row.id),
                    "query": row.inchi_key or row.name,
                    "name": row.name,
                    "inchi_key": row.inchi_key,
                    "formula": row.formula,
                    "smiles": row.smiles,
                }
                for row in selected
            ]

    # -------------------------------------------------------------------------
    @staticmethod
    def _pubchem_record_for(adsorbate_id: Any) -> Any:
        return (
            select(SourceRecord.id, SourceRecord.external_id)
            .join(AdsorbateSourceRecord, AdsorbateSourceRecord.source_record_id == SourceRecord.id)
            .join(DataSource, DataSource.id == SourceRecord.source_id)
            .where(
                DataSource.key == "pubchem",
                SourceRecord.record_type == "chemical",
                AdsorbateSourceRecord.adsorbate_id == adsorbate_id,
            )
        )

    # -------------------------------------------------------------------------
    def conformer_state(self, adsorbate_id: int) -> dict[str, Any] | None:
        """PubChem CID, 3D availability and cached SDF; None without a PubChem link."""
        with self.database.session_factory() as session:
            record = session.execute(self._pubchem_record_for(adsorbate_id)).first()
            if record is None:
                return None
            values = dict(
                session.execute(
                    select(ChemicalProperty.key, ChemicalProperty.value_text).where(
                        ChemicalProperty.source_record_id == record.id,
                        ChemicalProperty.key.in_(("pubchem_3d_available", CONFORMER_PROPERTY)),
                    )
                ).all()
            )
        return {
            "cid": str(record.external_id),
            "available": "pubchem_3d_available" in values,
            "sdf": values.get(CONFORMER_PROPERTY),
        }

    # -------------------------------------------------------------------------
    def store_conformer(self, adsorbate_id: int, sdf: str) -> None:
        with self.database.transaction() as session:
            record = session.execute(self._pubchem_record_for(adsorbate_id)).first()
            if record is None:
                raise LookupError(f"Chemical {adsorbate_id} has no PubChem record.")
            for key, value in (("pubchem_3d_available", "true"), (CONFORMER_PROPERTY, sdf)):
                session.execute(
                    delete(ChemicalProperty).where(
                        ChemicalProperty.source_record_id == record.id,
                        ChemicalProperty.key == key,
                    )
                )
                session.add(
                    ChemicalProperty(
                        adsorbate_id=adsorbate_id,
                        source_record_id=record.id,
                        key=key,
                        value_text=value,
                    )
                )

    # -------------------------------------------------------------------------
    def conformer_candidates(self, limit: int | None, fraction: float = 1.0) -> list[dict[str, Any]]:
        """Enriched chemicals with a published 3D conformer that is not cached yet.

        Chemicals whose earlier download failed permanently are excluded so
        repeated runs make progress.
        """
        with self.database.session_factory() as session:
            cached = select(ChemicalProperty.source_record_id).where(
                ChemicalProperty.key == CONFORMER_PROPERTY
            )
            settled = select(PublicDataSyncFailure.external_id).where(
                PublicDataSyncFailure.source_key == "pubchem",
                PublicDataSyncFailure.retry_eligible_at.is_(None),
                PublicDataSyncFailure.external_id.like("conformer:%"),
            )
            rows = session.execute(
                select(AdsorbateSourceRecord.adsorbate_id)
                .join(ChemicalProperty, ChemicalProperty.source_record_id == AdsorbateSourceRecord.source_record_id)
                .where(
                    ChemicalProperty.key == "pubchem_3d_available",
                    AdsorbateSourceRecord.source_record_id.not_in(cached),
                    (literal("conformer:") + cast(AdsorbateSourceRecord.adsorbate_id, String)).not_in(settled),
                )
                .order_by(AdsorbateSourceRecord.adsorbate_id)
            ).all()
        selected = rows[: self._fraction_count(len(rows), fraction, limit)]
        return [{"id": int(row.adsorbate_id)} for row in selected]

    # -------------------------------------------------------------------------
    @staticmethod
    def _fraction_count(total: int, fraction: float, limit: int | None) -> int:
        share = min(1.0, max(0.0, float(fraction)))
        count = math.ceil(share * total)
        return count if limit is None else min(max(1, limit), count)

    # -------------------------------------------------------------------------
    def cod_bulk_candidates(
        self, limit: int | None, fraction: float = 1.0
    ) -> list[dict[str, Any]]:
        """Return stored materials that have no structure and no final COD outcome.

        Materials already judged unavailable or ambiguous (a stored failure
        without retry time) are excluded so repeated bulk runs make progress.
        """
        with self.database.session_factory() as session:
            linked = select(Structure.adsorbent_id).where(
                Structure.adsorbent_id.is_not(None)
            )
            settled = select(PublicDataSyncFailure.external_id).where(
                PublicDataSyncFailure.source_key == "cod",
                PublicDataSyncFailure.retry_eligible_at.is_(None),
                PublicDataSyncFailure.external_id.like("adsorbent:%"),
            )
            rows = session.execute(
                select(Adsorbent.id, Adsorbent.name)
                .where(
                    Adsorbent.id.not_in(linked),
                    (literal("adsorbent:") + cast(Adsorbent.id, String)).not_in(
                        settled
                    ),
                )
                .order_by(Adsorbent.id)
            ).all()
            selected = rows[: self._fraction_count(len(rows), fraction, limit)]
            return [{"id": int(row.id), "name": row.name} for row in selected]

    # -------------------------------------------------------------------------
    def upsert_cod_structure(
        self,
        *,
        metadata: dict[str, Any],
        cif_text: str,
        atoms: list[dict[str, Any]],
        adsorbent_id: int | None,
    ) -> int:
        cod_id = str(metadata["cod_id"])
        with self.database.transaction() as session:
            if adsorbent_id is not None and session.get(Adsorbent, adsorbent_id) is None:
                raise LookupError(f"Material {adsorbent_id} does not exist.")
            source = self._source(session, "cod")
            record = self._record(
                session,
                source=source,
                record_type="structure",
                external_id=cod_id,
                source_url=metadata.get("source_url"),
                source_version=metadata.get("source_version"),
                raw_metadata=metadata.get("raw_metadata") or {},
            )
            link = session.get(StructureSourceRecord, record.id)
            structure = session.get(Structure, link.structure_id) if link else None
            if (
                structure is not None
                and adsorbent_id is not None
                and structure.adsorbent_id not in (None, adsorbent_id)
            ):
                raise ValueError(
                    f"COD structure {cod_id} is already linked to another material."
                )
            if structure is None:
                structure = Structure(
                    format="cif",
                    content=cif_text,
                    content_sha256=hashlib.sha256(cif_text.encode("utf-8")).hexdigest(),
                )
                session.add(structure)
                session.flush()
                session.add(
                    StructureSourceRecord(
                        source_record_id=record.id, structure_id=structure.id
                    )
                )
            if adsorbent_id is not None:
                structure.adsorbent_id = adsorbent_id
            structure.name = metadata.get("name")
            structure.formula = metadata.get("formula")
            structure.content = cif_text
            structure.content_sha256 = hashlib.sha256(cif_text.encode("utf-8")).hexdigest()
            structure.space_group = metadata.get("space_group")
            structure.space_group_number = metadata.get("space_group_number")
            structure.cell_a_angstrom = metadata.get("cell_a_angstrom")
            structure.cell_b_angstrom = metadata.get("cell_b_angstrom")
            structure.cell_c_angstrom = metadata.get("cell_c_angstrom")
            structure.cell_alpha_deg = metadata.get("cell_alpha_deg")
            structure.cell_beta_deg = metadata.get("cell_beta_deg")
            structure.cell_gamma_deg = metadata.get("cell_gamma_deg")
            structure.cell_volume_angstrom3 = metadata.get("cell_volume_angstrom3")
            structure.has_coordinates = bool(atoms) or bool(metadata.get("has_coordinates"))

            session.execute(
                delete(StructureAtom).where(StructureAtom.structure_id == structure.id)
            )
            for atom in atoms:
                session.add(StructureAtom(structure_id=structure.id, **atom))

            self._attach_reference(
                session,
                record,
                doi=metadata.get("doi"),
                title=metadata.get("title"),
                journal=metadata.get("journal"),
                year=metadata.get("year"),
            )
            session.flush()
            return structure.id

    # -------------------------------------------------------------------------
    def find_cod_structure_id(self, cod_id: str) -> int | None:
        with self.database.session_factory() as session:
            return session.scalar(
                select(StructureSourceRecord.structure_id)
                .join(
                    SourceRecord,
                    SourceRecord.id == StructureSourceRecord.source_record_id,
                )
                .join(DataSource, DataSource.id == SourceRecord.source_id)
                .where(
                    DataSource.key == "cod",
                    SourceRecord.record_type == "structure",
                    SourceRecord.external_id == str(cod_id),
                )
            )

    # -------------------------------------------------------------------------
    def link_cod_structure_to_material(self, cod_id: str, adsorbent_id: int) -> int:
        with self.database.transaction() as session:
            if session.get(Adsorbent, adsorbent_id) is None:
                raise LookupError(f"Material {adsorbent_id} does not exist.")
            structure = session.scalar(
                select(Structure)
                .join(
                    StructureSourceRecord,
                    StructureSourceRecord.structure_id == Structure.id,
                )
                .join(
                    SourceRecord,
                    SourceRecord.id == StructureSourceRecord.source_record_id,
                )
                .join(DataSource, DataSource.id == SourceRecord.source_id)
                .where(
                    DataSource.key == "cod",
                    SourceRecord.record_type == "structure",
                    SourceRecord.external_id == str(cod_id),
                )
            )
            if structure is None:
                raise LookupError(f"COD structure {cod_id!r} was not imported.")
            if structure.adsorbent_id not in (None, adsorbent_id):
                raise ValueError(
                    f"COD structure {cod_id} is already linked to another material."
                )
            structure.adsorbent_id = adsorbent_id
            return structure.id

    # -------------------------------------------------------------------------
    def set_structure_material(
        self, structure_id: int, adsorbent_id: int | None
    ) -> None:
        """Explicitly link a structure to a material, or unlink it with None."""
        with self.database.transaction() as session:
            structure = session.get(Structure, structure_id)
            if structure is None:
                raise LookupError(f"Structure {structure_id} was not found.")
            if adsorbent_id is not None:
                if session.get(Adsorbent, adsorbent_id) is None:
                    raise LookupError(f"Material {adsorbent_id} does not exist.")
                if structure.adsorbent_id not in (None, adsorbent_id):
                    raise ValueError(
                        "The structure is linked to another material; unlink it first."
                    )
            structure.adsorbent_id = adsorbent_id

    # -------------------------------------------------------------------------
    def get_structure_content(self, structure_id: int) -> tuple[str, str]:
        with self.database.session_factory() as session:
            structure = session.get(Structure, structure_id)
            if structure is None:
                raise LookupError(f"Structure {structure_id} was not found.")
            return structure.format, structure.content

    # -------------------------------------------------------------------------
    def list_structures(
        self,
        *,
        page: int,
        page_size: int,
        query_text: str | None = None,
        source: str | None = None,
        linked_only: bool | None = None,
        material_id: int | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        with self.database.session_factory() as session:
            query = select(Structure)
            if material_id is not None:
                query = query.where(Structure.adsorbent_id == material_id)
            if query_text:
                pattern = f"%{query_text.strip()}%"
                query = query.where(
                    or_(Structure.name.ilike(pattern), Structure.formula.ilike(pattern))
                )
            if linked_only is not None:
                query = query.where(
                    Structure.adsorbent_id.is_not(None)
                    if linked_only
                    else Structure.adsorbent_id.is_(None)
                )
            if source:
                ids = (
                    select(StructureSourceRecord.structure_id)
                    .join(
                        SourceRecord,
                        SourceRecord.id == StructureSourceRecord.source_record_id,
                    )
                    .join(DataSource, DataSource.id == SourceRecord.source_id)
                    .where(DataSource.key == source)
                )
                query = query.where(Structure.id.in_(ids))
            total = int(
                session.scalar(select(func.count()).select_from(query.subquery())) or 0
            )
            rows = session.scalars(
                query.order_by(Structure.updated_at.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            ).all()
            return self._structure_views(session, rows, include_atoms=False), total

    # -------------------------------------------------------------------------
    def get_structure(self, structure_id: int) -> dict[str, Any]:
        with self.database.session_factory() as session:
            item = session.get(Structure, structure_id)
            if item is None:
                raise LookupError(f"Structure {structure_id} was not found.")
            return self._structure_view(session, item, include_atoms=True)

    # -------------------------------------------------------------------------
    def _structure_views(
        self,
        session: Session,
        items: list[Structure],
        *,
        include_atoms: bool,
    ) -> list[dict[str, Any]]:
        if not items:
            return []
        item_ids = [item.id for item in items]
        provenance = provenance_entries(session, "structure", item_ids)
        material_ids = [item.adsorbent_id for item in items if item.adsorbent_id is not None]
        material_names = {
            int(material_id): str(material_name)
            for material_id, material_name in session.execute(
                select(Adsorbent.id, Adsorbent.name).where(Adsorbent.id.in_(material_ids))
            )
        }
        atom_counts = {
            int(structure_id): int(count)
            for structure_id, count in session.execute(
                select(StructureAtom.structure_id, func.count(StructureAtom.id))
                .where(StructureAtom.structure_id.in_(item_ids))
                .group_by(StructureAtom.structure_id)
            )
        }
        atoms_by_structure: dict[int, list[dict[str, Any]]] = {}
        if include_atoms:
            atoms = session.scalars(
                select(StructureAtom)
                .where(StructureAtom.structure_id.in_(item_ids))
                .order_by(StructureAtom.structure_id, StructureAtom.sequence_index)
            ).all()
            for atom in atoms:
                atoms_by_structure.setdefault(atom.structure_id, []).append(
                    {
                        "sequence_index": atom.sequence_index,
                        "label": atom.label,
                        "element": atom.element,
                        "fractional_x": atom.fractional_x,
                        "fractional_y": atom.fractional_y,
                        "fractional_z": atom.fractional_z,
                        "occupancy": atom.occupancy,
                    }
                )
        primary_record_ids = [
            entries[0][0]
            for entries in provenance.values()
            if entries
        ]
        references = references_for_records(session, primary_record_ids)

        result: list[dict[str, Any]] = []
        for item in items:
            entries = provenance.get(item.id, [])
            source_record_id, primary = entries[0] if entries else (None, None)
            reference = references.get(source_record_id, {}) if source_record_id else {}
            result.append(
                {
                    "id": item.id,
                    "source": primary["source"] if primary else "local",
                    "external_id": primary["external_id"] if primary else str(item.id),
                    "source_url": primary["source_url"] if primary else None,
                    "material_id": item.adsorbent_id,
                    "material_name": material_names.get(item.adsorbent_id)
                    if item.adsorbent_id is not None
                    else None,
                    "name": item.name,
                    "formula": item.formula,
                    "format": item.format,
                    "content_sha256": item.content_sha256,
                    "space_group": item.space_group,
                    "space_group_number": item.space_group_number,
                    "cell_a_angstrom": item.cell_a_angstrom,
                    "cell_b_angstrom": item.cell_b_angstrom,
                    "cell_c_angstrom": item.cell_c_angstrom,
                    "cell_alpha_deg": item.cell_alpha_deg,
                    "cell_beta_deg": item.cell_beta_deg,
                    "cell_gamma_deg": item.cell_gamma_deg,
                    "cell_volume_angstrom3": item.cell_volume_angstrom3,
                    "has_coordinates": item.has_coordinates,
                    "atom_count": atom_counts.get(item.id, 0),
                    "doi": reference.get("doi"),
                    "reference_title": reference.get("title"),
                    "retrieved_at": primary["retrieved_at"] if primary else None,
                    "atoms": atoms_by_structure.get(item.id, []),
                }
            )
        return result

    # -------------------------------------------------------------------------
    def _structure_view(
        self, session: Session, item: Structure, *, include_atoms: bool
    ) -> dict[str, Any]:
        return self._structure_views(session, [item], include_atoms=include_atoms)[0]


__all__ = ["PublicDataRepository"]
