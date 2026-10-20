# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import String, cast, delete, func, literal, or_, select, union, update
from sqlalchemy.orm import Session

from server.repositories.database.manager import DatabaseManager
from server.repositories.datasets import stable_material_key
from server.repositories.schemas.models import (
    Adsorbate,
    Adsorbent,
    Dataset,
    Isotherm,
    IsothermComponent,
    Observation,
)
from server.repositories.schemas.public_data import (
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
from server.repositories.schemas.types import normalize_identity


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
                    .select_from(self._public_adsorbent_ids().subquery())
                    .scalar_subquery()
                    .label("materials"),
                    select(func.count())
                    .select_from(self._public_adsorbate_ids().subquery())
                    .scalar_subquery()
                    .label("chemicals"),
                    select(func.count(func.distinct(StructureSourceRecord.structure_id)))
                    .scalar_subquery()
                    .label("structures"),
                    select(func.count(PublicDataSyncRun.id))
                    .where(PublicDataSyncRun.status.in_(("queued", "running")))
                    .scalar_subquery()
                    .label("active_sync_runs"),
                    select(func.count(PublicDataSyncRun.id))
                    .where(
                        PublicDataSyncRun.status.in_(
                            ("failed", "partially_completed")
                        )
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
                "active_sync_runs": int(row.active_sync_runs or 0),
                "failed_sync_runs": int(row.failed_sync_runs or 0),
                "latest_successful_sync_at": row.latest_successful_sync_at,
            }

    # -------------------------------------------------------------------------
    @staticmethod
    def _public_adsorbate_ids(source: str | None = None) -> Any:
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
        if source:
            linked = linked.where(DataSource.key == source)
            measured = measured.where(DataSource.key == source)
        return union(linked, measured)

    # -------------------------------------------------------------------------
    @staticmethod
    def _public_adsorbent_ids(source: str | None = None) -> Any:
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
        if source:
            linked = linked.where(DataSource.key == source)
            measured = measured.where(DataSource.key == source)
        return union(linked, measured)

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
        run = PublicDataSyncRun(
            id=str(uuid4()),
            source_key=source_key,
            operation=operation,
            parameters=parameters,
            status="queued",
        )
        with self.database.transaction() as session:
            self._source(session, source_key)
            session.add(run)
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
    ) -> tuple[list[dict[str, Any]], int]:
        with self.database.session_factory() as session:
            query = select(PublicDataSyncRun)
            if source_key:
                query = query.where(PublicDataSyncRun.source_key == source_key)
            if status_filter:
                query = query.where(PublicDataSyncRun.status == status_filter)
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

    # -------------------------------------------------------------------------
    def link_adsorbent_record(
        self,
        *,
        source_key: str,
        adsorbent_id: int,
        external_id: str,
        source_url: str | None = None,
        raw_metadata: dict[str, Any] | None = None,
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
    ) -> None:
        with self.database.transaction() as session:
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
            raise ValueError(f"PubChem CID {cid} is already linked to another chemical.")
        if inchi_key and session.scalar(
            select(Adsorbate.id).where(
                Adsorbate.inchi_key == inchi_key, Adsorbate.id != adsorbate_id
            )
        ):
            raise ValueError(
                f"PubChem CID {cid} matches another chemical's InChIKey; "
                "automatic merging was refused."
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
            return int(matches[0]) if len(matches) == 1 else None

    # -------------------------------------------------------------------------
    def pubchem_enrichment_candidates(self, limit: int) -> list[dict[str, Any]]:
        """Return chemicals without PubChem identity that may still resolve.

        Chemicals whose earlier lookup failed permanently (no retry time) are
        excluded so repeated enrichment runs make progress.
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
                select(Adsorbate.id, Adsorbate.name, Adsorbate.inchi_key)
                .where(
                    Adsorbate.id.not_in(linked),
                    (literal("adsorbate:") + cast(Adsorbate.id, String)).not_in(
                        unresolvable
                    ),
                )
                .order_by(Adsorbate.id)
                .limit(max(1, limit))
            ).all()
            return [
                {"id": int(row.id), "query": row.inchi_key or row.name}
                for row in rows
            ]

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
    @staticmethod
    def _provenance_entries(
        session: Session, entity: str, entity_ids: list[int]
    ) -> dict[int, list[tuple[int, dict[str, Any]]]]:
        if not entity_ids:
            return {}
        if entity == "adsorbate":
            link_model = AdsorbateSourceRecord
            id_column = link_model.adsorbate_id
        elif entity == "adsorbent":
            link_model = AdsorbentSourceRecord
            id_column = link_model.adsorbent_id
        elif entity == "isotherm":
            link_model = IsothermSourceRecord
            id_column = link_model.isotherm_id
        elif entity == "structure":
            link_model = StructureSourceRecord
            id_column = link_model.structure_id
        else:
            raise ValueError(f"Unsupported provenance entity: {entity}")

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
            .order_by(
                id_column,
                SourceRecord.retrieved_at.desc(),
                SourceRecord.id.desc(),
            )
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

    # -------------------------------------------------------------------------
    @classmethod
    def _external_identifiers(
        cls, session: Session, entity: str, entity_id: int
    ) -> list[dict[str, Any]]:
        return [
            identifier
            for _, identifier in cls._provenance_entries(session, entity, [entity_id]).get(
                entity_id, []
            )
        ]

    # -------------------------------------------------------------------------
    @staticmethod
    def _references_for_records(
        session: Session, source_record_ids: list[int]
    ) -> dict[int, dict[str, str | None]]:
        if not source_record_ids:
            return {}
        rows = session.execute(
            select(
                SourceRecordReference.source_record_id,
                Reference.doi,
                Reference.title,
            )
            .join(
                Reference,
                SourceRecordReference.reference_id == Reference.id,
            )
            .where(SourceRecordReference.source_record_id.in_(source_record_ids))
            .order_by(SourceRecordReference.source_record_id, Reference.id)
        ).all()
        result: dict[int, dict[str, str | None]] = {}
        for row in rows:
            result.setdefault(
                int(row.source_record_id), {"doi": row.doi, "title": row.title}
            )
        return result

    # -------------------------------------------------------------------------
    def list_adsorption(
        self,
        *,
        page: int,
        page_size: int,
        source: str | None = None,
        material: str | None = None,
        adsorbate: str | None = None,
        material_id: int | None = None,
        adsorbate_id: int | None = None,
        temperature_min_k: float | None = None,
        temperature_max_k: float | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        with self.database.session_factory() as session:
            query = (
                select(
                    Isotherm.id,
                    Isotherm.external_key,
                    Isotherm.temperature_k,
                    Isotherm.pressure_basis,
                    Dataset.source.label("dataset_source"),
                    Adsorbent.id.label("material_id"),
                    Adsorbent.name.label("material"),
                )
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .join(Adsorbent, Adsorbent.id == Isotherm.adsorbent_id)
            )
            if material_id is not None:
                query = query.where(Isotherm.adsorbent_id == material_id)
            if adsorbate_id is not None:
                query = query.where(
                    Isotherm.id.in_(
                        select(IsothermComponent.isotherm_id).where(
                            IsothermComponent.adsorbate_id == adsorbate_id
                        )
                    )
                )
            if source:
                source_ids = (
                    select(IsothermSourceRecord.isotherm_id)
                    .join(
                        SourceRecord,
                        SourceRecord.id == IsothermSourceRecord.source_record_id,
                    )
                    .join(DataSource, DataSource.id == SourceRecord.source_id)
                    .where(DataSource.key == source)
                )
                query = query.where(Isotherm.id.in_(source_ids))
            if material:
                query = query.where(Adsorbent.name.ilike(f"%{material.strip()}%"))
            if adsorbate:
                matching_adsorbates = (
                    select(IsothermComponent.isotherm_id)
                    .join(Adsorbate, Adsorbate.id == IsothermComponent.adsorbate_id)
                    .where(Adsorbate.name.ilike(f"%{adsorbate.strip()}%"))
                )
                query = query.where(Isotherm.id.in_(matching_adsorbates))
            if temperature_min_k is not None:
                query = query.where(Isotherm.temperature_k >= temperature_min_k)
            if temperature_max_k is not None:
                query = query.where(Isotherm.temperature_k <= temperature_max_k)

            total = int(
                session.scalar(select(func.count()).select_from(query.subquery())) or 0
            )
            rows = session.execute(
                query.order_by(Isotherm.id.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            ).all()
            return self._adsorption_summaries(session, rows), total

    # -------------------------------------------------------------------------
    def _adsorption_summaries(
        self, session: Session, rows: list[Any]
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        isotherm_ids = [int(row.id) for row in rows]
        adsorbates_by_isotherm: dict[int, list[dict[str, Any]]] = {}
        for isotherm_id, adsorbate_id, adsorbate_name in session.execute(
            select(IsothermComponent.isotherm_id, Adsorbate.id, Adsorbate.name)
            .join(Adsorbate, IsothermComponent.adsorbate_id == Adsorbate.id)
            .where(IsothermComponent.isotherm_id.in_(isotherm_ids))
            .order_by(IsothermComponent.isotherm_id, IsothermComponent.position)
        ):
            adsorbates_by_isotherm.setdefault(int(isotherm_id), []).append(
                {"id": int(adsorbate_id), "name": str(adsorbate_name)}
            )

        stats_by_isotherm = {
            int(row.isotherm_id): row
            for row in session.execute(
                select(
                    Observation.isotherm_id.label("isotherm_id"),
                    func.min(Observation.pressure_canonical).label("pressure_min"),
                    func.max(Observation.pressure_canonical).label("pressure_max"),
                    func.min(Observation.uptake_mol_kg).label("uptake_min"),
                    func.max(Observation.uptake_mol_kg).label("uptake_max"),
                    func.count(Observation.id).label("point_count"),
                )
                .where(Observation.isotherm_id.in_(isotherm_ids))
                .group_by(Observation.isotherm_id)
            )
        }
        provenance_by_isotherm = self._provenance_entries(
            session, "isotherm", isotherm_ids
        )
        primary_record_ids = [
            entries[0][0]
            for entries in provenance_by_isotherm.values()
            if entries
        ]
        references = self._references_for_records(session, primary_record_ids)

        items: list[dict[str, Any]] = []
        for row in rows:
            isotherm_id = int(row.id)
            entries = provenance_by_isotherm.get(isotherm_id, [])
            source_record_id, primary = entries[0] if entries else (None, None)
            stats = stats_by_isotherm.get(isotherm_id)
            reference = references.get(source_record_id, {}) if source_record_id else {}
            items.append(
                {
                    "id": isotherm_id,
                    "external_id": primary["external_id"] if primary else row.external_key,
                    "source": primary["source"] if primary else row.dataset_source,
                    "source_url": primary["source_url"] if primary else None,
                    "material_id": int(row.material_id),
                    "material": row.material,
                    "adsorbates": adsorbates_by_isotherm.get(isotherm_id, []),
                    "temperature_k": row.temperature_k,
                    "pressure_min_pa": stats.pressure_min if stats else None,
                    "pressure_max_pa": stats.pressure_max if stats else None,
                    "uptake_min_mol_kg": stats.uptake_min if stats else None,
                    "uptake_max_mol_kg": stats.uptake_max if stats else None,
                    "point_count": int(stats.point_count if stats else 0),
                    "doi": reference.get("doi"),
                    "reference_title": reference.get("title"),
                    "retrieved_at": primary["retrieved_at"] if primary else None,
                }
            )
        return items

    # -------------------------------------------------------------------------
    def _adsorption_summary(self, session: Session, row: Any) -> dict[str, Any]:
        return self._adsorption_summaries(session, [row])[0]

    # -------------------------------------------------------------------------
    def get_adsorption(self, isotherm_id: int) -> dict[str, Any]:
        with self.database.session_factory() as session:
            row = session.execute(
                select(
                    Isotherm.id,
                    Isotherm.external_key,
                    Isotherm.temperature_k,
                    Isotherm.pressure_basis,
                    Isotherm.conditions,
                    Isotherm.provenance,
                    Dataset.source.label("dataset_source"),
                    Adsorbent.id.label("material_id"),
                    Adsorbent.name.label("material"),
                )
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .join(Adsorbent, Adsorbent.id == Isotherm.adsorbent_id)
                .where(Isotherm.id == isotherm_id)
            ).first()
            if row is None:
                raise LookupError(f"Adsorption record {isotherm_id} was not found.")
            summary = self._adsorption_summary(session, row)
            measurements = session.execute(
                select(
                    Observation.sequence_index,
                    Adsorbate.name,
                    Observation.pressure_original,
                    Observation.pressure_original_unit,
                    Observation.pressure_canonical,
                    Observation.uptake_original,
                    Observation.uptake_original_unit,
                    Observation.uptake_mol_kg,
                )
                .join(
                    IsothermComponent,
                    IsothermComponent.id == Observation.component_id,
                )
                .join(Adsorbate, Adsorbate.id == IsothermComponent.adsorbate_id)
                .where(Observation.isotherm_id == isotherm_id)
                .order_by(Observation.sequence_index, IsothermComponent.position)
            ).all()
            summary.update(
                {
                    "pressure_basis": row.pressure_basis,
                    "conditions": dict(row.conditions),
                    "provenance": dict(row.provenance),
                    "measurements": [
                        {
                            "sequence_index": item[0],
                            "adsorbate": item[1],
                            "pressure_original": item[2],
                            "pressure_original_unit": item[3],
                            "pressure_pa": item[4],
                            "uptake_original": item[5],
                            "uptake_original_unit": item[6],
                            "uptake_mol_kg": item[7],
                        }
                        for item in measurements
                    ],
                    "external_identifiers": self._external_identifiers(
                        session, "isotherm", isotherm_id
                    ),
                }
            )
            return summary

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
                ids = self._public_adsorbent_ids(source).subquery()
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
                query.order_by(Adsorbent.name)
                .offset((page - 1) * page_size)
                .limit(page_size)
            ).all()
            return self._material_views(session, rows), total

    # -------------------------------------------------------------------------
    def _material_views(
        self, session: Session, items: list[Adsorbent]
    ) -> list[dict[str, Any]]:
        row_ids = [item.id for item in items]
        structure_counts = self._counts_by(
            session, Structure.adsorbent_id, Structure.id, row_ids
        )
        isotherm_counts = self._counts_by(
            session, Isotherm.adsorbent_id, Isotherm.id, row_ids
        )
        provenance = self._provenance_entries(session, "adsorbent", row_ids)
        return [
            {
                "id": item.id,
                "name": item.name,
                "formula": item.formula,
                "molar_mass_g_mol": item.molar_mass_g_mol,
                "structure_count": structure_counts.get(item.id, 0),
                "isotherm_count": isotherm_counts.get(item.id, 0),
                "external_identifiers": [
                    identifier for _, identifier in provenance.get(item.id, [])
                ],
            }
            for item in items
        ]

    # -------------------------------------------------------------------------
    @staticmethod
    def _counts_by(
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

    # -------------------------------------------------------------------------
    def get_material(self, adsorbent_id: int) -> dict[str, Any]:
        with self.database.session_factory() as session:
            item = session.get(Adsorbent, adsorbent_id)
            if item is None:
                raise LookupError(f"Material {adsorbent_id} was not found.")
            view = self._material_views(session, [item])[0]
            structures = session.scalars(
                select(Structure)
                .where(Structure.adsorbent_id == adsorbent_id)
                .order_by(Structure.updated_at.desc())
            ).all()
            view["structures"] = self._structure_views(
                session, list(structures), include_atoms=False
            )
            return view

    # -------------------------------------------------------------------------
    def list_chemicals(
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
                ids = self._public_adsorbate_ids(source).subquery()
                query = query.where(Adsorbate.id.in_(select(ids.c.entity_id)))
            total = int(
                session.scalar(select(func.count()).select_from(query.subquery())) or 0
            )
            rows = session.scalars(
                query.order_by(Adsorbate.name)
                .offset((page - 1) * page_size)
                .limit(page_size)
            ).all()
            return self._chemical_views(session, rows), total

    # -------------------------------------------------------------------------
    def get_chemical(self, adsorbate_id: int) -> dict[str, Any]:
        with self.database.session_factory() as session:
            item = session.get(Adsorbate, adsorbate_id)
            if item is None:
                raise LookupError(f"Chemical {adsorbate_id} was not found.")
            return self._chemical_view(session, item)

    # -------------------------------------------------------------------------
    def _chemical_views(
        self, session: Session, items: list[Adsorbate]
    ) -> list[dict[str, Any]]:
        if not items:
            return []
        item_ids = [item.id for item in items]
        provenance = self._provenance_entries(session, "adsorbate", item_ids)
        properties_by_adsorbate: dict[int, list[tuple[ChemicalProperty, str]]] = {}
        for adsorbate_id, prop, source_key in session.execute(
            select(ChemicalProperty.adsorbate_id, ChemicalProperty, DataSource.key)
            .join(SourceRecord, SourceRecord.id == ChemicalProperty.source_record_id)
            .join(DataSource, DataSource.id == SourceRecord.source_id)
            .where(ChemicalProperty.adsorbate_id.in_(item_ids))
            .order_by(ChemicalProperty.adsorbate_id, ChemicalProperty.key, ChemicalProperty.id)
        ):
            properties_by_adsorbate.setdefault(int(adsorbate_id), []).append(
                (prop, str(source_key))
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

        isotherm_counts = self._counts_by(
            session, IsothermComponent.adsorbate_id, IsothermComponent.isotherm_id, item_ids
        )
        return [
            {
                **self._chemical_payload(
                    item,
                    [identifier for _, identifier in provenance.get(item.id, [])],
                    properties_by_adsorbate.get(item.id, []),
                    synonyms_by_adsorbate.get(item.id, []),
                ),
                "isotherm_count": isotherm_counts.get(item.id, 0),
            }
            for item in items
        ]

    # -------------------------------------------------------------------------
    def _chemical_view(self, session: Session, item: Adsorbate) -> dict[str, Any]:
        return self._chemical_views(session, [item])[0]

    # -------------------------------------------------------------------------
    @staticmethod
    def _chemical_payload(
        item: Adsorbate,
        identifiers: list[dict[str, Any]],
        properties: list[tuple[ChemicalProperty, str]],
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
                if prop.key not in {"preferred_name", "connectivity_smiles", "pubchem_3d_available"}
            ],
            "external_identifiers": identifiers,
            "structure_2d_url": (
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG?image_size=300x300"
                if cid
                else None
            ),
            "conformer_3d_url": (
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/record/SDF?record_type=3d"
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
        provenance = self._provenance_entries(session, "structure", item_ids)
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
        references = self._references_for_records(session, primary_record_ids)

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
