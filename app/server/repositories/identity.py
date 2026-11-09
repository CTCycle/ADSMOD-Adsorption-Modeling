# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""The single place where an incoming name becomes a canonical identity.

The resolver only reads the local database, so ingest never waits on a network
and never scans for fuzzy matches. Anything it cannot establish is returned as
a new row plus a review item instead of a guessed link.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.orm import Session

from server.common.identity import (
    FUZZY_REVIEW_THRESHOLD,
    STATUS_GENERIC,
    STATUS_PROVISIONAL,
    STATUS_RESOLVED,
    adsorbate_record_key,
    adsorbent_record_key,
    clean_name,
    exact_key,
    fuzzy_score,
    guard_signature,
    is_generic_adsorbent,
    is_inchi_key,
    names_compatible,
    separator_key,
)
from server.common.identity_catalog import (
    CURATED_ADSORBATES,
    CURATED_ADSORBENT_GROUPS,
    CuratedAdsorbate,
)
from server.repositories.database.manager import DatabaseManager
from server.repositories.schemas.models import (
    Adsorbate,
    Adsorbent,
    Isotherm,
    IsothermComponent,
    utc_now,
)
from server.repositories.schemas.public_data import (
    AdsorbateSourceRecord,
    AdsorbateSynonym,
    AdsorbentSourceRecord,
    AdsorbentSynonym,
    ChemicalProperty,
    IdentityReview,
    Structure,
)
from server.repositories.schemas.types import normalize_identity

# Short names are compared case-sensitively: CO/Co, HF/Hf, and NI/Ni are
# different species that casefold to the same text.
_CASE_SENSITIVE_LENGTH = 4

_CURATED_BY_FORMULA = {item.formula: item for item in CURATED_ADSORBATES}
_CURATED_BY_NAME = {
    exact_key(name): item
    for item in CURATED_ADSORBATES
    for name in (item.name, *item.aliases)
}
_CURATED_GROUP_BY_NAME = {
    exact_key(name): group for group in CURATED_ADSORBENT_GROUPS for name in group
}

METHOD_NEW = "new"
METHOD_GENERIC = "generic"


###############################################################################
class IdentityConflictError(ValueError):
    """Strong identifiers place one record on two canonical entities."""

    def __init__(
        self, message: str, kind: str, entity_id: int, candidate_id: int, name: str
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.entity_id = entity_id
        self.candidate_id = candidate_id
        self.name = name


###############################################################################
@dataclass(slots=True)
class PendingReview:
    reason: str
    candidate_ids: list[int] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


###############################################################################
@dataclass(slots=True)
class Resolution:
    entity: Any | None
    method: str
    review: PendingReview | None = None
    curated: CuratedAdsorbate | None = None


###############################################################################
def review_fingerprint(
    kind: str, name: str, entity_id: int | None, candidate_id: int | None
) -> str:
    """Identify a question so that a decided pair or name is never asked again."""
    if entity_id is not None and candidate_id is not None:
        low, high = sorted((entity_id, candidate_id))
        return f"{kind}:pair:{low}:{high}"
    digest = hashlib.sha256(exact_key(name).encode("utf-8")).hexdigest()[:40]
    return f"{kind}:name:{digest}"


###############################################################################
def raise_review(
    session: Session,
    *,
    kind: str,
    name: str,
    reason: str,
    entity_id: int | None = None,
    candidate_id: int | None = None,
    score: float | None = None,
    evidence: dict[str, Any] | None = None,
) -> bool:
    fingerprint = review_fingerprint(kind, name, entity_id, candidate_id)
    if session.scalar(
        select(IdentityReview.id).where(IdentityReview.fingerprint == fingerprint)
    ):
        return False
    session.add(
        IdentityReview(
            entity_kind=kind,
            entity_id=entity_id,
            candidate_id=candidate_id,
            raw_name=name[:1024],
            reason=reason,
            score=score,
            evidence=evidence or {},
            fingerprint=fingerprint,
        )
    )
    session.flush()
    return True


###############################################################################
def add_alias(
    session: Session,
    kind: str,
    entity_id: int,
    alias: str,
    origin: str,
    source_record_id: int | None = None,
) -> None:
    """Attach an alias, doing nothing when this entity already holds it."""
    model = AdsorbateSynonym if kind == "adsorbate" else AdsorbentSynonym
    owner = model.adsorbate_id if kind == "adsorbate" else model.adsorbent_id
    value = clean_name(alias)[:1024]
    if not value:
        return
    normalized = normalize_identity(value)
    if session.scalar(
        select(model.id).where(
            owner == entity_id,
            model.normalized_synonym == normalized,
            (
                model.source_record_id.is_(None)
                if source_record_id is None
                else model.source_record_id == source_record_id
            ),
        )
    ):
        return
    session.add(
        model(
            **{f"{kind}_id": entity_id},
            source_record_id=source_record_id,
            origin=origin,
            synonym=value,
            normalized_synonym=normalized,
        )
    )
    session.flush()


###############################################################################
class _Index:
    """Rows and aliases of one kind, loaded in batches for the current session."""

    def __init__(
        self, session: Session, kind: str, model: Any, alias_model: Any, strong: str
    ) -> None:
        self.session = session
        self.kind = kind
        self.model = model
        self.alias_model = alias_model
        self.strong_name = strong
        self.rows: dict[int, Any] = {}
        self.by_key: dict[str, int] = {}
        self.by_strong: dict[str, int] = {}
        self.by_smiles: dict[str, set[int]] = defaultdict(set)
        self.by_name: dict[str, dict[int, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.loaded_names: set[str] = set()
        self.loaded_keys: set[str] = set()
        self.loaded_strong: set[str] = set()
        self.loaded_smiles: set[str] = set()
        self.separators: dict[str, dict[int, str]] | None = None

    # -------------------------------------------------------------------------
    def remember(self, row: Any) -> None:
        self.rows[row.id] = row
        self.by_key[row.key] = row.id
        strong = getattr(row, self.strong_name)
        if strong:
            self.by_strong[strong] = row.id
        if getattr(row, "smiles", None):
            self.by_smiles[row.smiles].add(row.id)
        self.by_name[exact_key(row.name)][row.id].add(clean_name(row.name))
        if self.separators is not None:
            self.separators.setdefault(separator_key(row.name), {})[row.id] = row.name

    # -------------------------------------------------------------------------
    def load(
        self, keys: set[str], strong: set[str], names: set[str], smiles: set[str]
    ) -> None:
        keys -= self.loaded_keys
        strong -= self.loaded_strong
        names -= self.loaded_names
        smiles -= self.loaded_smiles
        self.loaded_smiles |= smiles
        self.loaded_keys |= keys
        self.loaded_strong |= strong
        self.loaded_names |= names
        if not (keys or strong or names or smiles):
            return
        column = getattr(self.model, self.strong_name)
        owner = (
            self.alias_model.adsorbate_id
            if self.kind == "adsorbate"
            else self.alias_model.adsorbent_id
        )
        aliases = (
            self.session.execute(
                select(owner, self.alias_model.synonym).where(
                    self.alias_model.normalized_synonym.in_(names)
                )
            ).all()
            if names
            else []
        )
        alias_owners = {row[0] for row in aliases} - set(self.rows)
        for chunk in self.session.scalars(
            select(self.model).where(
                or_(
                    self.model.key.in_(keys),
                    column.in_(strong),
                    self.model.normalized_name.in_(names),
                    self.model.id.in_(alias_owners),
                    *(
                        [self.model.smiles.in_(smiles)]
                        if smiles and self.kind == "adsorbate"
                        else []
                    ),
                )
            )
        ):
            self.remember(chunk)
        for entity_id, synonym in aliases:
            self.by_name[exact_key(synonym)][int(entity_id)].add(clean_name(synonym))

    # -------------------------------------------------------------------------
    def named(self, name: str) -> list[int]:
        """Ids holding this exact name; short names must match case as well."""
        cleaned = clean_name(name)
        forms = self.by_name.get(exact_key(name), {})
        if len(cleaned) <= _CASE_SENSITIVE_LENGTH:
            return sorted(key for key, held in forms.items() if cleaned in held)
        return sorted(forms)

    # -------------------------------------------------------------------------
    def separator_matches(self, name: str) -> dict[int, str]:
        """Rows whose name or alias differs only by separators, guards passing."""
        if len(clean_name(name)) <= _CASE_SENSITIVE_LENGTH:
            return {}
        if self.separators is None:
            self.separators = defaultdict(dict)
            for row in self.session.scalars(select(self.model)):
                self.rows.setdefault(row.id, row)
                self.separators[separator_key(row.name)][row.id] = row.name
            owner = (
                self.alias_model.adsorbate_id
                if self.kind == "adsorbate"
                else self.alias_model.adsorbent_id
            )
            for entity_id, synonym in self.session.execute(
                select(owner, self.alias_model.synonym)
            ):
                self.separators[separator_key(synonym)][int(entity_id)] = synonym
        return {
            entity_id: held
            for entity_id, held in self.separators.get(separator_key(name), {}).items()
            if names_compatible(name, held)
        }


###############################################################################
class IdentityResolver:
    """Resolves records within one session; create rows through ``register``."""

    def __init__(
        self,
        session: Session,
        *,
        dataset_id: int | None = None,
        alias_origin: str = "upload",
        read_only: bool = False,
    ) -> None:
        self.session = session
        self.dataset_id = dataset_id
        self.alias_origin = alias_origin
        self.read_only = read_only
        self.adsorbates = _Index(
            session, "adsorbate", Adsorbate, AdsorbateSynonym, "inchi_key"
        )
        self.adsorbents = _Index(
            session, "adsorbent", Adsorbent, AdsorbentSynonym, "external_identifier"
        )
        self.provisional_created = False

    # -------------------------------------------------------------------------
    def prefetch(self, experiments: Iterable[dict[str, Any]]) -> None:
        """Load every identity a batch can match using a fixed number of queries."""
        adsorbates: list[dict[str, Any]] = []
        adsorbents: list[dict[str, Any]] = []
        for experiment in experiments:
            adsorbates.extend(experiment["adsorbates"])
            adsorbents.append(experiment["adsorbent"])
        self._prefetch_adsorbates(adsorbates)
        self._prefetch_adsorbents(adsorbents)

    # -------------------------------------------------------------------------
    def _prefetch_adsorbates(self, records: list[dict[str, Any]]) -> None:
        names = {key for r in records for key in _lookup_names(r["name"])}
        strong = {r["inchi_key"] for r in records if is_inchi_key(r.get("inchi_key"))}
        for record in records:
            curated = curated_adsorbate(record["name"])
            if curated is not None:
                strong.add(curated.inchi_key)
        self.adsorbates.load(
            {adsorbate_record_key(r) for r in records},
            strong,
            names,
            {r["smiles"] for r in records if r.get("smiles")},
        )

    # -------------------------------------------------------------------------
    def _prefetch_adsorbents(self, records: list[dict[str, Any]]) -> None:
        names = {key for r in records for key in _lookup_names(r["name"])}
        for record in records:
            group = _CURATED_GROUP_BY_NAME.get(exact_key(record["name"]), ())
            names.update(key for name in group for key in _lookup_names(name))
        self.adsorbents.load(
            {adsorbent_record_key(r, self.dataset_id) for r in records},
            {r["external_identifier"] for r in records if r.get("external_identifier")},
            names,
            set(),
        )

    # -------------------------------------------------------------------------
    def resolve_adsorbate(self, record: dict[str, Any], key: str) -> Resolution:
        index = self.adsorbates
        self._prefetch_adsorbates([record])
        name = record["name"]
        inchi_key = record.get("inchi_key") if is_inchi_key(record.get("inchi_key")) else None
        smiles = record.get("smiles")
        strong = self._adsorbate_by_identifier(index, record, key, inchi_key, smiles)
        if strong is not None:
            return strong

        curated = curated_adsorbate(name) if not (inchi_key or smiles) else None
        if curated is not None and curated.inchi_key in index.by_strong:
            return self._matched(index, record, index.by_strong[curated.inchi_key], "curated")

        named = index.named(name)
        compatible = [
            entity_id
            for entity_id in named
            if not self._strong_conflict(index.rows[entity_id], inchi_key)
        ]
        if len(compatible) == 1 and not smiles_conflicts(index.rows[compatible[0]], smiles):
            return self._matched(index, record, compatible[0], "alias")
        if named:
            return Resolution(
                None,
                METHOD_NEW,
                PendingReview(
                    "ambiguous_alias" if len(named) > 1 else "identifier_conflict",
                    named,
                    {"inchi_key": inchi_key, "smiles": smiles},
                ),
            )
        if curated is not None:
            return Resolution(None, "curated", curated=curated)

        matches = index.separator_matches(name)
        if len(matches) == 1:
            return self._matched(index, record, next(iter(matches)), "separator")
        if matches:
            return Resolution(
                None,
                METHOD_NEW,
                PendingReview("ambiguous_alias", sorted(matches), {"names": matches}),
            )
        return Resolution(None, METHOD_NEW)

    # -------------------------------------------------------------------------
    def _adsorbate_by_identifier(
        self,
        index: _Index,
        record: dict[str, Any],
        key: str,
        inchi_key: str | None,
        smiles: str | None,
    ) -> Resolution | None:
        if inchi_key and inchi_key in index.by_strong:
            return self._matched(index, record, index.by_strong[inchi_key], "inchi_key")
        if key in index.by_key:
            return self._matched(index, record, index.by_key[key], "key")
        if smiles and len(index.by_smiles.get(smiles, ())) == 1:
            entity_id = next(iter(index.by_smiles[smiles]))
            if not self._strong_conflict(index.rows[entity_id], inchi_key):
                return self._matched(index, record, entity_id, "smiles")
        return None

    # -------------------------------------------------------------------------
    def resolve_adsorbent(self, record: dict[str, Any], key: str) -> Resolution:
        index = self.adsorbents
        self._prefetch_adsorbents([record])
        name = record["name"]
        external = record.get("external_identifier")
        if external and external in index.by_strong:
            return self._matched(index, record, index.by_strong[external], "external_identifier")
        if key in index.by_key:
            return self._matched(index, record, index.by_key[key], "key")
        if external is None and is_generic_adsorbent(name):
            return Resolution(None, METHOD_GENERIC)

        named = [
            entity_id
            for entity_id in index.named(name)
            if index.rows[entity_id].identity_status != STATUS_GENERIC
        ]
        if len(named) == 1:
            return self._matched(index, record, named[0], "alias")
        if named:
            return Resolution(
                None, METHOD_NEW, PendingReview("ambiguous_alias", named, {})
            )

        group = _CURATED_GROUP_BY_NAME.get(exact_key(name), ())
        curated_ids = sorted(
            {entity_id for member in group for entity_id in index.named(member)}
        )
        if len(curated_ids) == 1:
            return self._matched(index, record, curated_ids[0], "curated")
        if curated_ids:
            return Resolution(
                None,
                METHOD_NEW,
                PendingReview("curated_conflict", curated_ids, {"group": list(group)}),
            )

        # Material names carry sample detail, so a near-identical spelling is
        # only ever a proposal.
        matches = index.separator_matches(name)
        if matches:
            return Resolution(
                None,
                METHOD_NEW,
                PendingReview("separator_variant", sorted(matches), {"names": matches}),
            )
        return Resolution(None, METHOD_NEW)

    # -------------------------------------------------------------------------
    @staticmethod
    def _strong_conflict(row: Any, inchi_key: str | None) -> bool:
        return bool(inchi_key and row.inchi_key and row.inchi_key != inchi_key)

    # -------------------------------------------------------------------------
    def _matched(
        self, index: _Index, record: dict[str, Any], entity_id: int, method: str
    ) -> Resolution:
        entity = index.rows[entity_id]
        name = record["name"]
        if (
            not self.read_only
            and method != "key"
            and exact_key(name) != exact_key(entity.name)
            and entity_id not in index.by_name.get(exact_key(name), {})
        ):
            add_alias(self.session, index.kind, entity_id, name, self.alias_origin)
            index.by_name[exact_key(name)][entity_id].add(clean_name(name))
        return Resolution(entity, method)

    # -------------------------------------------------------------------------
    def register(
        self, kind: str, entity: Any, record: dict[str, Any], resolution: Resolution
    ) -> None:
        """Index a row created for ``resolution`` and file its review, if any."""
        index = self.adsorbates if kind == "adsorbate" else self.adsorbents
        index.remember(entity)
        if entity.identity_status == "provisional":
            self.provisional_created = True
        review = resolution.review
        if review is None:
            return
        for candidate_id in review.candidate_ids or [None]:
            if candidate_id == entity.id:
                continue
            raise_review(
                self.session,
                kind=kind,
                name=record["name"],
                reason=review.reason,
                entity_id=entity.id,
                candidate_id=candidate_id,
                evidence=_json_evidence(review.evidence),
            )


###############################################################################
def _lookup_names(name: str) -> set[str]:
    """Stored normalized forms a name may be held under."""
    return {normalize_identity(name), exact_key(name)}


###############################################################################
def curated_adsorbate(name: str) -> CuratedAdsorbate | None:
    return _CURATED_BY_FORMULA.get(clean_name(name)) or _CURATED_BY_NAME.get(
        exact_key(name)
    )


###############################################################################
def smiles_conflicts(row: Any, smiles: str | None) -> bool:
    return False if not smiles or not row.smiles else row.smiles != smiles


###############################################################################
def _json_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    if "names" not in evidence:
        return evidence
    return {**evidence, "names": {str(k): v for k, v in evidence["names"].items()}}


###############################################################################
class MergeRefusedError(ValueError):
    """A merge would destroy scientific information and was not applied."""


###############################################################################
class IdentityRepository:
    """Review queue, read-only audit, and the only code path that merges rows."""

    # -------------------------------------------------------------------------
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    # -------------------------------------------------------------------------
    def preview(self, experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Report how an import would resolve, without writing anything."""
        seen: dict[tuple[str, str], dict[str, Any]] = {}
        with self.database.session_factory() as session:
            resolver = IdentityResolver(session, read_only=True)
            resolver.prefetch(experiments)
            for experiment in experiments:
                record = experiment["adsorbent"]
                seen.setdefault(
                    ("adsorbent", record["name"]),
                    _preview_row(
                        "adsorbent",
                        record["name"],
                        resolver.resolve_adsorbent(
                            record, adsorbent_record_key(record, None)
                        ),
                    ),
                )
                for record in experiment["adsorbates"]:
                    seen.setdefault(
                        ("adsorbate", record["name"]),
                        _preview_row(
                            "adsorbate",
                            record["name"],
                            resolver.resolve_adsorbate(
                                record, adsorbate_record_key(record)
                            ),
                        ),
                    )
        return list(seen.values())

    # -------------------------------------------------------------------------
    def list_reviews(
        self, *, status: str, kind: str | None, page: int, page_size: int
    ) -> tuple[list[dict[str, Any]], int]:
        with self.database.session_factory() as session:
            query = select(IdentityReview).where(IdentityReview.status == status)
            if kind:
                query = query.where(IdentityReview.entity_kind == kind)
            total = int(
                session.scalar(select(func.count()).select_from(query.subquery())) or 0
            )
            reviews = session.scalars(
                query.order_by(IdentityReview.id)
                .offset((page - 1) * page_size)
                .limit(page_size)
            ).all()
            return [self._review_view(session, review) for review in reviews], total

    # -------------------------------------------------------------------------
    @staticmethod
    def _entity_view(
        session: Session, kind: str, entity_id: int | None
    ) -> dict[str, Any] | None:
        if entity_id is None:
            return None
        model, identifier, owner = _usage_columns(kind)
        row = session.get(model, entity_id)
        if row is None:
            return None
        count = session.scalar(select(func.count()).where(owner == entity_id))
        return {
            "id": row.id,
            "name": row.name,
            "identity_status": row.identity_status,
            "identifier": getattr(row, identifier),
            "experiment_count": int(count or 0),
        }

    # -------------------------------------------------------------------------
    def _review_view(self, session: Session, review: IdentityReview) -> dict[str, Any]:
        kind = review.entity_kind
        return {
            "id": review.id,
            "kind": kind,
            "raw_name": review.raw_name,
            "reason": review.reason,
            "score": review.score,
            "evidence": review.evidence,
            "status": review.status,
            "entity": self._entity_view(session, kind, review.entity_id),
            "candidate": self._entity_view(session, kind, review.candidate_id),
            "created_at": review.created_at,
            "decided_at": review.decided_at,
        }

    # -------------------------------------------------------------------------
    def reject(self, review_id: int) -> None:
        with self.database.transaction() as session:
            review = self._open_review(session, review_id)
            review.status = "rejected"
            review.decided_at = utc_now()

    # -------------------------------------------------------------------------
    @staticmethod
    def _open_review(session: Session, review_id: int) -> IdentityReview:
        review = session.get(IdentityReview, review_id)
        if review is None:
            raise LookupError(f"Identity review {review_id} was not found.")
        if review.status != "open":
            raise MergeRefusedError("This review has already been decided.")
        return review

    # -------------------------------------------------------------------------
    def merge(self, review_id: int, keep_id: int) -> None:
        """Fold one row into the other in a single transaction.

        Measurements and original source names are not modified; only the
        references to the removed row move.
        """
        with self.database.transaction() as session:
            review = self._open_review(session, review_id)
            pair = {review.entity_id, review.candidate_id}
            if None in pair or keep_id not in pair:
                raise MergeRefusedError("Choose one of the two records in this review.")
            remove_id = int(next(iter(pair - {keep_id})))
            merge = (
                _merge_adsorbates
                if review.entity_kind == "adsorbate"
                else _merge_adsorbents
            )
            review.merge_snapshot = merge(session, keep_id, remove_id)
            review.status = "merged"
            review.decided_at = utc_now()
            session.execute(
                delete(IdentityReview).where(
                    IdentityReview.status == "open",
                    IdentityReview.entity_kind == review.entity_kind,
                    IdentityReview.id != review.id,
                    or_(
                        IdentityReview.entity_id == remove_id,
                        IdentityReview.candidate_id == remove_id,
                    ),
                )
            )

    # -------------------------------------------------------------------------
    def audit(self) -> dict[str, int]:
        """File proposals for likely duplicates; canonical rows are not changed."""
        with self.database.transaction() as session:
            return {
                "adsorbate": self._audit_kind(session, "adsorbate"),
                "adsorbent": self._audit_kind(session, "adsorbent"),
            }

    # -------------------------------------------------------------------------
    @staticmethod
    def _audit_kind(session: Session, kind: str) -> int:
        model = Adsorbate if kind == "adsorbate" else Adsorbent
        rows = [
            row
            for row in session.scalars(select(model).order_by(model.id))
            if row.identity_status != STATUS_GENERIC
        ]
        created = 0
        groups: dict[tuple[object, ...], list[Any]] = defaultdict(list)
        for row in rows:
            groups[guard_signature(row.name)].append(row)
        for group in groups.values():
            for position, left in enumerate(group):
                for right in group[position + 1 :]:
                    if kind == "adsorbate" and left.inchi_key and right.inchi_key:
                        continue
                    score = fuzzy_score(left.name, right.name)
                    if score < FUZZY_REVIEW_THRESHOLD:
                        continue
                    created += raise_review(
                        session,
                        kind=kind,
                        name=left.name,
                        reason="separator_variant" if score == 1.0 else "fuzzy_name",
                        entity_id=left.id,
                        candidate_id=right.id,
                        score=round(score, 3),
                        evidence={"names": [left.name, right.name]},
                    )
        if kind == "adsorbent":
            created += IdentityRepository._audit_curated(session, rows)
        return created

    # -------------------------------------------------------------------------
    @staticmethod
    def _audit_curated(session: Session, rows: list[Any]) -> int:
        created = 0
        for group in CURATED_ADSORBENT_GROUPS:
            names = {exact_key(name) for name in group}
            members = [row for row in rows if exact_key(row.name) in names]
            for candidate in members[1:]:
                created += raise_review(
                    session,
                    kind="adsorbent",
                    name=members[0].name,
                    reason="curated_synonym",
                    entity_id=members[0].id,
                    candidate_id=candidate.id,
                    evidence={"group": list(group)},
                )
        return created


###############################################################################
def _usage_columns(kind: str) -> tuple[Any, str, Any]:
    if kind == "adsorbate":
        return Adsorbate, "inchi_key", IsothermComponent.adsorbate_id
    return Adsorbent, "external_identifier", Isotherm.adsorbent_id


###############################################################################
def _preview_row(kind: str, received: str, resolution: Resolution) -> dict[str, Any]:
    entity = resolution.entity
    canonical = entity.name if entity is not None else None
    if resolution.curated is not None:
        canonical = resolution.curated.name
    if entity is not None:
        status = entity.identity_status
    elif resolution.curated is not None:
        status = STATUS_RESOLVED
    elif resolution.method == METHOD_GENERIC:
        status = STATUS_GENERIC
    else:
        status = STATUS_PROVISIONAL
    return {
        "kind": kind,
        "received_name": received,
        "canonical_name": canonical,
        "method": resolution.method,
        "identity_status": status,
        "needs_review": resolution.review is not None,
    }


###############################################################################
def _snapshot(row: Any) -> dict[str, Any]:
    fields = ("key", "name", "formula", "molar_mass_g_mol", "smiles", "identity_status")
    return {name: getattr(row, name) for name in fields}


###############################################################################
def _merge_rows(
    session: Session, kind: str, keep_id: int, remove_id: int
) -> tuple[Any, Any]:
    model = Adsorbate if kind == "adsorbate" else Adsorbent
    keep = session.get(model, keep_id)
    remove = session.get(model, remove_id)
    if keep is None or remove is None:
        raise LookupError("One of the records in this review no longer exists.")
    return keep, remove


###############################################################################
def _move_aliases(session: Session, kind: str, keep_id: int, remove: Any) -> None:
    alias_model = AdsorbateSynonym if kind == "adsorbate" else AdsorbentSynonym
    owner = alias_model.adsorbate_id if kind == "adsorbate" else alias_model.adsorbent_id
    held = {
        (source, normalized, None if source is not None else origin)
        for source, normalized, origin in session.execute(
            select(
                alias_model.source_record_id,
                alias_model.normalized_synonym,
                alias_model.origin,
            ).where(owner == keep_id)
        )
    }
    for alias in session.scalars(select(alias_model).where(owner == remove.id)):
        signature = (
            alias.source_record_id,
            alias.normalized_synonym,
            None if alias.source_record_id is not None else alias.origin,
        )
        if signature in held:
            session.delete(alias)
        else:
            held.add(signature)
            setattr(alias, f"{kind}_id", keep_id)
    session.flush()
    add_alias(session, kind, keep_id, remove.name, "manual")


###############################################################################
def _fill_missing(keep: Any, remove: Any, fields: tuple[str, ...]) -> None:
    for name in fields:
        if getattr(keep, name) is None:
            setattr(keep, name, getattr(remove, name))


###############################################################################
def _merge_adsorbates(session: Session, keep_id: int, remove_id: int) -> dict[str, Any]:
    keep, remove = _merge_rows(session, "adsorbate", keep_id, remove_id)
    if keep.inchi_key and remove.inchi_key and keep.inchi_key != remove.inchi_key:
        raise MergeRefusedError(
            "These records have different InChIKeys, so they are different compounds."
        )
    shared = session.scalar(
        select(func.count()).where(
            IsothermComponent.adsorbate_id == remove_id,
            IsothermComponent.isotherm_id.in_(
                select(IsothermComponent.isotherm_id).where(
                    IsothermComponent.adsorbate_id == keep_id
                )
            ),
        )
    )
    if shared:
        raise MergeRefusedError(
            f"{shared} experiment(s) list both species as separate components."
        )
    snapshot = {**_snapshot(remove), "id": remove_id, "inchi_key": remove.inchi_key}
    moved = session.execute(
        update(IsothermComponent)
        .where(IsothermComponent.adsorbate_id == remove_id)
        .values(adsorbate_id=keep_id)
    )
    for link in (AdsorbateSourceRecord, ChemicalProperty):
        session.execute(
            update(link).where(link.adsorbate_id == remove_id).values(adsorbate_id=keep_id)
        )
    _move_aliases(session, "adsorbate", keep_id, remove)
    # The unique InChIKey must leave the removed row before the survivor takes it.
    inchi_key, remove.inchi_key = remove.inchi_key, None
    session.flush()
    keep.inchi_key = keep.inchi_key or inchi_key
    _fill_missing(keep, remove, ("inchi", "formula", "molar_mass_g_mol", "smiles"))
    if keep.inchi_key:
        keep.identity_status = STATUS_RESOLVED
    session.delete(remove)
    session.flush()
    return {**snapshot, "moved_experiments": int(moved.rowcount or 0)}


###############################################################################
def _merge_adsorbents(session: Session, keep_id: int, remove_id: int) -> dict[str, Any]:
    keep, remove = _merge_rows(session, "adsorbent", keep_id, remove_id)
    snapshot = {
        **_snapshot(remove),
        "id": remove_id,
        "external_identifier": remove.external_identifier,
    }
    moved = session.execute(
        update(Isotherm)
        .where(Isotherm.adsorbent_id == remove_id)
        .values(adsorbent_id=keep_id)
    )
    for link in (AdsorbentSourceRecord, Structure):
        session.execute(
            update(link).where(link.adsorbent_id == remove_id).values(adsorbent_id=keep_id)
        )
    _move_aliases(session, "adsorbent", keep_id, remove)
    # A second source identifier survives as the source-record link just moved.
    external, remove.external_identifier = remove.external_identifier, None
    session.flush()
    keep.external_identifier = keep.external_identifier or external
    _fill_missing(keep, remove, ("formula", "molar_mass_g_mol", "smiles"))
    if keep.external_identifier:
        keep.identity_status = STATUS_RESOLVED
    if keep.scope_dataset_id != remove.scope_dataset_id:
        keep.scope_dataset_id = None
    session.delete(remove)
    session.flush()
    return {**snapshot, "moved_experiments": int(moved.rowcount or 0)}
