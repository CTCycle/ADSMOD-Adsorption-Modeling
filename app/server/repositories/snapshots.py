# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import uuid

from sqlalchemy import insert, select

from server.repositories.database.manager import DatabaseManager
from server.repositories.schemas.models import (
    TrainingSnapshot,
    TrainingSnapshotRow,
)

###############################################################################
@dataclass(frozen=True)
class SnapshotRecord:
    snapshot_id: str
    content_hash: str
    created_at: str
    row_count: int

###############################################################################
@dataclass(frozen=True)
class SnapshotPage:
    snapshot_id: str
    content_hash: str
    page: int
    page_size: int
    total_rows: int
    rows: tuple[dict[str, Any], ...]

###############################################################################
class SnapshotHasher:
    """Incrementally hashes the canonical JSON array of snapshot rows."""

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self._digest = hashlib.sha256(b"[")
        self._empty = True

    # -------------------------------------------------------------------------
    def update(self, row: dict[str, Any]) -> None:
        encoded = json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        self._digest.update(("" if self._empty else ",").encode() + encoded.encode("utf-8"))
        self._empty = False

    # -------------------------------------------------------------------------
    def hexdigest(self) -> str:
        digest = self._digest.copy()
        digest.update(b"]")
        return digest.hexdigest()

###############################################################################
class SnapshotStore:
    """Core-owned immutable snapshot storage in the operational database."""

    INSERT_CHUNK_SIZE = 5000

    # -------------------------------------------------------------------------
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    # -------------------------------------------------------------------------
    def create(
        self,
        rows: list[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> SnapshotRecord:
        if not rows:
            raise ValueError("A snapshot must contain at least one row.")
        hasher = SnapshotHasher()
        for row in rows:
            hasher.update(row)
        content_hash = hasher.hexdigest()
        snapshot_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        with self.database.transaction() as session:
            existing = session.scalar(
                select(TrainingSnapshot).where(
                    TrainingSnapshot.content_hash == content_hash
                )
            )
            if existing is not None:
                return SnapshotRecord(
                    snapshot_id=existing.snapshot_id,
                    content_hash=existing.content_hash,
                    created_at=existing.created_at.isoformat(),
                    row_count=existing.row_count,
                )
            session.add(
                TrainingSnapshot(
                    snapshot_id=snapshot_id,
                    content_hash=content_hash,
                    created_at=created_at,
                    row_count=len(rows),
                    snapshot_metadata=dict(metadata or {}),
                )
            )
            session.flush()
            for start in range(0, len(rows), self.INSERT_CHUNK_SIZE):
                session.execute(
                    insert(TrainingSnapshotRow),
                    [
                        {
                            "snapshot_id": snapshot_id,
                            "row_index": start + offset,
                            "payload": row,
                        }
                        for offset, row in enumerate(
                            rows[start : start + self.INSERT_CHUNK_SIZE]
                        )
                    ],
                )
        return SnapshotRecord(
            snapshot_id=snapshot_id,
            content_hash=content_hash,
            created_at=created_at.isoformat(),
            row_count=len(rows),
        )

    # -------------------------------------------------------------------------
    def get_page(self, snapshot_id: str, page: int, page_size: int) -> SnapshotPage:
        if page < 1:
            raise ValueError("page must be >= 1")
        if not 1 <= page_size <= 1000:
            raise ValueError("page_size must be between 1 and 1000")
        with self.database.session_factory() as session:
            snapshot = session.scalar(
                select(TrainingSnapshot).where(
                    TrainingSnapshot.snapshot_id == snapshot_id
                )
            )
            if snapshot is None:
                raise KeyError(snapshot_id)
            offset = (page - 1) * page_size
            rows = session.scalars(
                select(TrainingSnapshotRow.payload)
                .where(TrainingSnapshotRow.snapshot_id == snapshot.snapshot_id)
                .order_by(TrainingSnapshotRow.row_index)
                .offset(offset)
                .limit(page_size)
            ).all()
            return SnapshotPage(
                snapshot_id=snapshot.snapshot_id,
                content_hash=snapshot.content_hash,
                page=page,
                page_size=page_size,
                total_rows=snapshot.row_count,
                rows=tuple(dict(row) for row in rows),
            )


__all__ = ["SnapshotHasher", "SnapshotPage", "SnapshotRecord", "SnapshotStore"]
