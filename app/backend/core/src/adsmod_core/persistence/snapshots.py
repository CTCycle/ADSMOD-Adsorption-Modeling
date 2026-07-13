from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

###############################################################################
@dataclass(frozen=True)
class SnapshotRecord:
    snapshot_id: str
    content_hash: str
    created_at: str
    row_count: int
    rows: tuple[dict[str, Any], ...]

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
class SnapshotStore:
    """Core-owned immutable snapshot storage for ML consumption."""

    # -------------------------------------------------------------------------
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS training_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )

    # -------------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    # -------------------------------------------------------------------------
    @staticmethod
    def _canonical_rows(rows: list[dict[str, Any]]) -> tuple[str, tuple[dict[str, Any], ...]]:
        frozen_rows = tuple(dict(row) for row in rows)
        payload = json.dumps(frozen_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return payload, frozen_rows

    # -------------------------------------------------------------------------
    def create(self, rows: list[dict[str, Any]]) -> SnapshotRecord:
        payload, frozen_rows = self._canonical_rows(rows)
        content_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        snapshot_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO training_snapshots(snapshot_id, content_hash, created_at, payload) VALUES (?, ?, ?, ?)",
                (snapshot_id, content_hash, created_at, payload),
            )
        return SnapshotRecord(snapshot_id, content_hash, created_at, len(frozen_rows), frozen_rows)

    # -------------------------------------------------------------------------
    def get_page(self, snapshot_id: str, page: int, page_size: int) -> SnapshotPage:
        if page < 1:
            raise ValueError("page must be >= 1")
        if not 1 <= page_size <= 1000:
            raise ValueError("page_size must be between 1 and 1000")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT content_hash, payload FROM training_snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
        if row is None:
            raise KeyError(snapshot_id)
        rows = tuple(json.loads(row["payload"]))
        offset = (page - 1) * page_size
        return SnapshotPage(
            snapshot_id=snapshot_id,
            content_hash=row["content_hash"],
            page=page,
            page_size=page_size,
            total_rows=len(rows),
            rows=rows[offset : offset + page_size],
        )