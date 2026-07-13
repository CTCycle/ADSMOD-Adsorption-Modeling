from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import httpx

###############################################################################
class SnapshotClientError(RuntimeError):
    """Raised when core snapshot retrieval or verification fails."""

###############################################################################
@dataclass(frozen=True)
class SnapshotPayload:
    snapshot_id: str
    content_hash: str
    rows: tuple[dict[str, Any], ...]

###############################################################################
class CoreSnapshotClient:

    # -------------------------------------------------------------------------
    def __init__(self, base_url: str, internal_token: str, transport: httpx.BaseTransport | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.internal_token = internal_token
        self.transport = transport

    # -------------------------------------------------------------------------
    def fetch_snapshot(self, snapshot_id: str, *, page_size: int = 1000) -> SnapshotPayload:
        headers = {"X-ADSMOD-Internal-Token": self.internal_token}
        rows: list[dict[str, Any]] = []
        page = 1
        content_hash: str | None = None
        with httpx.Client(base_url=self.base_url, headers=headers, transport=self.transport, timeout=15.0) as client:
            while True:
                response = client.get(
                    f"/api/v1/internal/snapshots/{snapshot_id}",
                    params={"page": page, "page_size": page_size},
                )
                if response.status_code != 200:
                    raise SnapshotClientError(f"core snapshot request failed with HTTP {response.status_code}")
                payload = response.json()
                current_hash = payload.get("content_hash")
                if not isinstance(current_hash, str):
                    raise SnapshotClientError("core response omitted content_hash")
                if content_hash is None:
                    content_hash = current_hash
                elif content_hash != current_hash:
                    raise SnapshotClientError("core returned inconsistent snapshot hashes")
                page_rows = payload.get("rows")
                if not isinstance(page_rows, list):
                    raise SnapshotClientError("core response omitted rows")
                rows.extend(page_rows)
                if len(rows) >= int(payload.get("total_rows", len(rows))) or not page_rows:
                    break
                page += 1
        canonical = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        calculated_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if content_hash != calculated_hash:
            raise SnapshotClientError("snapshot hash verification failed")
        return SnapshotPayload(snapshot_id=snapshot_id, content_hash=calculated_hash, rows=tuple(rows))