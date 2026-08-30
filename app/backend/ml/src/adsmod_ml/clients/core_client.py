from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any

import httpx

from adsmod_common.config import AdsmodConfig


###############################################################################
class SnapshotClientError(RuntimeError):
    """Raised when core snapshot retrieval or verification fails."""


###############################################################################
@dataclass(frozen=True)
class SnapshotPayload:
    snapshot_id: str
    content_hash: str
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class SnapshotReference:
    snapshot_id: str
    content_hash: str
    row_count: int


###############################################################################
class CoreSnapshotClient:
    # -------------------------------------------------------------------------
    def __init__(
        self,
        base_url: str,
        internal_token: str,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.internal_token = internal_token
        self.transport = transport

    @classmethod
    def from_config(
        cls,
        config: AdsmodConfig,
        *,
        internal_token: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> "CoreSnapshotClient":
        token = internal_token
        if token is None:
            token = os.environ.get(config.security.internal_token_env, "")
        return cls(
            f"http://{config.runtime.host}:{config.runtime.core_port}",
            token,
            transport,
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        headers = dict(kwargs.pop("headers", {}))
        headers["X-ADSMOD-Internal-Token"] = self.internal_token
        with httpx.Client(
            base_url=self.base_url,
            headers=headers,
            transport=self.transport,
            timeout=15.0,
        ) as client:
            response = client.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise SnapshotClientError(
                f"core snapshot request failed with HTTP {response.status_code}"
            )
        return response

    def list_sources(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/api/v1/internal/training/sources").json()
        sources = payload.get("datasets")
        if not isinstance(sources, list):
            raise SnapshotClientError("core response omitted training sources")
        return [dict(item) for item in sources if isinstance(item, dict)]

    def create_snapshot_from_selections(
        self,
        selections: list[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> SnapshotReference:
        response = self._request(
            "POST",
            "/api/v1/internal/training/snapshots",
            json={"selections": selections, "metadata": metadata or {}},
        )
        payload = response.json()
        try:
            return SnapshotReference(
                snapshot_id=str(payload["snapshot_id"]),
                content_hash=str(payload["content_hash"]),
                row_count=int(payload["row_count"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SnapshotClientError(
                "core response omitted snapshot metadata"
            ) from exc

    def create_snapshot(
        self,
        rows: list[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> SnapshotReference:
        response = self._request(
            "POST",
            "/api/v1/internal/snapshots",
            json={"rows": rows, "metadata": metadata or {}},
        )
        payload = response.json()
        try:
            return SnapshotReference(
                snapshot_id=str(payload["snapshot_id"]),
                content_hash=str(payload["content_hash"]),
                row_count=int(payload["row_count"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SnapshotClientError(
                "core response omitted snapshot metadata"
            ) from exc

    # -------------------------------------------------------------------------
    def fetch_snapshot(
        self, snapshot_id: str, *, page_size: int = 1000
    ) -> SnapshotPayload:
        rows: list[dict[str, Any]] = []
        page = 1
        content_hash: str | None = None
        while True:
            response = self._request(
                "GET",
                f"/api/v1/internal/snapshots/{snapshot_id}",
                params={"page": page, "page_size": page_size},
            )
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
        canonical = json.dumps(
            rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        calculated_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if content_hash != calculated_hash:
            raise SnapshotClientError("snapshot hash verification failed")
        return SnapshotPayload(
            snapshot_id=snapshot_id, content_hash=calculated_hash, rows=tuple(rows)
        )
