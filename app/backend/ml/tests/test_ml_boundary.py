from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from adsmod_common.config import load_config
from adsmod_ml.app import create_app
from adsmod_ml.clients.core_client import CoreSnapshotClient, SnapshotClientError


CONFIG_PATH = Path("app/resources/adsmod.json")

###############################################################################
def test_ml_runtime_contracts() -> None:
    config = load_config(CONFIG_PATH).model_copy(update={"runtime": load_config(CONFIG_PATH).runtime.model_copy(update={"mode": "core-ml", "ml_restart_attempts": 1})})
    with TestClient(create_app(config)) as client:
        assert client.get("/health/live").json()["service"] == "ml"
        assert client.get("/health/ready").json()["state"] == "ready"
        assert client.get("/api/v1/system/capabilities").json()["features"]["training"] is True
        configuration = client.get("/api/v1/system/configuration")
        assert configuration.status_code == 200
        payload = configuration.json()
        assert payload["defaults"]["batch_size"] == 32
        assert payload["dataset_defaults"]["max_measurements"] == 30
        assert payload["runtime"]["keras_backend"] == "torch"

###############################################################################
def test_snapshot_client_fetches_pages_and_verifies_hash() -> None:
    pages = {
        1: {"content_hash": "", "page": 1, "page_size": 1, "total_rows": 2, "rows": [{"id": 1}]},
        2: {"content_hash": "", "page": 2, "page_size": 1, "total_rows": 2, "rows": [{"id": 2}]},
    }
    import hashlib
    import json

    content_hash = hashlib.sha256(json.dumps([{"id": 1}, {"id": 2}], separators=(",", ":"), sort_keys=True).encode()).hexdigest()
    for payload in pages.values():
        payload["content_hash"] = content_hash

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        assert request.headers["x-adsmod-internal-token"] == "secret"
        return httpx.Response(200, json=pages[page])

    result = CoreSnapshotClient("http://core", "secret", httpx.MockTransport(handler)).fetch_snapshot("snapshot")
    assert result.rows == ({"id": 1}, {"id": 2})

###############################################################################
def test_snapshot_client_rejects_hash_mismatch() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"content_hash": "bad", "total_rows": 1, "rows": [{"id": 1}]}))
    try:
        CoreSnapshotClient("http://core", "secret", transport).fetch_snapshot("snapshot")
    except SnapshotClientError as exc:
        assert "hash verification" in str(exc)
    else:
        raise AssertionError("hash mismatch was accepted")
