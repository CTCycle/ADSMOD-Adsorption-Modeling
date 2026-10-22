# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.api.frontend import register_frontend_routes


###############################################################################
def build_client(dist: Path) -> tuple[TestClient, bool]:
    app = FastAPI()
    app.add_api_route("/api/v1/ping", lambda: {"ok": True}, methods=["GET"])
    registered = register_frontend_routes(app, dist)
    return TestClient(app), registered


###############################################################################
def test_frontend_bundle_is_served_on_api_origin(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<app-root></app-root>", encoding="utf-8")
    (tmp_path / "main.js").write_text("console.log(1)", encoding="utf-8")
    client, registered = build_client(tmp_path)

    assert registered
    assert client.get("/api/v1/ping").json() == {"ok": True}
    assert client.get("/main.js").text == "console.log(1)"
    assert client.get("/").text == "<app-root></app-root>"
    assert client.get("/datasets/42").text == "<app-root></app-root>"
    assert client.get("/missing.js").status_code == 404
    assert client.get("/api/v1/unknown").status_code == 404
    assert client.get("/health/unknown").status_code == 404


###############################################################################
def test_frontend_routes_are_skipped_without_bundle(tmp_path: Path) -> None:
    _, registered = build_client(tmp_path)

    assert not registered
