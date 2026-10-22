# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "client" / "dist" / "browser"
RESERVED_PREFIXES = ("api/", "health/", "docs", "redoc", "openapi.json")


###############################################################################
def register_frontend_routes(app: FastAPI, dist_root: Path = FRONTEND_DIST) -> bool:
    """Serve the built Angular bundle from the API origin when it exists."""

    root = dist_root.resolve()
    index_path = root / "index.html"
    if not index_path.is_file():
        return False

    def serve(path: str) -> Response:
        if path.startswith(RESERVED_PREFIXES):
            raise HTTPException(status_code=404)
        candidate = (root / path).resolve()
        if path and candidate.is_relative_to(root) and candidate.is_file():
            immutable = "public, max-age=31536000, immutable"
            return FileResponse(candidate, headers={"Cache-Control": immutable})
        if Path(path).suffix:
            raise HTTPException(status_code=404)
        return FileResponse(index_path, headers={"Cache-Control": "no-cache"})

    app.add_api_route("/", lambda: serve(""), methods=["GET"], include_in_schema=False)
    app.add_api_route("/{path:path}", serve, methods=["GET"], include_in_schema=False)
    return True


__all__ = ["FRONTEND_DIST", "register_frontend_routes"]
