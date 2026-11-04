# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "client" / "dist" / "browser"
RESERVED_PREFIXES = ("api/", "health/", "docs", "redoc", "openapi.json")


###############################################################################
class FrontendBundle:
    """Static handlers for the built Angular bundle rooted at ``root``."""

    # -------------------------------------------------------------------------
    def __init__(self, root: Path) -> None:
        self.root = root
        self.index_path = root / "index.html"

    # -------------------------------------------------------------------------
    def serve(self, path: str) -> Response:
        if path.startswith(RESERVED_PREFIXES):
            raise HTTPException(status_code=404)
        candidate = (self.root / path).resolve()
        if path and candidate.is_relative_to(self.root) and candidate.is_file():
            immutable = "public, max-age=31536000, immutable"
            return FileResponse(candidate, headers={"Cache-Control": immutable})
        if Path(path).suffix:
            raise HTTPException(status_code=404)
        return FileResponse(self.index_path, headers={"Cache-Control": "no-cache"})

    # -------------------------------------------------------------------------
    def serve_index(self) -> Response:
        return self.serve("")

    # -------------------------------------------------------------------------
    @staticmethod
    def not_found(path: str) -> Response:
        raise HTTPException(status_code=404)


###############################################################################
def register_frontend_routes(app: FastAPI, dist_root: Path = FRONTEND_DIST) -> bool:
    """Serve the built Angular bundle from the API origin when it exists."""

    bundle = FrontendBundle(dist_root.resolve())
    if not bundle.index_path.is_file():
        return False

    app.add_api_route("/", bundle.serve_index, methods=["GET"], include_in_schema=False)
    app.add_api_route("/{path:path}", bundle.serve, methods=["GET"], include_in_schema=False)
    app.add_api_route(
        "/{path:path}",
        bundle.not_found,
        methods=["POST", "PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    return True


__all__ = ["FRONTEND_DIST", "register_frontend_routes"]
