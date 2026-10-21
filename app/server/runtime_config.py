# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import argparse
import json

from server.configurations.bootstrap import (
    load_database_configuration,
    load_runtime_settings,
    load_storage_root,
)
from server.repositories.database.manager import resolve_sqlite_path

OUTPUT_MARKER = "ADSMOD_RUNTIME:"


###############################################################################
def build_runtime_record() -> dict[str, object]:
    runtime = load_runtime_settings()
    storage_root = load_storage_root()
    database = load_database_configuration(storage_root=storage_root).config
    return {
        "host": runtime.host,
        "backend_port": runtime.backend_port,
        "frontend_port": runtime.frontend_port,
        "backend_url": runtime.backend_url,
        "frontend_url": runtime.frontend_url,
        "storage_root": str(storage_root),
        "database_backend": "sqlite" if database.embedded_database else "postgres",
        "database_path": (
            str(resolve_sqlite_path(database, storage_root=storage_root))
            if database.embedded_database and database.sqlite_path != ":memory:"
            else None
        ),
    }


###############################################################################
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read ADSMOD runtime settings from environment configuration."
    )
    parser.add_argument(
        "--json", action="store_true", help="Write a machine-readable runtime record."
    )
    args = parser.parse_args()
    record = build_runtime_record()
    if args.json:
        print(OUTPUT_MARKER + json.dumps(record, separators=(",", ":")))
    else:
        print(f"Runtime settings loaded for {record['backend_url']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
