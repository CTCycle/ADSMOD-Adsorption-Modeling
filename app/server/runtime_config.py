# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import argparse
import json

from server.configurations.runtime import load_runtime_configuration
from server.repositories.database.manager import resolve_sqlite_path

OUTPUT_MARKER = "ADSMOD_RUNTIME:"


def build_runtime_record() -> dict[str, object]:
    runtime = load_runtime_configuration(initialize_database=False)
    database = runtime.config.application.database
    storage_root = runtime.config.storage.root
    try:
        return {
            "host": runtime.config.runtime.host,
            "backend_port": runtime.config.runtime.backend_port,
            "frontend_port": runtime.config.runtime.frontend_port,
            "storage_root": str(storage_root),
            "database_backend": "sqlite" if database.embedded_database else "postgres",
            "database_path": (
                str(resolve_sqlite_path(database, storage_root=storage_root))
                if database.embedded_database and database.sqlite_path != ":memory:"
                else None
            ),
        }
    finally:
        runtime.settings_store.database.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description="Read ADSMOD runtime settings from the active database.")
    parser.add_argument("--json", action="store_true", help="Write a machine-readable runtime record.")
    args = parser.parse_args()
    record = build_runtime_record()
    if args.json:
        print(OUTPUT_MARKER + json.dumps(record, separators=(",", ":")))
    else:
        print(f"Runtime settings loaded for {record['host']}:{record['backend_port']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
