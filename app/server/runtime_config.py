# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import argparse
import json

from server.configurations.bootstrap import load_database_config, load_storage_root, resolve_settings_database_path
from server.configurations.runtime import load_runtime_configuration

OUTPUT_MARKER = "ADSMOD_RUNTIME:"


def build_runtime_record() -> dict[str, object]:
    database = load_database_config()
    storage_root = load_storage_root()
    runtime = load_runtime_configuration(
        database=database,
        storage_root=storage_root,
        initialize_database=False,
    )
    try:
        return {
            "host": runtime.config.runtime.host,
            "backend_port": runtime.config.runtime.backend_port,
            "frontend_port": runtime.config.runtime.frontend_port,
            "storage_root": str(storage_root),
            "settings_database_path": str(resolve_settings_database_path(database, storage_root)),
            "embedded_database": database.embedded_database,
        }
    finally:
        runtime.settings_store.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description="Read ADSMOD runtime bootstrap values from SQLite settings.")
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
