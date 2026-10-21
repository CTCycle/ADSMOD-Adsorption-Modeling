# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import argparse
from pathlib import Path

from server.configurations.legacy_import import import_legacy_configuration


def main() -> int:
    parser = argparse.ArgumentParser(description="One-time import of a legacy ADSMOD configuration.")
    parser.add_argument("--legacy-config", required=True, type=Path)
    parser.add_argument("--storage-root", type=Path)
    arguments = parser.parse_args()
    store = import_legacy_configuration(
        arguments.legacy_config,
        storage_root_override=arguments.storage_root,
    )
    store.database.dispose()
    print("Legacy database and application settings were imported into the selected database.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
