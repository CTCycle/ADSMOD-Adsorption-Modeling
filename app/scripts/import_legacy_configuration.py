# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import argparse
from pathlib import Path

from server.configurations.bootstrap import DOTENV_PATH
from server.configurations.legacy_import import import_legacy_configuration


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-time import of ADSMOD settings from a legacy adsmod.json file."
    )
    parser.add_argument("--legacy-config", required=True, type=Path)
    parser.add_argument("--storage-root", type=Path)
    parser.add_argument("--dotenv", type=Path, default=DOTENV_PATH)
    arguments = parser.parse_args()
    store, _ = import_legacy_configuration(
        arguments.legacy_config,
        storage_root_override=arguments.storage_root,
        dotenv_path=arguments.dotenv,
    )
    store.dispose()
    print("Legacy settings imported into SQLite. Database bootstrap values were written only where unset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
