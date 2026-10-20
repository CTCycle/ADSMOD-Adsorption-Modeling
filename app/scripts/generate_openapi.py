# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import argparse
import json
from pathlib import Path

from server.configurations.bootstrap import load_database_config, load_storage_root
from server.configurations.settings import AdsmodConfig, UserSettings
from server.app import create_app

###############################################################################
def build_openapi_schema() -> dict[str, object]:
    """Build the canonical contract from the ML-enabled application surface."""
    config = AdsmodConfig.assemble(
        database=load_database_config(),
        storage_root=load_storage_root(),
        settings=UserSettings(),
    )
    application = create_app(config)
    if not application.state.runtime.machine_learning_available:
        raise RuntimeError(
            "Canonical OpenAPI generation requires the ML-enabled backend profile; "
            "the Base profile omits the training and checkpoint routes."
        )
    return application.openapi()

###############################################################################
def main() -> int:
    parser = argparse.ArgumentParser(description="Generate OpenAPI JSON for the unified ADSMOD backend.")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    schema = build_openapi_schema()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"OpenAPI written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
