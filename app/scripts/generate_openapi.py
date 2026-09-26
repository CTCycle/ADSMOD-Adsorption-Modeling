from __future__ import annotations

import argparse
import json
from pathlib import Path

from server.configurations.settings import load_config
from server.app import create_app

###############################################################################
def build_openapi_schema(config_path: Path) -> dict[str, object]:
    """Build the canonical contract from the ML-enabled application surface."""
    application = create_app(load_config(config_path))
    if not application.state.runtime.machine_learning_available:
        raise RuntimeError(
            "Canonical OpenAPI generation requires the ML-enabled backend profile; "
            "the Base profile omits the training and checkpoint routes."
        )
    return application.openapi()


###############################################################################
def main() -> int:
    parser = argparse.ArgumentParser(description="Generate OpenAPI JSON for the unified ADSMOD backend.")
    parser.add_argument("--config", required=True, type=Path, help="Canonical adsmod.json path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    schema = build_openapi_schema(args.config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"OpenAPI written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
