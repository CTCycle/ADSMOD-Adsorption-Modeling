from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path


def load_asgi_app(spec: str):
    module_name, _, app_name = spec.partition(":")
    if not module_name or not app_name:
        raise ValueError("Expected --app in format '<module>:<attribute>'")
    module = importlib.import_module(module_name)
    app = getattr(module, app_name, None)
    if app is None:
        raise ValueError(f"App attribute '{app_name}' not found in module '{module_name}'")
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate OpenAPI JSON for a FastAPI app.")
    parser.add_argument("--app", required=True, help="ASGI app path, e.g. core_service.app:app")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    app = load_asgi_app(args.app)
    schema = app.openapi()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"OpenAPI written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
