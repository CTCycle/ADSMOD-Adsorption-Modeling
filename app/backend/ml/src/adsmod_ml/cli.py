from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from adsmod_common.config import load_config

from .app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ADSMOD ML service.")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the canonical adsmod.json configuration.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if config.runtime.mode != "core-ml":
        raise SystemExit("The ML service requires runtime.mode=core-ml.")
    uvicorn.run(
        create_app(config),
        host=config.runtime.host,
        port=config.runtime.ml_port,
    )


if __name__ == "__main__":
    main()
