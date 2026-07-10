from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .app import create_app_from_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adsmod-core")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", default=None, type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    application = create_app_from_path(args.config)
    runtime = application.state.config.runtime
    uvicorn.run(
        application,
        host=args.host or runtime.host,
        port=args.port or runtime.core_port,
    )


if __name__ == "__main__":
    main()