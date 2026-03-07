from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("FASTAPI_HOST", "127.0.0.1")
    port_raw = os.getenv("FASTAPI_PORT", "8000")
    try:
        port = int(port_raw)
    except ValueError:
        raise RuntimeError(f"Invalid FASTAPI_PORT: {port_raw}")

    uvicorn.run(
        "ADSMOD.server.app:app",
        host=host,
        port=port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
