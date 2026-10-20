# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import uvicorn

from .app import create_app
from .configurations.runtime import load_runtime_configuration


def main() -> None:
    runtime = load_runtime_configuration()
    application = create_app(runtime.config, settings_store=runtime.settings_store)
    configuration = runtime.config.runtime
    uvicorn.run(
        application,
        host=configuration.host,
        port=configuration.backend_port,
    )


if __name__ == "__main__":
    main()
