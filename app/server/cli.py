# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import sys

import uvicorn

from .app import create_app
from .common.utils.logger import uvicorn_logging_config
from .configurations.runtime import load_runtime_configuration

# Fitting runs in a CPU-bound worker thread, and sqlite3 releases the GIL for
# every fetched row. At the default 5 ms switch interval a request reading
# thousands of rows during a fit waits up to 5 ms per row (seconds to a minute
# in total). A 0.1 ms interval keeps such reads responsive; it only applies
# when another thread is waiting, so the fit itself is not measurably slower.
GIL_SWITCH_INTERVAL_SECONDS = 1e-4


###############################################################################
def main() -> None:
    sys.setswitchinterval(GIL_SWITCH_INTERVAL_SECONDS)
    runtime = load_runtime_configuration()
    application = create_app(runtime.config, settings_store=runtime.settings_store)
    configuration = runtime.config.runtime
    server = uvicorn.Server(
        uvicorn.Config(
            application,
            host=configuration.host,
            port=configuration.backend_port,
            log_config=uvicorn_logging_config(),
        )
    )

    def request_shutdown() -> None:
        # Same graceful path as Ctrl+C: the lifespan stops jobs and the database.
        server.should_exit = True

    application.state.request_shutdown = request_shutdown
    server.run()


if __name__ == "__main__":
    main()
