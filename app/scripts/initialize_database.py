# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import time

from server.configurations.runtime import load_runtime_configuration
from server.common.utils.logger import logger


###############################################################################
def main() -> int:
    started_at = time.perf_counter()
    runtime = load_runtime_configuration()
    runtime.settings_store.database.dispose()
    logger.info(
        "Database initialization completed in %.2f seconds (backend=%s).",
        time.perf_counter() - started_at,
        "sqlite" if runtime.config.application.database.embedded_database else "postgres",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
