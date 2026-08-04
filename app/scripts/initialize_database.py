from __future__ import annotations

import time

from shared.common.settings import get_server_settings
from shared.repositories.database.initializer import initialize_database
from shared.common.utils.logger import logger


###############################################################################
if __name__ == "__main__":
    start = time.perf_counter()
    settings = get_server_settings().database
    mode = "SQLite" if settings.embedded_database else "PostgreSQL"
    logger.info("Starting explicit %s database initialization.", mode)
    initialize_database(settings)
    elapsed = time.perf_counter() - start
    logger.info("Database initialization completed in %.2f seconds.", elapsed)

