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
    result = initialize_database(settings)
    elapsed = time.perf_counter() - start
    logger.info(
        "Database initialization completed in %.2f seconds (before=%s, after=%s, "
        "head=%s, adopted_legacy=%s, applied_migrations=%s).",
        elapsed,
        ",".join(result.before) or "base",
        ",".join(result.after) or "base",
        result.head,
        result.adopted_legacy_schema,
        result.applied_migrations,
    )

