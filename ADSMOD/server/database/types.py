from __future__ import annotations

import json
from typing import Any

from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON, TypeDecorator


###############################################################################
class JSONSequence(TypeDecorator):
    """
    SQLAlchemy type that stores lists as JSON.
    Uses JSONB for PostgreSQL to allow indexing, and standard JSON for SQLite.
    Includes backward compatibility for reading legacy CSV strings.
    """

    impl = JSON
    cache_ok = True

    # -------------------------------------------------------------------------
    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB)
        return dialect.type_descriptor(JSON)

    # -------------------------------------------------------------------------
    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        # Ensure we are storing a list/dict, or let SQLAlchemy JSON fail if invalid
        return value

    # -------------------------------------------------------------------------
    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            trimmed = value.strip()
            if not trimmed:
                return []
            try:
                # Try parsing as JSON first
                return json.loads(trimmed)
            except json.JSONDecodeError:
                # Fallback: Treat as CSV string (Legacy support)
                return [
                    x.strip() for x in trimmed.split(",") if x.strip()
                ]
        return value
