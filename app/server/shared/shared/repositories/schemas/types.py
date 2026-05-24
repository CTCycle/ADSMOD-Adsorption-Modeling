from __future__ import annotations

from typing import Any

from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON, TypeDecorator


###############################################################################
class JSONSequence(TypeDecorator):
    """
    SQLAlchemy type that stores lists as JSON.
    Uses JSONB for PostgreSQL to allow indexing, and standard JSON for SQLite.
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
            return normalize_sequence_values(value)
        if isinstance(value, str):
            raise ValueError(
                "Invalid JSONSequence payload: expected JSON list, received string."
            )
        return value


###############################################################################
def normalize_sequence_values(values: list[Any]) -> list[Any]:
    normalized: list[Any] = []
    for item in values:
        if isinstance(item, str):
            trimmed = item.strip()
            if trimmed == "":
                continue
            try:
                normalized.append(float(trimmed))
                continue
            except ValueError:
                normalized.append(trimmed)
                continue
        normalized.append(item)
    return normalized
