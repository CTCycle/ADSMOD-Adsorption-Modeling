from __future__ import annotations

import sqlalchemy


# -----------------------------------------------------------------------------
def build_postgres_create_database_sql(
    database_name: str,
) -> sqlalchemy.sql.elements.TextClause:
    safe_database = database_name.replace('"', '""')
    return sqlalchemy.text(
        f"CREATE DATABASE \"{safe_database}\" WITH ENCODING 'UTF8' TEMPLATE template0"
    )


# -----------------------------------------------------------------------------
def build_postgres_database_exists_sql() -> sqlalchemy.sql.elements.TextClause:
    return sqlalchemy.text("SELECT 1 FROM pg_database WHERE datname=:name")


# -----------------------------------------------------------------------------
def build_postgres_server_encoding_sql() -> sqlalchemy.sql.elements.TextClause:
    return sqlalchemy.text("SHOW SERVER_ENCODING")


# -----------------------------------------------------------------------------
def postgres_set_client_encoding_sql() -> str:
    return "SET client_encoding TO 'UTF8'"


# -----------------------------------------------------------------------------
def sqlite_enable_foreign_keys_sql() -> str:
    return "PRAGMA foreign_keys=ON"


# -----------------------------------------------------------------------------
def build_table_select_sql(
    table_name: str,
    primary_key_columns: list[str],
    limit: int | None = None,
    offset: int | None = None,
) -> tuple[sqlalchemy.sql.elements.TextClause, dict[str, int]]:
    query = f'SELECT * FROM "{table_name}"'
    if primary_key_columns:
        ordered_columns = ", ".join(f'"{column}"' for column in primary_key_columns)
        query += f" ORDER BY {ordered_columns}"

    query_params: dict[str, int] = {}
    if limit is not None:
        query += " LIMIT :limit"
        query_params["limit"] = limit
    if offset is not None:
        query += " OFFSET :offset"
        query_params["offset"] = offset

    return sqlalchemy.text(query), query_params


# -----------------------------------------------------------------------------
def build_table_count_sql(table_name: str) -> sqlalchemy.sql.elements.TextClause:
    return sqlalchemy.text(f'SELECT COUNT(*) FROM "{table_name}"')
