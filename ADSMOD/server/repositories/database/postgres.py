from __future__ import annotations

import json
import urllib.parse
from typing import Any

import pandas as pd
import sqlalchemy
from sqlalchemy import event
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from ADSMOD.server.configurations import DatabaseSettings
from ADSMOD.server.repositories.database.sql import (
    build_postgres_server_encoding_sql,
    postgres_set_client_encoding_sql,
)
from ADSMOD.server.repositories.database.utils import normalize_postgres_engine
from ADSMOD.server.repositories.database.upsert import resolve_conflict_columns
from ADSMOD.server.repositories.schemas.models import Base
from ADSMOD.server.repositories.schemas.types import JSONSequence
from ADSMOD.server.common.utils.encoding import sanitize_dataframe_strings
from ADSMOD.server.common.utils.security import ensure_safe_sql_identifier
from ADSMOD.server.common.utils.logger import logger


###############################################################################
class PostgresRepository:
    MAX_STATEMENT_PARAMETERS = 65535

    def __init__(
        self,
        settings: DatabaseSettings,
        initialize_schema: bool = False,
    ) -> None:
        if not settings.host:
            raise ValueError("Database host must be provided for external database.")
        if not settings.database_name:
            raise ValueError("Database name must be provided for external database.")
        if not settings.username:
            raise ValueError(
                "Database username must be provided for external database."
            )

        port = settings.port or 5432
        engine_name = normalize_postgres_engine(settings.engine)
        password = settings.password or ""
        connect_args = self.build_connect_args(settings)

        safe_username = urllib.parse.quote_plus(settings.username)
        safe_password = urllib.parse.quote_plus(password)
        self.db_path: str | None = None
        self.engine: Engine = sqlalchemy.create_engine(
            f"{engine_name}://{safe_username}:{safe_password}@{settings.host}:{port}/{settings.database_name}",
            echo=False,
            future=True,
            connect_args=connect_args,
            pool_pre_ping=True,
        )
        event.listen(self.engine, "connect", self._ensure_utf8_client_encoding)
        self.session = sessionmaker(bind=self.engine, future=True)
        self.insert_batch_size = settings.insert_batch_size
        self._ensure_server_utf8()
        if initialize_schema:
            Base.metadata.create_all(self.engine, checkfirst=True)
            logger.info(
                "Initialized PostgreSQL schema for database %s",
                settings.database_name,
            )

    # -------------------------------------------------------------------------
    @staticmethod
    def build_connect_args(settings: DatabaseSettings) -> dict[str, Any]:
        connect_args: dict[str, Any] = {
            "connect_timeout": settings.connect_timeout,
            "client_encoding": "utf8",
        }
        if settings.ssl:
            connect_args["sslmode"] = "require"
            if settings.ssl_ca:
                connect_args["sslrootcert"] = settings.ssl_ca
        return connect_args

    # -------------------------------------------------------------------------
    @staticmethod
    def _ensure_utf8_client_encoding(
        dbapi_connection: Any, connection_record: Any
    ) -> None:
        try:
            with dbapi_connection.cursor() as cursor:
                cursor.execute(postgres_set_client_encoding_sql())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to enforce UTF-8 client encoding: %s", exc)

    # -------------------------------------------------------------------------
    def _ensure_server_utf8(self) -> None:
        try:
            with self.engine.connect() as conn:
                encoding = conn.execute(build_postgres_server_encoding_sql()).scalar()
            normalized = str(encoding or "").upper()
            if normalized not in {"UTF8", "UTF-8"}:
                raise ValueError(
                    f"Postgres server encoding must be UTF8. Found: {encoding!r}. "
                    "Migrate the database to UTF8 before starting ADSMOD."
                )
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, ValueError):
                raise
            raise RuntimeError(
                "Failed to verify Postgres server encoding. "
                "Ensure the database is reachable and UTF8-configured."
            ) from exc

    # -------------------------------------------------------------------------
    def get_table_class(self, table_name: str) -> Any:
        for cls in Base.__subclasses__():
            if getattr(cls, "__tablename__", None) == table_name:
                return cls
        raise ValueError(f"No table class found for name {table_name}")

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
        return sanitize_dataframe_strings(df)

    # -------------------------------------------------------------------------
    @staticmethod
    def parse_json_column_value(value: Any) -> Any:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            trimmed = value.strip()
            if not trimmed:
                return None
            try:
                return json.loads(trimmed)
            except json.JSONDecodeError:
                return value
        return value

    # -------------------------------------------------------------------------
    @staticmethod
    def coerce_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
        if dataframe.empty:
            return dataframe
        coerced = dataframe.astype(object)
        return coerced.where(pd.notna(coerced), None)

    # -------------------------------------------------------------------------
    def prepare_for_storage(
        self,
        df: pd.DataFrame,
        table_cls: Any | None = None,
    ) -> pd.DataFrame:
        prepared = self.normalize_string_columns(df)
        if table_cls is None or prepared.empty:
            return prepared

        json_columns = [
            column.name
            for column in table_cls.__table__.columns
            if isinstance(column.type, JSONSequence)
        ]
        for column in json_columns:
            if column in prepared.columns:
                prepared[column] = prepared[column].apply(self.parse_json_column_value)
        return self.coerce_missing_values(prepared)

    # -------------------------------------------------------------------------
    def restore_after_load(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    # -------------------------------------------------------------------------
    @staticmethod
    def build_conflict_key(
        record: dict[str, Any], conflict_columns: list[str]
    ) -> tuple[Any, ...] | None:
        key_values: list[Any] = []
        for column in conflict_columns:
            value = record.get(column)
            if value is None:
                return None
            if isinstance(value, float) and pd.isna(value):
                return None
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True, default=str)
            key_values.append(value)
        return tuple(key_values)

    # -------------------------------------------------------------------------
    @staticmethod
    def deduplicate_conflict_batch(
        batch: list[dict[str, Any]],
        conflict_columns: list[str],
    ) -> tuple[list[dict[str, Any]], int]:
        deduplicated: list[dict[str, Any]] = []
        key_to_index: dict[tuple[Any, ...], int] = {}
        dropped = 0

        for record in batch:
            conflict_key = PostgresRepository.build_conflict_key(
                record, conflict_columns
            )
            if conflict_key is None:
                deduplicated.append(record)
                continue
            existing_index = key_to_index.get(conflict_key)
            if existing_index is None:
                key_to_index[conflict_key] = len(deduplicated)
                deduplicated.append(record)
                continue
            deduplicated[existing_index] = record
            dropped += 1

        return deduplicated, dropped

    # -------------------------------------------------------------------------
    def upsert_dataframe(self, df: pd.DataFrame, table_cls) -> None:
        table = table_cls.__table__
        session = self.session()
        try:
            unique_cols = resolve_conflict_columns(table)
            prepared_df = self.prepare_for_storage(df, table_cls)
            records = prepared_df.to_dict(orient="records")
            if not records:
                return

            columns_per_row = max(1, len(records[0]))
            max_rows_per_statement = max(
                1, self.MAX_STATEMENT_PARAMETERS // columns_per_row
            )
            effective_batch_size = min(self.insert_batch_size, max_rows_per_statement)
            if effective_batch_size < self.insert_batch_size:
                logger.info(
                    "Reducing Postgres upsert batch size from %s to %s for %s to avoid parameter limit.",
                    self.insert_batch_size,
                    effective_batch_size,
                    table.name,
                )

            for i in range(0, len(records), effective_batch_size):
                batch = records[i : i + effective_batch_size]
                if not batch:
                    continue
                batch, dropped = self.deduplicate_conflict_batch(batch, unique_cols)
                if dropped > 0:
                    logger.warning(
                        "Dropped %d duplicate rows in upsert batch for %s on conflict columns %s.",
                        dropped,
                        table.name,
                        unique_cols,
                    )
                stmt = insert(table).values(batch)
                update_cols = {
                    col: getattr(stmt.excluded, col)  # type: ignore[attr-defined]
                    for col in batch[0]
                    if col not in unique_cols
                }
                stmt = stmt.on_conflict_do_update(
                    index_elements=unique_cols, set_=update_cols
                )
                session.execute(stmt)
                session.commit()
        finally:
            session.close()

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_pagination_value(
        value: int | None, field_name: str
    ) -> int | None:
        if value is None:
            return None
        try:
            candidate = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {field_name} value.") from exc
        if candidate < 0:
            raise ValueError(f"{field_name} must be >= 0.")
        return candidate

    # -------------------------------------------------------------------------
    def load_from_database(
        self,
        table_name: str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> pd.DataFrame:
        table_name = ensure_safe_sql_identifier(table_name, "table name")
        safe_limit = self.normalize_pagination_value(limit, "limit")
        safe_offset = self.normalize_pagination_value(offset, "offset")
        try:
            table_cls = self.get_table_class(table_name)
        except ValueError:
            logger.warning("Table %s does not map to an ORM model.", table_name)
            return pd.DataFrame()

        statement = select(table_cls)
        primary_key_columns = [column.name for column in table_cls.__table__.primary_key]
        if primary_key_columns:
            statement = statement.order_by(
                *(getattr(table_cls, column) for column in primary_key_columns)
            )
        if safe_offset is not None:
            statement = statement.offset(safe_offset)
        if safe_limit is not None:
            statement = statement.limit(safe_limit)

        column_names = [column.name for column in table_cls.__table__.columns]
        with self.session() as session:
            rows = session.execute(statement).scalars().all()
        data = pd.DataFrame.from_records(
            [{column: getattr(row, column) for column in column_names} for row in rows],
            columns=column_names,
        )
        return self.restore_after_load(data)

    # -------------------------------------------------------------------------
    def upsert_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        table_name = ensure_safe_sql_identifier(table_name, "table name")
        table_cls = self.get_table_class(table_name)
        self.upsert_dataframe(df, table_cls)

    # -------------------------------------------------------------------------
    def count_rows(self, table_name: str) -> int:
        table_name = ensure_safe_sql_identifier(table_name, "table name")
        try:
            table_cls = self.get_table_class(table_name)
        except ValueError:
            logger.warning("Table %s does not map to an ORM model.", table_name)
            return 0

        with self.session() as session:
            statement = select(func.count()).select_from(table_cls)
            value = session.execute(statement).scalar_one()
        return int(value)
