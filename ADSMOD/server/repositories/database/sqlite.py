from __future__ import annotations

import os
from typing import Any

import pandas as pd
import sqlalchemy
from sqlalchemy import inspect
from sqlalchemy import event
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from ADSMOD.server.configurations import DatabaseSettings
from ADSMOD.server.common.constants import RESOURCES_PATH, DATABASE_FILENAME
from ADSMOD.server.common.utils.encoding import sanitize_dataframe_strings
from ADSMOD.server.common.utils.logger import logger
from ADSMOD.server.repositories.database.upsert import resolve_conflict_columns
from ADSMOD.server.repositories.schemas.models import Base


###############################################################################
class SQLiteRepository:
    def __init__(self, settings: DatabaseSettings) -> None:
        self.db_path: str | None = os.path.join(RESOURCES_PATH, DATABASE_FILENAME)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.engine: Engine = sqlalchemy.create_engine(
            f"sqlite:///{self.db_path}", echo=False, future=True
        )
        event.listen(self.engine, "connect", self._enable_foreign_keys)
        self.session_factory = sessionmaker(bind=self.engine, future=True)
        self.insert_batch_size = settings.insert_batch_size
        Base.metadata.create_all(self.engine, checkfirst=True)

    # -------------------------------------------------------------------------
    @staticmethod
    def _enable_foreign_keys(dbapi_connection, connection_record) -> None:  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()

    # -------------------------------------------------------------------------
    def get_table_class(self, table_name: str) -> Any:
        for cls in Base.__subclasses__():
            if getattr(cls, "__tablename__", None) == table_name:
                return cls
        raise ValueError(f"No table class found for name {table_name}")

    # -------------------------------------------------------------------------
    @staticmethod
    def coerce_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
        if dataframe.empty:
            return dataframe
        coerced = dataframe.astype(object)
        return coerced.where(pd.notna(coerced), None)

    # -------------------------------------------------------------------------
    def upsert_dataframe(self, df: pd.DataFrame, table_cls) -> None:
        table = table_cls.__table__
        session = self.session_factory()
        try:
            unique_cols = resolve_conflict_columns(table)
            sanitized_df = sanitize_dataframe_strings(df)
            sanitized_df = self.coerce_missing_values(sanitized_df)
            records = sanitized_df.to_dict(orient="records")
            for i in range(0, len(records), self.insert_batch_size):
                batch = records[i : i + self.insert_batch_size]
                if not batch:
                    continue
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
    def load_from_database(
        self,
        table_name: str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> pd.DataFrame:
        with self.engine.connect() as conn:
            inspector = inspect(conn)
            if not inspector.has_table(table_name):
                logger.warning("Table %s does not exist", table_name)
                return pd.DataFrame()

            query = f'SELECT * FROM "{table_name}"'
            primary_key_columns = []
            try:
                primary_key = inspector.get_pk_constraint(table_name)
                primary_key_columns = primary_key.get("constrained_columns") or []
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to resolve primary key columns for %s: %s",
                    table_name,
                    exc,
                )
            if primary_key_columns:
                ordered_columns = ", ".join(
                    f'"{column}"' for column in primary_key_columns
                )
                query += f" ORDER BY {ordered_columns}"
            if limit is not None:
                query += f" LIMIT {limit}"
            if offset is not None:
                query += f" OFFSET {offset}"

            data = pd.read_sql_query(query, conn)
        return data

    # -------------------------------------------------------------------------
    def save_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        sanitized_df = sanitize_dataframe_strings(df)
        with self.engine.begin() as conn:
            inspector = inspect(conn)
            table_cls = None
            try:
                table_cls = self.get_table_class(table_name)
            except ValueError:
                table_cls = None

            if inspector.has_table(table_name) and table_cls is not None:
                existing_cols = {
                    column["name"] for column in inspector.get_columns(table_name)
                }
                expected_cols = set(table_cls.__table__.columns.keys())
                if existing_cols != expected_cols:
                    table_cls.__table__.drop(conn, checkfirst=True)
                    table_cls.__table__.create(conn, checkfirst=True)
                else:
                    conn.execute(sqlalchemy.text(f'DELETE FROM "{table_name}"'))
            sanitized_df.to_sql(table_name, conn, if_exists="append", index=False)

    # -------------------------------------------------------------------------
    def upsert_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        table_cls = self.get_table_class(table_name)
        self.upsert_dataframe(df, table_cls)

    # -------------------------------------------------------------------------
    def count_rows(self, table_name: str) -> int:
        with self.engine.connect() as conn:
            result = conn.execute(
                sqlalchemy.text(f'SELECT COUNT(*) FROM "{table_name}"')
            )
            value = result.scalar() or 0
        return int(value)
