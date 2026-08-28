from __future__ import annotations

from alembic import context
from sqlalchemy.engine import Connection

from shared.common.settings import get_server_settings
from shared.repositories.database.manager import DatabaseManager
from shared.repositories.schemas.models import Base


config = context.config
target_metadata = Base.metadata

###############################################################################
def include_object(
    object_: object,
    name: str | None,
    type_: str,
    reflected: bool,
    compare_to: object | None,
) -> bool:
    """Keep autogenerate scoped to tables owned by this application."""

    if type_ == "table" and reflected and name not in target_metadata.tables:
        return False
    return True

###############################################################################
def _configure(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_object=include_object,
        compare_type=True,
        compare_server_default=True,
        render_as_batch=connection.dialect.name == "sqlite",
        user_module_prefix="schema_types.",
        # The coordinator owns the real transaction. This also makes the
        # command-line path explicit and consistent across SQLite/PostgreSQL.
        transactional_ddl=False,
        transaction_per_migration=False,
    )

###############################################################################
def run_migrations_offline() -> None:
    settings = get_server_settings().database
    manager = DatabaseManager(settings)
    try:
        context.configure(
            url=manager.engine.url,
            target_metadata=target_metadata,
            include_object=include_object,
            compare_type=True,
            compare_server_default=True,
            render_as_batch=manager.backend == "sqlite",
            user_module_prefix="schema_types.",
            literal_binds=True,
            transactional_ddl=True,
        )
        with context.begin_transaction():
            context.run_migrations()
    finally:
        manager.dispose()

###############################################################################
def _run_online(connection: Connection) -> None:
    _configure(connection)
    with context.begin_transaction():
        context.run_migrations()

###############################################################################
def run_migrations_online() -> None:
    injected = config.attributes.get("connection")
    if injected is not None:
        _run_online(injected)
        return

    settings = get_server_settings().database
    manager = DatabaseManager(settings)
    try:
        with manager.engine.connect() as connection:
            with connection.begin():
                _run_online(connection)
    finally:
        manager.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
