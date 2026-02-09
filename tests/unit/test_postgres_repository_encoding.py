from ADSMOD.server.configurations.server import DatabaseSettings
from ADSMOD.server.repositories.database.postgres import PostgresRepository


def test_postgres_connect_args_enforce_utf8_client_encoding() -> None:
    settings = DatabaseSettings(
        embedded_database=False,
        engine="postgres",
        host="localhost",
        port=5432,
        database_name="ADSMOD",
        username="postgres",
        password="admin",
        ssl=False,
        ssl_ca=None,
        connect_timeout=30,
        insert_batch_size=1000,
    )
    connect_args = PostgresRepository.build_connect_args(settings)
    assert connect_args["client_encoding"] == "utf8"
    assert connect_args["connect_timeout"] == 30
