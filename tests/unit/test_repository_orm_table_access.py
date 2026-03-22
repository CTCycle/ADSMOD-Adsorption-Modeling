from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ADSMOD.server.repositories.database.postgres import PostgresRepository
from ADSMOD.server.repositories.database.sqlite import SQLiteRepository
from ADSMOD.server.repositories.schemas.models import Base, Dataset


def seed_datasets(engine) -> None:  # type: ignore[no-untyped-def]
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all(
            [
                Dataset(dataset_name="dataset_a", source="uploaded", created_at="t1"),
                Dataset(dataset_name="dataset_b", source="uploaded", created_at="t2"),
            ]
        )
        session.commit()


def test_sqlite_repository_load_and_count_use_mapped_models() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    seed_datasets(engine)

    repository = SQLiteRepository.__new__(SQLiteRepository)
    repository.session_factory = sessionmaker(bind=engine, future=True)

    frame = repository.load_from_database("datasets", limit=1, offset=1)

    assert frame.shape[0] == 1
    assert frame.loc[0, "dataset_name"] == "dataset_b"
    assert list(frame.columns) == ["id", "dataset_name", "source", "created_at"]
    assert repository.count_rows("datasets") == 2
    assert repository.load_from_database("missing_table").empty
    assert repository.count_rows("missing_table") == 0


def test_postgres_repository_load_and_count_use_mapped_models() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    seed_datasets(engine)

    repository = PostgresRepository.__new__(PostgresRepository)
    repository.session = sessionmaker(bind=engine, future=True)

    frame = repository.load_from_database("datasets")

    assert frame["dataset_name"].tolist() == ["dataset_a", "dataset_b"]
    assert list(frame.columns) == ["id", "dataset_name", "source", "created_at"]
    assert repository.count_rows("datasets") == 2
    assert repository.load_from_database("missing_table").empty
    assert repository.count_rows("missing_table") == 0
