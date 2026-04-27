import pytest
from sqlalchemy import Column, Integer, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from ADSMOD.server.repositories.schemas.types import JSONSequence

Base = declarative_base()


class TestModel(Base):
    __tablename__ = "test_data"
    id = Column(Integer, primary_key=True)
    sequence = Column(JSONSequence)


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    yield Session()


def test_json_sequence_round_trip(session):
    """Verify list data is stored and retrieved correctly."""
    data = [1.1, 2.2, 3.3]
    item = TestModel(sequence=data)
    session.add(item)
    session.commit()

    retrieved = session.query(TestModel).first()
    assert retrieved.sequence == data
    assert isinstance(retrieved.sequence, list)


def test_json_sequence_empty_list(session):
    """Verify empty lists are handled correctly."""
    item = TestModel(sequence=[])
    session.add(item)
    session.commit()

    retrieved = session.query(TestModel).first()
    assert retrieved.sequence == []


def test_json_sequence_none(session):
    """Verify valid None storage."""
    item = TestModel(sequence=None)
    session.add(item)
    session.commit()

    retrieved = session.query(TestModel).first()
    assert retrieved.sequence is None


def test_string_payload_raises_for_json_sequence(session):
    """Verify string payloads are rejected for JSON sequence columns."""
    session.execute(TestModel.__table__.insert().values(sequence="1.1, 2.2, 3.3"))
    session.commit()

    with pytest.raises(ValueError, match="Invalid JSONSequence payload"):
        _ = session.query(TestModel).first()
