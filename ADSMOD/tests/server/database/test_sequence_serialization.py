import pytest
from sqlalchemy import Column, Integer, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from ADSMOD.server.repositories.types import JSONSequence

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

def test_legacy_string_fallback(session):
    """Verify that a manually inserted legacy CSV string is read as a list."""
    # Bypass ORM to insert raw string
    session.execute(
        TestModel.__table__.insert().values(sequence="1.1, 2.2, 3.3")
    )
    session.commit()

    retrieved = session.query(TestModel).first()
    assert retrieved.sequence == [1.1, 2.2, 3.3]

def test_mixed_legacy_string(session):
    """Verify comma-separated string trimming."""
    session.execute(
        TestModel.__table__.insert().values(sequence=" 10 , 20 ,30 ")
    )
    session.commit()
    
    # The fallback path normalizes numeric strings into floats.
    retrieved = session.query(TestModel).first()
    assert retrieved.sequence == [10.0, 20.0, 30.0]
