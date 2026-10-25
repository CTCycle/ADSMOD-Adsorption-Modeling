# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import pytest
from sqlalchemy import Column, Integer, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from server.repositories.schemas.types import JSONSequence


Base = declarative_base()

###############################################################################
class SequenceModel(Base):
    __tablename__ = "test_data"

    id = Column(Integer, primary_key=True)
    sequence = Column(JSONSequence)

###############################################################################
@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as current_session:
        yield current_session

###############################################################################
@pytest.mark.parametrize("data", [[1.1, 2.2, 3.3], [], None])
def test_json_sequence_round_trip(session, data) -> None:  # type: ignore[no-untyped-def]
    session.add(SequenceModel(sequence=data))
    session.commit()

    retrieved = session.query(SequenceModel).first()
    assert retrieved is not None
    assert retrieved.sequence == data

###############################################################################
def test_string_payload_raises_for_json_sequence(session) -> None:  # type: ignore[no-untyped-def]
    session.execute(SequenceModel.__table__.insert().values(sequence="1.1, 2.2, 3.3"))
    session.commit()

    with pytest.raises(ValueError, match="Invalid JSONSequence payload"):
        session.query(SequenceModel).first()
