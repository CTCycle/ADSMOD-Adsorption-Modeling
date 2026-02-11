from sqlalchemy import UniqueConstraint, create_engine, event, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateTable

from ADSMOD.server.repositories.schemas.models import (
    Adsorbate,
    Adsorbent,
    AdsorptionIsotherm,
    AdsorptionIsothermComponent,
    AdsorptionFit,
    AdsorptionPoint,
    AdsorptionPointComponent,
    Base,
    Dataset,
)


def test_adsorption_point_component_uses_composite_primary_key() -> None:
    table = AdsorptionPointComponent.__table__
    assert "id" not in table.columns
    assert [column.name for column in table.primary_key.columns] == [
        "point_id",
        "component_id",
    ]


def test_adsorption_fit_keeps_surrogate_id_and_natural_uniqueness() -> None:
    table = AdsorptionFit.__table__
    assert [column.name for column in table.primary_key.columns] == ["id"]
    assert table.columns["id"].autoincrement is True

    unique_constraints = {
        tuple(constraint.columns.keys())
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    assert ("processed_id", "model_name", "optimization_method") in unique_constraints


def test_adsorption_point_component_ddl_compiles_for_postgresql() -> None:
    ddl = str(
        CreateTable(AdsorptionPointComponent.__table__).compile(
            dialect=postgresql.dialect()
        )
    )
    assert "PRIMARY KEY (point_id, component_id)" in ddl
    assert " id " not in ddl.lower()


def test_sqlite_drop_and_recreate_restarts_identity() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        session.add(Dataset(dataset_name="first", source="uploaded", created_at="t1"))
        session.commit()
        first_id = session.execute(select(Dataset.id)).scalar_one()
    assert first_id == 1

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        session.add(Dataset(dataset_name="second", source="uploaded", created_at="t2"))
        session.commit()
        second_id = session.execute(select(Dataset.id)).scalar_one()
    assert second_id == 1


def test_point_component_enforces_composite_identity_and_cascade() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, connection_record) -> None:  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()

    Base.metadata.create_all(engine)

    with Session(engine) as session:
        dataset = Dataset(dataset_name="d1", source="uploaded", created_at="t1")
        adsorbent = Adsorbent(adsorbent_key="a1", name="host")
        adsorbate = Adsorbate(adsorbate_key="g1", name="guest")
        session.add_all([dataset, adsorbent, adsorbate])
        session.flush()

        isotherm = AdsorptionIsotherm(
            dataset_id=dataset.id,
            source_record_id="s1",
            experiment_name="exp1",
            adsorbent_id=adsorbent.id,
            temperature_k=298.15,
            pressure_units="pa",
            adsorption_units="mol/g",
            created_at="t1",
        )
        session.add(isotherm)
        session.flush()

        component = AdsorptionIsothermComponent(
            isotherm_id=isotherm.id,
            component_index=1,
            adsorbate_id=adsorbate.id,
            mole_fraction=1.0,
        )
        point = AdsorptionPoint(isotherm_id=isotherm.id, point_index=0)
        session.add_all([component, point])
        session.flush()

        session.add(
            AdsorptionPointComponent(
                point_id=point.id,
                component_id=component.id,
                partial_pressure_pa=1000.0,
                uptake_mol_g=0.5,
                original_pressure=1000.0,
                original_uptake=0.5,
            )
        )
        session.commit()

        session.add(
            AdsorptionPointComponent(
                point_id=point.id,
                component_id=component.id,
                partial_pressure_pa=1200.0,
                uptake_mol_g=0.6,
                original_pressure=1200.0,
                original_uptake=0.6,
            )
        )
        try:
            session.commit()
            raise AssertionError("Expected duplicate composite primary key to fail.")
        except IntegrityError:
            session.rollback()

        session.delete(point)
        session.commit()

        count = session.execute(select(AdsorptionPointComponent)).scalars().all()
        assert count == []
