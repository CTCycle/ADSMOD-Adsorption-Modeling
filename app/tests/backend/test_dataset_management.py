# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""Dataset management: hierarchy listing, in-place editing, and safe deletion."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event

from server.app import create_app
from server.configurations.settings import AdsmodConfig, DatabaseConfig, UserSettings
from server.repositories import dataset_edits
from server.repositories.database.manager import DatabaseManager
from server.repositories.dataset_edits import DatasetEditRepository
from server.repositories.datasets import DatasetRepository
from server.repositories.materials import MaterialRepository
from server.repositories.nist import NISTRepository
from server.repositories.schemas.models import Base, Observation

CSV = (
    "experiment,adsorbent,adsorbate,temperature [K],pressure [bar],uptake [mol/kg]\n"
    "A,Zeolite 13X,CO2,298.15,0.1,1.0\n"
    "A,Zeolite 13X,CO2,298.15,0.5,2.0\n"
    "A,Zeolite 13X,CO2,298.15,1.0,2.5\n"
    "B,Activated carbon,CH4,273.15,0.1,0.5\n"
    "B,Activated carbon,CH4,273.15,1.0,1.5\n"
)
MAPPING: dict[str, Any] = {
    "dataset_name": "Screening set",
    "structure": "atomic",
    "column_roles": {
        "experiment": "experiment_name",
        "adsorbent": "adsorbent",
        "adsorbate": "adsorbate",
        "temperature [K]": "temperature",
        "pressure [bar]": "pressure",
        "uptake [mol/kg]": "uptake",
    },
    "grouping_columns": ["experiment"],
    "unit_overrides": {"pressure": "bar", "uptake": "mol/kg", "temperature": "K"},
    "pressure_basis": "absolute",
    "duplicate_policy": "reject",
}


###############################################################################
@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    config = AdsmodConfig.assemble(
        database=DatabaseConfig(embedded_database=True, sqlite_path="data/test.db"),
        storage_root=tmp_path,
        settings=UserSettings(),
    )
    with TestClient(create_app(config)) as test_client:
        yield test_client


###############################################################################
def _import(client: TestClient, name: str = "Screening set") -> dict[str, Any]:
    mapping = {**MAPPING, "dataset_name": name}
    files = {"file": ("screening.csv", CSV.encode("utf-8"), "text/csv")}
    validation = client.post(
        "/api/v1/datasets/import/validate",
        data={"mapping": json.dumps(mapping)},
        files=files,
    ).json()
    mapping["confirmed_issue_codes"] = sorted(
        {issue["code"] for issue in validation["issues"] if issue["severity"] == "confirmation"}
    )
    response = client.post(
        "/api/v1/datasets/import/commit",
        data={"mapping": json.dumps(mapping)},
        files=files,
    )
    assert response.status_code == 201, response.text
    return response.json()["dataset"]


###############################################################################
def _experiments(client: TestClient, dataset_id: int, **params: Any) -> dict[str, Any]:
    response = client.get(f"/api/v1/datasets/{dataset_id}/experiments", params=params)
    assert response.status_code == 200, response.text
    return response.json()


###############################################################################
def _observations(client: TestClient, dataset_id: int, experiment_id: int) -> list[dict[str, Any]]:
    response = client.get(
        f"/api/v1/datasets/{dataset_id}/experiments/{experiment_id}/observations",
        params={"order": "sequence"},
    )
    assert response.status_code == 200, response.text
    return response.json()["rows"]


###############################################################################
def _experiment(client: TestClient, dataset_id: int, name: str) -> dict[str, Any]:
    return next(
        item for item in _experiments(client, dataset_id)["experiments"] if item["name"] == name
    )


###############################################################################
def test_multi_experiment_import_is_listed_as_one_dataset(client: TestClient) -> None:
    dataset = _import(client)

    listed = client.get("/api/v1/datasets").json()["datasets"]
    assert [item["id"] for item in listed] == [dataset["id"]]
    summary = listed[0]
    assert summary["experiment_count"] == 2
    assert summary["observation_count"] == 5
    assert summary["material_count"] == 2 and summary["species_count"] == 2
    assert sorted(summary["material_names"]) == ["Activated carbon", "Zeolite 13X"]
    assert sorted(summary["species_names"]) == ["Carbon dioxide", "Methane"]
    assert summary["last_update"] and summary["edited_experiment_count"] == 0

    page = _experiments(client, dataset["id"], limit=1)
    assert page["total"] == 2 and len(page["experiments"]) == 1
    co2 = _experiment(client, dataset["id"], "A")
    assert co2["observation_count"] == 3 and co2["temperature_k"] == 298.15
    assert [component["name"] for component in co2["components"]] == ["Carbon dioxide"]
    filtered = _experiments(client, dataset["id"], adsorbate_id=co2["components"][0]["adsorbate_id"])
    assert [item["id"] for item in filtered["experiments"]] == [co2["id"]]

    detail = client.get(f"/api/v1/datasets/{dataset['id']}").json()
    assert detail["latest_import"]["original_filename"] == "screening.csv"
    experiment = client.get(f"/api/v1/datasets/{dataset['id']}/experiments/{co2['id']}")
    assert experiment.json()["experiment"]["adsorbent"] == "Zeolite 13X"


###############################################################################
def test_wide_species_columns_are_stored_as_one_experiment_with_two_species(
    client: TestClient,
) -> None:
    csv = (
        "experiment,adsorbent,temperature [K],pressure [bar],uptake_CO2 [mol/kg],uptake_N2 [mol/kg]\n"
        "M1,Zeolite 13X,298.15,0.1,1.0,0.2\n"
        "M1,Zeolite 13X,298.15,0.5,2.0,0.3\n"
        "M1,Zeolite 13X,298.15,1.0,2.5,0.4\n"
    )
    mapping = {
        "dataset_name": "Mixture",
        "structure": "atomic",
        "column_roles": {
            "experiment": "experiment_name",
            "adsorbent": "adsorbent",
            "temperature [K]": "temperature",
            "pressure [bar]": "pressure",
        },
        "grouping_columns": ["experiment"],
        "unit_overrides": {"pressure": "bar", "uptake": "mol/kg", "temperature": "K"},
        "pressure_basis": "absolute",
        "species_columns": [
            {"uptake_column": "uptake_CO2 [mol/kg]", "adsorbate": "CO2"},
            {"uptake_column": "uptake_N2 [mol/kg]", "adsorbate": "N2"},
        ],
    }
    files = {"file": ("mixture.csv", csv.encode("utf-8"), "text/csv")}
    response = client.post(
        "/api/v1/datasets/import/commit", data={"mapping": json.dumps(mapping)}, files=files
    )
    assert response.status_code == 201, response.text
    dataset = response.json()["dataset"]

    assert dataset["experiment_count"] == 1 and dataset["observation_count"] == 6
    experiment = _experiment(client, dataset["id"], "M1")
    assert experiment["observation_count"] == 6
    assert [component["name"] for component in experiment["components"]] == [
        "Carbon dioxide",
        "Nitrogen",
    ]
    rows = _observations(client, dataset["id"], experiment["id"])
    by_component: dict[int, list[float]] = {}
    for row in rows:
        by_component.setdefault(row["component_id"], []).append(row["uptake_mol_kg"])
    assert sorted(by_component.values()) == [[0.2, 0.3, 0.4], [1.0, 2.0, 2.5]]


###############################################################################
def test_edit_overwrites_values_recomputes_canonical_units_and_feeds_fitting(
    client: TestClient,
) -> None:
    dataset = _import(client)
    co2 = _experiment(client, dataset["id"], "A")
    first, second, _ = _observations(client, dataset["id"], co2["id"])
    before = client.get(f"/api/v1/datasets/{dataset['id']}").json()["dataset"]

    response = client.patch(
        f"/api/v1/datasets/{dataset['id']}",
        json={
            "name": "Screening set (curated)",
            "description": "Corrected first point.",
            "experiments": [
                {"id": co2["id"], "temperature_original": 30, "temperature_original_unit": "°C"}
            ],
            "observations": [
                {"id": first["id"], "pressure_original": 0.2, "uptake_stddev_mol_kg": 0.05}
            ],
        },
    )

    assert response.status_code == 200, response.text
    after = response.json()["dataset"]
    assert after["id"] == dataset["id"] and after["name"] == "Screening set (curated)"
    assert after["last_update"] > before["last_update"]
    assert after["edited_experiment_count"] == 1
    edited = _experiment(client, dataset["id"], "A")
    assert edited["temperature_k"] == pytest.approx(303.15)
    assert edited["edited_at"] is not None
    rows = {row["id"]: row for row in _observations(client, dataset["id"], co2["id"])}
    assert rows[first["id"]]["pressure_original"] == 0.2
    assert rows[first["id"]]["pressure_canonical"] == pytest.approx(20000.0)
    assert rows[first["id"]]["uptake_stddev_mol_kg"] == 0.05
    assert rows[second["id"]]["pressure_canonical"] == pytest.approx(50000.0)

    series = client.app.state.core_container.datasets.fitting_series(dataset["id"], co2["id"])
    assert series["temperature_k"] == pytest.approx(303.15)
    assert series["pressure"][0] == pytest.approx(20000.0)

    # Changing the experiment pressure unit reinterprets every point of it.
    client.patch(
        f"/api/v1/datasets/{dataset['id']}",
        json={"experiments": [{"id": co2["id"], "pressure_unit": "kPa"}]},
    ).raise_for_status()
    rows = {row["id"]: row for row in _observations(client, dataset["id"], co2["id"])}
    assert rows[second["id"]]["pressure_original_unit"] == "kPa"
    assert rows[second["id"]]["pressure_canonical"] == pytest.approx(500.0)


###############################################################################
def test_invalid_edit_reports_field_issues_and_persists_nothing(client: TestClient) -> None:
    dataset = _import(client)
    co2 = _experiment(client, dataset["id"], "A")
    first, second, _ = _observations(client, dataset["id"], co2["id"])

    response = client.patch(
        f"/api/v1/datasets/{dataset['id']}",
        json={
            "name": "Should not persist",
            "experiments": [
                {"id": co2["id"], "components": [{"id": co2["components"][0]["id"], "adsorbate_id": 99999}]}
            ],
            "observations": [
                {"id": first["id"], "pressure_original": -1.0},
                {"id": second["id"], "uptake_stddev_mol_kg": 0},
            ],
        },
    )

    assert response.status_code == 422
    issues = response.json()["detail"]["issues"]
    assert {(item["observation_id"], item["field"]) for item in issues if item["scope"] == "observation"} == {
        (first["id"], "pressure_original"),
        (second["id"], "uptake_stddev_mol_kg"),
    }
    assert any(item["field"] == "components" for item in issues)
    assert client.get(f"/api/v1/datasets/{dataset['id']}").json()["dataset"]["name"] == "Screening set"
    assert _observations(client, dataset["id"], co2["id"])[0]["pressure_original"] == 0.1


###############################################################################
def test_failed_write_rolls_back_the_whole_edit(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _import(client)
    co2 = _experiment(client, dataset["id"], "A")
    first = _observations(client, dataset["id"], co2["id"])[0]
    original_update = dataset_edits.update

    def failing_update(entity: Any) -> Any:
        if entity is Observation:
            raise RuntimeError("simulated storage failure")
        return original_update(entity)

    monkeypatch.setattr(dataset_edits, "update", failing_update)
    with pytest.raises(RuntimeError, match="simulated storage failure"):
        client.patch(
            f"/api/v1/datasets/{dataset['id']}",
            json={
                "name": "Partially saved",
                "experiments": [{"id": co2["id"], "name": "Renamed"}],
                "observations": [{"id": first["id"], "pressure_original": 0.3}],
            },
        )
    monkeypatch.undo()

    summary = client.get(f"/api/v1/datasets/{dataset['id']}").json()["dataset"]
    assert summary["name"] == "Screening set" and summary["edited_experiment_count"] == 0
    assert _experiment(client, dataset["id"], "A")["edited_at"] is None
    assert _observations(client, dataset["id"], co2["id"])[0]["pressure_original"] == 0.1


###############################################################################
def test_delete_keeps_shared_entities_and_entities_in_use_cannot_be_deleted(
    client: TestClient,
) -> None:
    first = _import(client, "First set")
    second = _import(client, "Second set")
    co2 = _experiment(client, first["id"], "A")
    material_id = co2["adsorbent_id"]
    species_id = co2["components"][0]["adsorbate_id"]

    material = client.get(f"/api/v1/materials/{material_id}").json()
    assert sorted(item["name"] for item in material["datasets"]) == ["First set", "Second set"]
    assert client.delete(f"/api/v1/materials/{material_id}").status_code == 409
    assert client.delete(f"/api/v1/species/{species_id}").status_code == 409

    impact = client.get(f"/api/v1/datasets/{first['id']}/delete-impact").json()
    assert (impact["experiment_count"], impact["observation_count"]) == (2, 5)
    assert client.delete(f"/api/v1/datasets/{first['id']}").status_code == 204
    assert client.get(f"/api/v1/datasets/{first['id']}").status_code == 404
    assert client.get(f"/api/v1/materials/{material_id}").json()["isotherm_count"] == 1
    assert client.delete(f"/api/v1/materials/{material_id}").status_code == 409

    assert client.delete(f"/api/v1/datasets/{second['id']}").status_code == 204
    assert client.delete(f"/api/v1/materials/{material_id}").status_code == 204
    assert client.delete(f"/api/v1/species/{species_id}").status_code == 204
    names = [item["name"] for item in client.get("/api/v1/species").json()["items"]]
    assert "CO2" not in names


###############################################################################
def _count_selects(manager: DatabaseManager, operation: Any) -> int:
    statements: list[str] = []

    def record(_conn: Any, _cursor: Any, statement: str, *_args: Any) -> None:
        if statement.lstrip().upper().startswith(("SELECT", "WITH")):
            statements.append(statement)

    event.listen(manager.engine, "before_cursor_execute", record)
    try:
        operation()
    finally:
        event.remove(manager.engine, "before_cursor_execute", record)
    return len(statements)


###############################################################################
def test_listing_queries_do_not_scale_with_dataset_or_experiment_count(
    client: TestClient,
) -> None:
    for index in range(4):
        _import(client, f"Set {index}")
    container = client.app.state.core_container
    repository: DatasetRepository = container.datasets
    dataset_id = repository.list_summaries()[0]["id"]

    assert _count_selects(container.database, repository.list_summaries) <= 3
    one_page = _count_selects(
        container.database, lambda: repository.experiments(dataset_id, limit=1)
    )
    full_page = _count_selects(
        container.database, lambda: repository.experiments(dataset_id, limit=50)
    )
    assert one_page == full_page


###############################################################################
def _canonical_experiment(external_key: str, uptake: float) -> dict[str, Any]:
    return {
        "external_key": external_key,
        "name": external_key,
        "adsorbent": {"name": "Silica", "external_identifier": "host-1"},
        "adsorbates": [{"name": "Methane", "inchi_key": "A" * 27}],
        "temperature_original": 298.0,
        "temperature_original_unit": "K",
        "temperature_k": 298.0,
        "pressure_basis": "absolute",
        "observations": [
            {
                "adsorbate": "Methane",
                "sequence_index": index,
                "pressure_original": pressure,
                "pressure_original_unit": "Pa",
                "pressure_canonical": pressure,
                "pressure_canonical_unit": "Pa",
                "uptake_original": uptake * (index + 1),
                "uptake_original_unit": "mol/kg",
                "uptake_mol_kg": uptake * (index + 1),
            }
            for index, pressure in enumerate((1000.0, 2000.0))
        ],
    }


###############################################################################
def test_nist_redownload_keeps_local_edits_unless_overwrite_is_confirmed() -> None:
    database = DatabaseManager(
        DatabaseConfig(embedded_database=True, connect_timeout=5, sqlite_path=":memory:")
    )
    Base.metadata.create_all(database.engine)
    try:
        datasets = DatasetRepository(database)
        edits = DatasetEditRepository(database)
        nist = NISTRepository(
            database=database,
            datasets=datasets,
            materials=MaterialRepository(database),
            edits=edits,
        )
        nist.save_experiments([_canonical_experiment("iso-1", 1.0)])
        dataset_id = nist.nist_dataset_id()
        assert dataset_id is not None
        experiment = datasets.experiments(dataset_id)[0][0]
        rows, _ = datasets.observations(dataset_id, experiment["id"])
        edits.apply(
            dataset_id,
            {
                "dataset": {},
                "experiments": {},
                "components": {},
                "observations": {rows[0]["id"]: {"uptake_original": 9.0, "uptake_mol_kg": 9.0}},
                "edited_experiment_ids": {experiment["id"]},
            },
        )
        assert nist.edited_experiment_count() == 1

        nist.save_experiments([_canonical_experiment("iso-1", 1.0)])
        rows, _ = datasets.observations(dataset_id, experiment["id"])
        assert rows[0]["uptake_mol_kg"] == 9.0

        nist.save_experiments([_canonical_experiment("iso-1", 1.0)], restore_edited=True)
        restored = datasets.experiments(dataset_id)[0][0]
        rows, _ = datasets.observations(dataset_id, restored["id"])
        assert restored["id"] == experiment["id"] and restored["edited_at"] is None
        assert [row["uptake_mol_kg"] for row in rows] == [1.0, 2.0]
        assert nist.edited_experiment_count() == 0
    finally:
        database.dispose()
