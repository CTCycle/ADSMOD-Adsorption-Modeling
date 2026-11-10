# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""Lifecycle of the unified adsorption model: one experiment, 1..N species,
one long-format measurement table keyed by (experiment, species, pressure)."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select, text
from sqlalchemy.exc import IntegrityError

from server.app import create_app
from server.configurations.settings import AdsmodConfig, DatabaseConfig, UserSettings
from server.domain.canonical import (
    CanonicalExperiment,
    CanonicalizationError,
    normalize_experiment,
    pressure_key,
)
from server.domain.datasets import ImportMapping
from server.repositories.schemas.models import (
    Adsorbate,
    Isotherm,
    IsothermComponent,
    Observation,
)
from server.services.data.importer import AdsorptionImportEngine
from server.services.data.nist_mapper import NISTCanonicalMapper
from server.services.data.nistads import NISTDatasetBuilder

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

SINGLE_CSV = (
    "experiment,adsorbent,adsorbate,temperature [K],pressure [bar],uptake [mol/kg]\n"
    + "".join(
        f"CO2-13X,Zeolite 13X,CO2,298.15,{pressure},{6.0 * 2.0 * pressure / (1 + 2.0 * pressure):.6f}\n"
        for pressure in (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)
    )
)
SINGLE_MAPPING: dict[str, Any] = {
    "dataset_name": "Single species",
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
}

# Wide layout: one uptake and one gas-fraction column per species.
MIXTURE_CSV = (
    "experiment,adsorbent,temperature [K],pressure [bar],CO2 uptake,N2 uptake,y CO2,y N2\n"
    "MIX-1,Zeolite 13X,298.15,1.0,2.15,0.32,0.15,0.85\n"
    "MIX-1,Zeolite 13X,298.15,2.0,3.41,0.48,0.15,0.85\n"
    "MIX-1,Zeolite 13X,298.15,4.0,4.02,0.71,0.15,0.85\n"
)
MIXTURE_MAPPING: dict[str, Any] = {
    "dataset_name": "Binary mixture",
    "structure": "atomic",
    "column_roles": {
        "experiment": "experiment_name",
        "adsorbent": "adsorbent",
        "temperature [K]": "temperature",
        "pressure [bar]": "pressure",
        "CO2 uptake": "ignore",
        "N2 uptake": "ignore",
        "y CO2": "ignore",
        "y N2": "ignore",
    },
    "grouping_columns": ["experiment"],
    "unit_overrides": {"pressure": "bar", "uptake": "mol/kg", "temperature": "K"},
    "pressure_basis": "absolute",
    "species_columns": [
        {
            "uptake_column": "CO2 uptake",
            "adsorbate": "CO2",
            "mole_fraction_column": "y CO2",
        },
        {
            "uptake_column": "N2 uptake",
            "adsorbate": "N2",
            "mole_fraction_column": "y N2",
        },
    ],
}


###############################################################################
@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    config = AdsmodConfig.assemble(
        database=DatabaseConfig(embedded_database=True, sqlite_path="data/unified.db"),
        storage_root=tmp_path,
        settings=UserSettings(),
    )
    with TestClient(create_app(config)) as test_client:
        yield test_client


###############################################################################
def _files(csv: str) -> dict[str, Any]:
    return {"file": ("data.csv", csv.encode("utf-8"), "text/csv")}


###############################################################################
def _validate(client: TestClient, csv: str, mapping: dict[str, Any]) -> dict[str, Any]:
    response = client.post(
        "/api/v1/datasets/import/validate",
        data={"mapping": json.dumps(mapping)},
        files=_files(csv),
    )
    assert response.status_code == 200, response.text
    return response.json()


###############################################################################
def _commit(client: TestClient, csv: str, mapping: dict[str, Any]) -> int:
    response = client.post(
        "/api/v1/datasets/import/commit",
        data={"mapping": json.dumps(mapping)},
        files=_files(csv),
    )
    assert response.status_code == 201, response.text
    return int(response.json()["dataset"]["id"])


###############################################################################
def _experiments(client: TestClient, dataset_id: int) -> list[dict[str, Any]]:
    return client.get(f"/api/v1/datasets/{dataset_id}/experiments").json()[
        "experiments"
    ]


###############################################################################
def _rows(
    client: TestClient, dataset_id: int, experiment_id: int
) -> list[dict[str, Any]]:
    return client.get(
        f"/api/v1/datasets/{dataset_id}/experiments/{experiment_id}/observations"
    ).json()["rows"]


###############################################################################
def _database(client: TestClient) -> Any:
    return client.app.state.core_container.database


###############################################################################
def _nist_payload(
    name: str, species: list[str], points: list[tuple[float, list[float]]]
) -> dict[str, Any]:
    keys = [f"{index}" * 27 for index in range(1, len(species) + 1)]
    fraction = round(1 / len(species), 6)
    return {
        "filename": name,
        "adsorbent": {"hashkey": "host-9", "name": "MOF-5"},
        "adsorbates": [
            {"name": item, "InChIKey": key}
            for item, key in zip(species, keys, strict=True)
        ],
        "temperature": 303.0,
        "pressureUnits": "bar",
        "adsorptionUnits": "mmol/g",
        "compositionType": "molefraction",
        "isotherm_data": [
            {
                "pressure": pressure,
                "species_data": [
                    {"InChIKey": key, "composition": fraction, "adsorption": uptake}
                    for key, uptake in zip(keys, uptakes, strict=True)
                ],
            }
            for pressure, uptakes in points
        ],
    }


###############################################################################
def _nist_records(
    payloads: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    mapper = NISTCanonicalMapper()
    records, skipped = mapper.experiment_records_with_skips(
        NISTDatasetBuilder().build_dataset(pd.DataFrame(payloads))
    )
    mapper.attach_source_identities(records, payloads)
    return records, skipped


###############################################################################
def _fit(client: TestClient, dataset_id: int) -> dict[str, Any]:
    started = client.post(
        "/api/v1/fitting/sessions",
        json={"dataset_id": dataset_id, "name": "Unified", "models": ["langmuir"]},
    )
    assert started.status_code == 200, started.text
    job_id = started.json()["job_id"]
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if client.get(f"/api/v1/fitting/jobs/{job_id}").json()["status"] in {
            "completed",
            "failed",
            "cancelled",
        }:
            break
        time.sleep(0.05)
    return client.get(f"/api/v1/fitting/sessions/{started.json()['session_id']}").json()


###############################################################################
def test_single_species_experiment_is_stored_plotted_and_fitted(
    client: TestClient,
) -> None:
    dataset_id = _commit(client, SINGLE_CSV, SINGLE_MAPPING)

    [experiment] = _experiments(client, dataset_id)
    assert [item["name"] for item in experiment["components"]] == ["Carbon dioxide"]
    assert experiment["temperature_k"] == 298.15
    rows = _rows(client, dataset_id, experiment["id"])
    assert len(rows) == 7
    assert {row["adsorbate_id"] for row in rows} == {
        experiment["components"][0]["adsorbate_id"]
    }
    assert all(row["gas_mole_fraction"] is None for row in rows)

    series = client.post(
        "/api/v1/analytics/series",
        json={"filter": {"dataset_ids": [dataset_id], "pressure_basis": "absolute"}},
    ).json()["series"]
    assert [
        (item["isotherm_id"], item["species_count"], len(item["points"]))
        for item in series
    ] == [(experiment["id"], 1, 7)]

    detail = _fit(client, dataset_id)
    assert detail["status"] in {"completed", "warning"}
    [result] = detail["experiments"][0]["results"]
    parameters = {item["name"]: item["value"] for item in result["parameters"]}
    # Synthetic Langmuir data: qsat = 6 mol/kg, K = 2 bar^-1 = 2e-5 Pa^-1.
    assert parameters["qsat"] == pytest.approx(6.0, rel=1e-3)


###############################################################################
def test_two_species_wide_file_uses_the_same_rows_and_reports_model_capability(
    client: TestClient,
) -> None:
    dataset_id = _commit(client, MIXTURE_CSV, MIXTURE_MAPPING)

    [experiment] = _experiments(client, dataset_id)
    names = {item["adsorbate_id"]: item["name"] for item in experiment["components"]}
    assert sorted(names.values()) == ["Carbon dioxide", "Nitrogen"]
    rows = _rows(client, dataset_id, experiment["id"])
    by_species = {
        names[adsorbate_id]: sorted(
            (row["pressure_canonical"], row["uptake_mol_kg"], row["gas_mole_fraction"])
            for row in rows
            if row["adsorbate_id"] == adsorbate_id
        )
        for adsorbate_id in names
    }
    # Both species share the experiment and the same total pressures.
    assert by_species == {
        "Carbon dioxide": [(1e5, 2.15, 0.15), (2e5, 3.41, 0.15), (4e5, 4.02, 0.15)],
        "Nitrogen": [(1e5, 0.32, 0.85), (2e5, 0.48, 0.85), (4e5, 0.71, 0.85)],
    }
    assert experiment["fitting_eligible"] is False
    assert "single-species" in experiment["ineligibility_reason"]

    readiness = client.get(f"/api/v1/fitting/datasets/{dataset_id}/readiness").json()
    assert readiness["ready"] is False
    series = client.post(
        "/api/v1/analytics/series",
        json={"filter": {"dataset_ids": [dataset_id], "include_multi_species": True}},
    ).json()["series"]
    assert sorted((item["adsorbate"], item["species_count"]) for item in series) == [
        ("Carbon dioxide", 2),
        ("Nitrogen", 2),
    ]


###############################################################################
def test_three_species_nist_experiment_persists_without_schema_change(
    client: TestClient,
) -> None:
    records, skipped = _nist_records(
        [
            _nist_payload(
                "ternary-1",
                ["CO2", "N2", "CH4"],
                [(1.0, [1.2, 0.2, 0.4]), (2.0, [1.9, 0.3, 0.7])],
            ),
            _nist_payload("pure-1", ["CO2"] * 1, [(1.0, [1.0]), (2.0, [1.6])]),
        ]
    )
    assert skipped == {}
    container = client.app.state.core_container
    container.nist_repository.save_experiments(records)

    with _database(client).session_factory() as session:
        species_counts = dict(
            session.execute(
                select(
                    Isotherm.external_key, func.count(IsothermComponent.adsorbate_id)
                )
                .join(IsothermComponent, IsothermComponent.isotherm_id == Isotherm.id)
                .group_by(Isotherm.external_key)
            ).all()
        )
        assert species_counts == {"ternary-1": 3, "pure-1": 1}
        ternary = session.scalars(
            select(Observation)
            .join(Isotherm, Isotherm.id == Observation.isotherm_id)
            .where(Isotherm.external_key == "ternary-1")
        ).all()
        assert len(ternary) == 6
        assert {row.pressure_canonical for row in ternary} == {1e5, 2e5}
        # Partial pressure remains exactly derivable from total pressure and y.
        assert {
            round(row.pressure_canonical * row.gas_mole_fraction, 3) for row in ternary
        } == {
            round(1e5 * 0.333333, 3),
            round(2e5 * 0.333333, 3),
        }
    assert container.nist_repository.count_nist_rows()["multi_species_experiments"] == 1


###############################################################################
def test_every_source_produces_the_same_canonical_record_structure() -> None:
    uploaded = (
        AdsorptionImportEngine()
        .validate(
            MIXTURE_CSV.encode("utf-8"),
            "data.csv",
            ImportMapping.model_validate(MIXTURE_MAPPING),
        )
        .experiments
    )
    nist, _ = _nist_records(
        [
            _nist_payload(
                "binary-1", ["CO2", "N2"], [(1.0, [1.0, 0.2]), (2.0, [1.5, 0.3])]
            )
        ]
    )
    for record in [*uploaded, *nist]:
        canonical = CanonicalExperiment.model_validate(record)
        assert canonical.pressure_basis == "absolute"
        assert {species.name for species in canonical.adsorbates} == {
            observation.adsorbate for observation in canonical.observations
        }
    observation_keys = {
        frozenset(observation)
        for record in [*uploaded, *nist]
        for observation in record["observations"]
    }
    assert len(observation_keys) == 1


###############################################################################
def test_duplicate_measurements_are_rejected_or_averaged_never_overwritten(
    client: TestClient,
) -> None:
    duplicated = SINGLE_CSV + "CO2-13X,Zeolite 13X,CO2,298.15,0.4,2.50\n"

    rejected = _validate(client, duplicated, SINGLE_MAPPING)
    assert rejected["status"] == "invalid"
    assert [issue["code"] for issue in rejected["issues"]] == ["duplicate_measurement"]

    dataset_id = _commit(
        client, duplicated, {**SINGLE_MAPPING, "duplicate_policy": "average"}
    )
    [experiment] = _experiments(client, dataset_id)
    rows = [
        row
        for row in _rows(client, dataset_id, experiment["id"])
        if row["pressure_canonical"] == 4e4
    ]
    assert len(rows) == 1
    assert rows[0]["uptake_mol_kg"] == pytest.approx((6.0 * 0.8 / 1.8 + 2.5) / 2)

    with _database(client).session_factory() as session:
        stored = session.get(Observation, rows[0]["id"])
        assert stored is not None
        assert stored.extra_metadata["replicates"]["count"] == 2
        session.add(
            Observation(
                isotherm_id=stored.isotherm_id,
                adsorbate_id=stored.adsorbate_id,
                pressure_key=pressure_key(4e4 + 1e-9, "Pa"),
                pressure_original=0.4,
                pressure_original_unit="bar",
                pressure_canonical=4e4 + 1e-9,
                pressure_canonical_unit="Pa",
                uptake_original=9.0,
                uptake_original_unit="mol/kg",
                uptake_mol_kg=9.0,
            )
        )
        with pytest.raises(IntegrityError, match="pressure_key"):
            session.flush()


###############################################################################
def test_pressure_key_and_canonical_rules_are_deterministic() -> None:
    assert pressure_key(101325.0, "Pa") == pressure_key(101325.0 + 4e-7, "Pa")
    assert pressure_key(101325.0, "Pa") != pressure_key(101325.0 + 2e-6, "Pa")
    assert pressure_key(0.5, "1") != pressure_key(0.500000002, "1")

    def experiment(**changes: Any) -> dict[str, Any]:
        record = {
            "external_key": "E",
            "name": "E",
            "adsorbent": {"name": "13X"},
            "adsorbates": [{"name": "CO2"}, {"name": "N2"}],
            "temperature_original": 298.15,
            "temperature_original_unit": "K",
            "temperature_k": 298.15,
            "pressure_basis": "absolute",
            "observations": [
                {
                    "adsorbate": species,
                    "pressure_original": pressure,
                    "pressure_original_unit": "Pa",
                    "pressure_canonical": pressure,
                    "pressure_canonical_unit": "Pa",
                    "uptake_original": 1.0,
                    "uptake_original_unit": "mol/kg",
                    "uptake_mol_kg": 1.0,
                    "gas_mole_fraction": fraction,
                }
                for species, fraction in (("CO2", 0.6), ("N2", 0.6))
                for pressure in (1e5, 2e5)
            ],
        }
        return {**record, **changes}

    with pytest.raises(CanonicalizationError) as excess:
        normalize_experiment(experiment())
    assert excess.value.code == "gas_fraction_sum"
    with pytest.raises(CanonicalizationError) as relative:
        normalize_experiment(experiment(pressure_basis="relative"))
    assert relative.value.code == "relative_pressure_multi_species"
    record = experiment()
    record["observations"][0]["adsorbate"] = "CH4"
    with pytest.raises(CanonicalizationError) as undeclared:
        normalize_experiment(record)
    assert undeclared.value.code == "undeclared_species"


###############################################################################
def test_one_experiment_has_exactly_one_temperature(client: TestClient) -> None:
    mixed = SINGLE_CSV.replace(
        "CO2-13X,Zeolite 13X,CO2,298.15,3.2", "CO2-13X,Zeolite 13X,CO2,308.15,3.2"
    )

    result = _validate(client, mixed, SINGLE_MAPPING)

    assert result["status"] == "invalid"
    assert any(
        issue["code"] == "inconsistent_experiment_metadata"
        and "temperature" in issue["message"]
        for issue in result["issues"]
    )
    # Grouping by temperature as well splits the rows into two experiments.
    split = _validate(
        client,
        mixed + "CO2-13X,Zeolite 13X,CO2,308.15,6.4,5.5\n",
        {**SINGLE_MAPPING, "grouping_columns": ["experiment", "temperature [K]"]},
    )
    assert split["status"] == "valid"
    assert sorted(item["temperature_k"] for item in split["experiments"]) == [
        298.15,
        308.15,
    ]


###############################################################################
def test_species_relationships_keep_database_integrity(client: TestClient) -> None:
    dataset_id = _commit(client, MIXTURE_CSV, MIXTURE_MAPPING)
    [experiment] = _experiments(client, dataset_id)
    database = _database(client)

    with database.session_factory() as session:
        assert session.execute(text("PRAGMA foreign_key_check")).all() == []
        other = session.scalar(
            select(Adsorbate.id).where(Adsorbate.name == "Carbon dioxide")
        )
        # A measurement can only reference a species of its own experiment.
        session.add(
            Observation(
                isotherm_id=experiment["id"],
                adsorbate_id=(other or 0) + 1000,
                pressure_key=pressure_key(1e5, "Pa"),
                pressure_original=1.0,
                pressure_original_unit="bar",
                pressure_canonical=1e5,
                pressure_canonical_unit="Pa",
                uptake_original=1.0,
                uptake_original_unit="mol/kg",
                uptake_mol_kg=1.0,
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()
        session.rollback()
        # Species referenced by an experiment cannot be deleted.
        with pytest.raises(IntegrityError):
            session.execute(delete(Adsorbate).where(Adsorbate.id == other))


###############################################################################
def test_no_component_depends_on_retired_single_or_binary_entities() -> None:
    retired = re.compile(
        r"single_component|binary_mixture|compound_[12]_|num_guests|component_id"
        r"|include_mixtures|mixture_isotherms|duplicate_policy.*keep"
    )
    roots = [
        REPOSITORY_ROOT / "app" / "server",
        REPOSITORY_ROOT / "app" / "client" / "src",
    ]
    skipped = {".venv", "__pycache__", "node_modules", "openapi"}
    hits = [
        f"{path.relative_to(REPOSITORY_ROOT)}:{number}"
        for root in roots
        for path in root.rglob("*")
        if path.suffix in {".py", ".ts", ".html"}
        and not skipped.intersection(path.parts)
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if retired.search(line)
    ]
    assert hits == []
