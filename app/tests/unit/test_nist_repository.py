# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path

import pandas as pd

from server.configurations.settings import DatabaseConfig
from server.repositories.public_data import PublicDataRepository
from server.services.data.nist_mapper import NISTCanonicalMapper
from server.repositories.database.manager import DatabaseManager
from server.repositories.datasets import DatasetRepository
from server.repositories.materials import MaterialRepository
from server.repositories.nist import NISTRepository
from server.repositories.schemas.models import Adsorbate, Adsorbent, Base
from server.repositories.schemas.public_data import Reference
from sqlalchemy import func, select

from server.services.providers import PROVIDER_CLASSES

PUBLIC_SOURCE_DEFINITIONS = [provider.source_definition() for provider in PROVIDER_CLASSES]

###############################################################################
def build_nist_repository(path: Path) -> NISTRepository:
    settings = DatabaseConfig(
        embedded_database=True,
        connect_timeout=30,
        sqlite_path=str(path),
    )
    database = DatabaseManager(settings)
    Base.metadata.create_all(database.engine)
    datasets = DatasetRepository(database)
    public_data = PublicDataRepository(database)
    public_data.ensure_sources(PUBLIC_SOURCE_DEFINITIONS)
    return NISTRepository(
        database=database,
        datasets=datasets,
        materials=MaterialRepository(database),
        public_data=public_data,
    )

###############################################################################
def test_nist_repository_counts_and_loader_frame_are_canonical(
    tmp_path: Path,
) -> None:
    repository = build_nist_repository(tmp_path / "nist.db")
    mapper = NISTCanonicalMapper()

    repository.save_materials(
        mapper.material_records(
            pd.DataFrame(
                [
                    {"name": "methane", "InChIKey": "A" * 27},
                    {"name": "nitrogen", "InChIKey": "B" * 27},
                    {"name": "argon", "InChIKey": "C" * 27},
                ]
            ),
            "adsorbate",
        ),
        mapper.material_records(
            pd.DataFrame(
                [
                    {"name": "silica", "hashkey": "host-1"},
                    {"name": "carbon", "hashkey": "host-2"},
                ]
            ),
            "adsorbent",
        ),
    )

    single_component = pd.DataFrame(
        [
            {
                "name": "single-1",
                "pressure_units": "kPa",
                "adsorption_units": "mol/kg",
                "temperature": 298.15,
                "adsorbent": "silica",
                "adsorbate": "methane",
                "adsorbate_molecular_weight": 16.04,
                "pressure": 1.0,
                "adsorbed_amount": 0.1,
            },
            {
                "name": "single-1",
                "pressure_units": "kPa",
                "adsorption_units": "mol/kg",
                "temperature": 298.15,
                "adsorbent": "silica",
                "adsorbate": "methane",
                "adsorbate_molecular_weight": 16.04,
                "pressure": 2.0,
                "adsorbed_amount": 0.2,
            },
        ]
    )
    binary_mixture = pd.DataFrame(
        [
            {
                "name": "binary-1",
                "pressure_units": "kPa",
                "adsorption_units": "mol/kg",
                "temperature": 300.0,
                "adsorbent_name": "carbon",
                "compound_1": "nitrogen",
                "compound_2": "argon",
                "compound_1_pressure": 1.0,
                "compound_1_adsorption": 0.3,
                "compound_2_pressure": 2.0,
                "compound_2_adsorption": 0.4,
            },
            {
                "name": "binary-1",
                "pressure_units": "kPa",
                "adsorption_units": "mol/kg",
                "temperature": 300.0,
                "adsorbent_name": "carbon",
                "compound_1": "nitrogen",
                "compound_2": "argon",
                "compound_1_pressure": 2.0,
                "compound_1_adsorption": 0.5,
                "compound_2_pressure": 3.0,
                "compound_2_adsorption": 0.6,
            },
        ]
    )

    experiments = mapper.experiment_records(single_component, binary_mixture)
    mapper.attach_source_identities(
        experiments,
        [
            {
                "filename": "single-1",
                "DOI": "10.1000/single",
                "adsorbent": {"hashkey": "host-1", "name": "Silica"},
                "adsorbates": [{"name": "Methane", "InChIKey": "A" * 27}],
            },
            {
                "filename": "binary-1",
                "adsorbent": {"hashkey": "host-2", "name": "Carbon"},
                "adsorbates": [
                    {"name": "Nitrogen", "InChIKey": "B" * 27},
                    {"name": "Argon", "InChIKey": "C" * 27},
                ],
            },
        ],
    )
    repository.save_experiments(experiments)

    counts = repository.count_nist_rows()
    assert counts == {
        "experiments_count": 2,
        "single_component_rows": 6,
        "binary_mixture_rows": 1,
        "guest_rows": 3,
        "host_rows": 2,
    }

    adsorption, guests, hosts = repository.load_adsorption_datasets()
    assert len(adsorption) == 6
    assert {"pressure", "adsorbed_amount"}.issubset(adsorption.columns)
    assert set(guests["name"]) == {"methane", "nitrogen", "argon"}
    assert set(hosts["name"]) == {"silica", "carbon"}
    # Experiment species and materials reuse the NIST catalogue identities.
    with repository.database.session_factory() as session:
        assert session.scalar(select(func.count(Adsorbate.id))) == 3
        assert session.scalar(select(func.count(Adsorbent.id))) == 2
        assert session.scalars(select(Reference.doi)).all() == ["10.1000/single"]
    assert repository.count_local_records_by_category() == {
        "experiments": 2,
        "guest": 3,
        "host": 2,
    }
    assert repository.list_adsorbate_inchi_keys() == {
        "a" * 27,
        "b" * 27,
        "c" * 27,
    }
    assert repository.list_adsorbent_hash_keys() == {"host-1", "host-2"}

###############################################################################
def test_nist_category_counts_include_standalone_reference_records(
    tmp_path: Path,
) -> None:
    repository = build_nist_repository(tmp_path / "nist.db")
    mapper = NISTCanonicalMapper()
    repository.save_materials(
        mapper.material_records(
            pd.DataFrame([{"name": "methane", "InChIKey": "A" * 27}]),
            "adsorbate",
        ),
        mapper.material_records(
            pd.DataFrame([{"name": "silica", "hashkey": "host-1"}]),
            "adsorbent",
        ),
    )

    assert repository.count_local_records_by_category() == {
        "experiments": 0,
        "guest": 1,
        "host": 1,
    }
    assert repository.list_adsorbate_inchi_keys() == {"a" * 27}
    assert repository.list_adsorbent_hash_keys() == {"host-1"}

###############################################################################
def test_nist_placeholder_identifier_keeps_identity_without_fake_inchikey(
    tmp_path: Path,
) -> None:
    repository = build_nist_repository(tmp_path / "nist.db")
    mapper = NISTCanonicalMapper()
    placeholder = "NIST-ISODB-95AA5E6F8FA9B6AE19EC"
    repository.save_materials(
        mapper.material_records(
            pd.DataFrame([{"name": "Coal gas", "InChIKey": placeholder}]),
            "adsorbate",
        ),
        [],
    )
    experiments = mapper.experiment_records(
        pd.DataFrame(
            [
                {
                    "name": "placeholder-1",
                    "pressure_units": "kPa",
                    "adsorption_units": "mol/kg",
                    "temperature": 298.15,
                    "adsorbent": "carbon",
                    "adsorbate": "coal gas",
                    "adsorbate_molecular_weight": None,
                    "pressure": pressure,
                    "adsorbed_amount": uptake,
                }
                for pressure, uptake in ((1.0, 0.1), (2.0, 0.2))
            ]
        ),
        None,
    )
    mapper.attach_source_identities(
        experiments,
        [
            {
                "filename": "placeholder-1",
                "adsorbent": {"hashkey": "host-9", "name": "Carbon"},
                "adsorbates": [{"name": "Coal gas", "InChIKey": placeholder}],
            }
        ],
    )
    repository.save_experiments(experiments)

    with repository.database.session_factory() as session:
        rows = session.execute(select(Adsorbate.name, Adsorbate.inchi_key)).all()
    assert rows == [("Coal gas", None)]
    assert repository.list_adsorbate_inchi_keys() == {placeholder.casefold()}

###############################################################################
def test_nist_mapper_skips_experiment_with_unsupported_uptake_unit(caplog) -> None:
    mapper = NISTCanonicalMapper()
    single_component = pd.DataFrame(
        [
            {
                "name": "unsupported-volume",
                "pressure_units": "bar",
                "adsorption_units": "% Volume Adsorbed",
                "temperature": 77.0,
                "adsorbent": "CuBTC",
                "adsorbate": "nitrogen",
                "pressure": 0.1,
                "adsorbed_amount": 1.0,
            },
            {
                "name": "supported-molar",
                "pressure_units": "bar",
                "adsorption_units": "mmol/g",
                "temperature": 77.0,
                "adsorbent": "CuBTC",
                "adsorbate": "nitrogen",
                "pressure": 0.1,
                "adsorbed_amount": 1.0,
            },
        ]
    )

    records = mapper.experiment_records(single_component, None)

    assert [record["external_key"] for record in records] == ["supported-molar"]
    assert "unsupported-volume" in caplog.text
