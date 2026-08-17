from __future__ import annotations

from pathlib import Path

import pandas as pd

from core_service.services.data.nist_mapper import NISTCanonicalMapper
from shared.common.settings import DatabaseSettings
from shared.repositories.database.manager import DatabaseManager
from shared.repositories.datasets import DatasetRepository
from shared.repositories.materials import MaterialRepository
from shared.repositories.nist import NISTRepository
from shared.repositories.schemas.models import Base


def build_nist_repository(path: Path) -> NISTRepository:
    settings = DatabaseSettings(
        embedded_database=True,
        engine=None,
        host=None,
        port=None,
        database_name=None,
        username=None,
        password=None,
        ssl=False,
        ssl_ca=None,
        connect_timeout=30,
        insert_batch_size=100,
        sqlite_path=str(path),
    )
    database = DatabaseManager(settings)
    Base.metadata.create_all(database.engine)
    datasets = DatasetRepository(database)
    return NISTRepository(
        database=database,
        datasets=datasets,
        materials=MaterialRepository(database),
    )


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

    repository.save_experiments(
        mapper.experiment_records(single_component, binary_mixture)
    )

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
