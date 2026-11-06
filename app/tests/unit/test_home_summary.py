# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path

from server.configurations.settings import DatabaseConfig
from server.repositories.database.manager import DatabaseManager
from server.repositories.home import HomeRepository
from server.repositories.materials import MaterialRepository
from server.repositories.public_data import PublicDataRepository
from server.repositories.schemas import Base
from server.services.providers import PROVIDER_CLASSES


###############################################################################
def test_home_summary_counts_stored_entities_and_represented_sources(
    tmp_path: Path,
) -> None:
    manager = DatabaseManager(
        DatabaseConfig(
            embedded_database=True,
            connect_timeout=5,
            sqlite_path=str(tmp_path / "home.db"),
        )
    )
    Base.metadata.create_all(manager.engine)
    try:
        home = HomeRepository(manager)
        empty = home.summary()
        assert empty["species"] == 0 and empty["sources"] == []

        public_data = PublicDataRepository(manager)
        public_data.ensure_sources([cls.source_definition() for cls in PROVIDER_CLASSES])
        public_data.upsert_pubchem_compound(
            {"cid": "297", "name": "Methane", "inchi_key": "VNWKTOKETHGBQD-UHFFFAOYSA-N"}
        )
        summary = home.summary()
        assert summary["species"] == 1
        assert [(s["key"], s["record_count"]) for s in summary["sources"]] == [("pubchem", 1)]
        assert summary["fitting_sessions"] == 0
        # The Sources tiles filter with source=any: every provider-backed species, none for a local-only one.
        materials = MaterialRepository(manager)
        materials.upsert_adsorbates([{"key": "local", "name": "Local gas"}])
        assert [row["name"] for row in materials.list_species(page=1, page_size=10, source="any")[0]] == ["Methane"]
        assert materials.list_species(page=1, page_size=10)[1] == 2
    finally:
        manager.dispose()
