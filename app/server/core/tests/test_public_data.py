from __future__ import annotations

import asyncio

import httpx
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from adsmod_common.config import DatabaseConfig
from adsmod_core.providers.cod import CODProvider
from adsmod_core.providers.pubchem import PubChemProvider
from adsmod_core.repositories.database.manager import DatabaseManager
from adsmod_core.repositories.public_data import PublicDataRepository
from adsmod_core.repositories.schemas import Base
from adsmod_core.repositories.schemas.models import Adsorbate, Adsorbent
from adsmod_core.repositories.schemas.public_data import DataSource, SourceRecord, Structure


def _manager() -> DatabaseManager:
    manager = DatabaseManager(
        DatabaseConfig(
            embedded_database=True,
            connect_timeout=5,
            insert_batch_size=100,
            sqlite_path=":memory:",
        )
    )
    Base.metadata.create_all(manager.engine)
    return manager


def test_provider_registry_and_source_identity_constraints() -> None:
    manager = _manager()
    try:
        repository = PublicDataRepository(manager)
        repository.ensure_sources()
        assert {row["key"] for row in repository.source_rows()} == {
            "nist",
            "pubchem",
            "cod",
        }

        with manager.transaction() as session:
            pubchem_id = session.scalar(select(DataSource.id).where(DataSource.key == "pubchem"))
            assert pubchem_id is not None
            session.add(
                SourceRecord(
                    source_id=pubchem_id,
                    record_type="chemical",
                    external_id="123",
                    source_url="https://pubchem.ncbi.nlm.nih.gov/compound/123",
                    raw_metadata={},
                )
            )
        with manager.transaction() as session:
            pubchem_id = session.scalar(select(DataSource.id).where(DataSource.key == "pubchem"))
            session.add(
                SourceRecord(
                    source_id=pubchem_id,
                    record_type="chemical",
                    external_id="123",
                    raw_metadata={},
                )
            )
            try:
                session.flush()
            except IntegrityError:
                session.rollback()
            else:
                raise AssertionError("duplicate source identity was accepted")
    finally:
        manager.dispose()


def test_pubchem_resolution_normalizes_properties_without_network(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    provider = PubChemProvider(parallel_requests=1)

    async def fake_request(method: str, url: str, **kwargs):  # type: ignore[no-untyped-def]
        del method, kwargs
        if "/property/" in url:
            return httpx.Response(
                200,
                json={
                    "PropertyTable": {
                        "Properties": [
                            {
                                "CID": 297,
                                "Title": "Methane",
                                "IUPACName": "methane",
                                "MolecularFormula": "CH4",
                                "MolecularWeight": "16.043",
                                "SMILES": "C",
                                "ConnectivitySMILES": "C",
                                "InChI": "InChI=1S/CH4/h1H4",
                                "InChIKey": "VNWKTOKETHGBQD-UHFFFAOYSA-N",
                                "TPSA": 0,
                                "HBondDonorCount": 0,
                                "HBondAcceptorCount": 0,
                            }
                        ]
                    }
                },
            )
        if "/synonyms/" in url:
            return httpx.Response(
                200,
                json={
                    "InformationList": {
                        "Information": [{"CID": 297, "Synonym": ["Methane", "Marsh gas"]}]
                    }
                },
            )
        return httpx.Response(200, text="3D SDF")

    monkeypatch.setattr(provider, "_request", fake_request)
    payload = asyncio.run(provider.resolve("methane"))

    assert payload["cid"] == "297"
    assert payload["inchi_key"] == "VNWKTOKETHGBQD-UHFFFAOYSA-N"
    assert payload["elemental_composition"] == {"C": 1.0, "H": 4.0}
    assert payload["descriptors"]["tpsa_angstrom2"] == 0.0
    assert payload["conformer_3d_url"] is not None


def test_pubchem_upsert_uses_strong_identity_and_does_not_merge_by_name() -> None:
    manager = _manager()
    try:
        repository = PublicDataRepository(manager)
        repository.ensure_sources()
        with manager.transaction() as session:
            session.add(
                Adsorbate(
                    key="upload-methane",
                    name="Methane",
                    inchi_key=None,
                )
            )

        pubchem_id = repository.upsert_pubchem_compound(
            {
                "cid": "297",
                "name": "Methane",
                "preferred_name": "methane",
                "formula": "CH4",
                "molecular_weight": 16.043,
                "smiles": "C",
                "connectivity_smiles": "C",
                "inchi": "InChI=1S/CH4/h1H4",
                "inchi_key": "VNWKTOKETHGBQD-UHFFFAOYSA-N",
                "synonyms": ["Methane"],
                "descriptors": {},
                "elemental_composition": {"C": 1.0, "H": 4.0},
                "source_url": "https://pubchem.ncbi.nlm.nih.gov/compound/297",
                "raw_metadata": {},
            }
        )
        with manager.session_factory() as session:
            records = session.scalars(select(Adsorbate).order_by(Adsorbate.id)).all()
            assert len(records) == 2
            assert records[1].id == pubchem_id
            assert records[1].inchi_key == "VNWKTOKETHGBQD-UHFFFAOYSA-N"
    finally:
        manager.dispose()


def test_cod_atom_parser_normalizes_fractional_coordinates() -> None:
    cif = """
data_test
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
C1 C 0.125 0.250 0.375 1.0
O1 O 0.500(2) 0.625 0.750 0.5
"""
    atoms = CODProvider.parse_atoms(cif)
    assert atoms == [
        {
            "sequence_index": 0,
            "label": "C1",
            "element": "C",
            "fractional_x": 0.125,
            "fractional_y": 0.25,
            "fractional_z": 0.375,
            "occupancy": 1.0,
        },
        {
            "sequence_index": 1,
            "label": "O1",
            "element": "O",
            "fractional_x": 0.5,
            "fractional_y": 0.625,
            "fractional_z": 0.75,
            "occupancy": 0.5,
        },
    ]


def test_cod_reimport_without_material_preserves_existing_association() -> None:
    manager = _manager()
    try:
        repository = PublicDataRepository(manager)
        repository.ensure_sources()
        with manager.transaction() as session:
            adsorbent = Adsorbent(
                key="test-silica",
                name="Test silica",
                normalized_name="test silica",
            )
            session.add(adsorbent)
            session.flush()
            adsorbent_id = adsorbent.id

        metadata = {
            "cod_id": "4502440",
            "name": "Test silica",
            "formula": "SiO2",
        }
        structure_id = repository.upsert_cod_structure(
            metadata=metadata,
            cif_text="data_test",
            atoms=[],
            adsorbent_id=adsorbent_id,
        )
        assert (
            repository.upsert_cod_structure(
                metadata=metadata,
                cif_text="data_test",
                atoms=[],
                adsorbent_id=None,
            )
            == structure_id
        )

        with manager.session_factory() as session:
            structure = session.get(Structure, structure_id)
            assert structure is not None
            assert structure.adsorbent_id == adsorbent_id
    finally:
        manager.dispose()
