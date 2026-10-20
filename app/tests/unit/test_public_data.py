# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
import gzip
from datetime import datetime, timezone

import httpx
import pytest
from fastapi import APIRouter, HTTPException
from sqlalchemy import event, func, select
from sqlalchemy.exc import IntegrityError

from server.api.public_data import PublicDataEndpoint
from server.configurations.settings import DatabaseConfig
from server.domain.public_data import StructureMaterialLinkRequest
from server.services.providers.cod import CODProvider
from server.services.providers.pubchem import PubChemProvider
from server.services.providers.public_data import (
    ProviderCapability,
    ProviderHealth,
    ProviderRateLimitError,
    ProviderUnavailableError,
    PublicDataProvider,
    RequestBudget,
)
from server.services.data.public_data import PublicDataService
from server.repositories.database.manager import DatabaseManager
from server.repositories.public_data import PublicDataRepository
from server.repositories.schemas import Base
from server.repositories.schemas.models import (
    Adsorbate,
    Adsorbent,
    Dataset,
    Isotherm,
    IsothermComponent,
    Observation,
)
from server.repositories.schemas.public_data import (
    DataSource,
    Reference,
    SourceRecord,
    SourceRecordReference,
    Structure,
    StructureSourceRecord,
)

from server.services.providers import PROVIDER_CLASSES

PUBLIC_SOURCE_DEFINITIONS = [provider.source_definition() for provider in PROVIDER_CLASSES]

###############################################################################
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

###############################################################################
def _count_selects(manager: DatabaseManager, operation) -> int:  # type: ignore[no-untyped-def]
    statements: list[str] = []

    def capture_selects(
        conn, cursor, statement, parameters, context, executemany  # type: ignore[no-untyped-def]
    ) -> None:
        del conn, cursor, parameters, context, executemany
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    event.listen(manager.engine, "before_cursor_execute", capture_selects)
    try:
        operation()
    finally:
        event.remove(manager.engine, "before_cursor_execute", capture_selects)
    return len(statements)


def test_structure_link_endpoint_maps_conflicts_to_http_409() -> None:
    class ConflictingService:
        def set_structure_material(self, structure_id: int, adsorbent_id: int | None) -> object:
            del structure_id, adsorbent_id
            raise ValueError("The structure is linked to another material; unlink it first.")

    endpoint = PublicDataEndpoint(
        router=APIRouter(),
        service=ConflictingService(),  # type: ignore[arg-type]
        sync=object(),  # type: ignore[arg-type]
    )

    with pytest.raises(HTTPException) as error:
        endpoint.set_structure_material(1, StructureMaterialLinkRequest(adsorbent_id=2))

    assert error.value.status_code == 409

###############################################################################
def _seed_listing_rows(repository: PublicDataRepository) -> int:
    with repository.database.transaction() as session:
        dataset = Dataset(name="Public data performance", source="nist")
        session.add(dataset)
        session.flush()
        first_structure_id = 0
        for index in range(2):
            material = Adsorbent(
                key=f"listing-material-{index}",
                name=f"Listing material {index}",
            )
            adsorbate = Adsorbate(
                key=f"listing-adsorbate-{index}",
                name=f"Listing adsorbate {index}",
            )
            session.add_all([material, adsorbate])
            session.flush()
            isotherm = Isotherm(
                dataset_id=dataset.id,
                external_key=f"listing-isotherm-{index}",
                name=f"Listing isotherm {index}",
                adsorbent_id=material.id,
                temperature_original=298.15,
                temperature_original_unit="K",
                temperature_k=298.15,
                pressure_basis="absolute",
            )
            session.add(isotherm)
            session.flush()
            component = IsothermComponent(
                isotherm_id=isotherm.id,
                position=1,
                adsorbate_id=adsorbate.id,
                mole_fraction=1.0,
            )
            session.add(component)
            session.flush()
            session.add(
                Observation(
                    isotherm_id=isotherm.id,
                    component_id=component.id,
                    sequence_index=0,
                    pressure_original=1.0,
                    pressure_original_unit="bar",
                    pressure_canonical=100_000.0,
                    pressure_canonical_unit="Pa",
                    uptake_original=1.0,
                    uptake_original_unit="mol/kg",
                    uptake_mol_kg=1.0,
                )
            )
            structure = Structure(
                adsorbent_id=material.id,
                name=f"Listing structure {index}",
                formula="C",
                format="cif",
                content=f"data_listing_{index}",
                content_sha256=f"{index + 1:064d}",
                has_coordinates=False,
            )
            session.add(structure)
            session.flush()
            if index == 0:
                first_structure_id = structure.id

    for index in range(2):
        with repository.database.session_factory() as session:
            material_id = session.scalar(
                select(Adsorbent.id).where(Adsorbent.key == f"listing-material-{index}")
            )
            adsorbate_id = session.scalar(
                select(Adsorbate.id).where(Adsorbate.key == f"listing-adsorbate-{index}")
            )
        assert material_id is not None
        assert adsorbate_id is not None
        repository.link_adsorbent_record(
            source_key="nist",
            adsorbent_id=material_id,
            external_id=f"material-{index}",
        )
        repository.link_adsorbate_record(
            source_key="nist",
            adsorbate_id=adsorbate_id,
            external_id=f"adsorbate-{index}",
        )
        repository.link_isotherm_record(
            source_key="nist",
            dataset_id=1,
            external_id=f"listing-isotherm-{index}",
        )
    return first_structure_id

###############################################################################
def test_overview_counts_use_one_local_query() -> None:
    manager = _manager()
    try:
        repository = PublicDataRepository(manager)
        repository.ensure_sources(PUBLIC_SOURCE_DEFINITIONS)
        _seed_listing_rows(repository)

        assert _count_selects(manager, repository.overview_counts) == 1
        assert repository.overview_counts() == {
            "adsorption_records": 2,
            "materials": 2,
            "chemicals": 2,
            # Seeded structures have no provider record, so they are not public data.
            "structures": 0,
            "active_sync_runs": 0,
            "failed_sync_runs": 0,
            "latest_successful_sync_at": None,
        }
    finally:
        manager.dispose()

###############################################################################
def test_provider_registry_and_source_identity_constraints() -> None:
    manager = _manager()
    try:
        repository = PublicDataRepository(manager)
        repository.ensure_sources(PUBLIC_SOURCE_DEFINITIONS)
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

###############################################################################
def test_source_registry_uses_cached_health_without_probing_providers() -> None:
    class ProbeCountingProvider(PublicDataProvider):
        key = "nist"
        name = "NIST test provider"
        description = "Provider used to verify local source listing."
        homepage_url = "https://example.org"
        license_name = None
        license_url = None
        terms_url = None
        capabilities = (ProviderCapability.ADSORPTION,)

        def __init__(self) -> None:
            self.health_checks = 0

        async def health(self) -> ProviderHealth:
            self.health_checks += 1
            return ProviderHealth(
                status="available",
                detail=None,
                checked_at=datetime.now(timezone.utc),
            )

    class FailingProvider(ProbeCountingProvider):
        key = "cod"
        name = "COD test provider"

        async def health(self) -> ProviderHealth:
            self.health_checks += 1
            raise RuntimeError("provider timeout")

    manager = _manager()
    try:
        repository = PublicDataRepository(manager)
        repository.ensure_sources(PUBLIC_SOURCE_DEFINITIONS)
        provider = ProbeCountingProvider()
        failing_provider = FailingProvider()
        service = PublicDataService(repository=repository, providers=[provider, failing_provider])

        registry = service.list_sources()

        assert provider.health_checks == 0
        assert failing_provider.health_checks == 0
        assert {source.key for source in registry.sources} == {"nist", "cod"}
        assert all(source.health_status == "unknown" for source in registry.sources)

        assert provider.health_checks == 0
        assert failing_provider.health_checks == 0
    finally:
        manager.dispose()

###############################################################################
def test_public_data_health_and_durable_sync_run_state_persist() -> None:
    manager = _manager()
    try:
        repository = PublicDataRepository(manager)
        repository.ensure_sources(PUBLIC_SOURCE_DEFINITIONS)
        checked_at = datetime.now(timezone.utc)
        repository.update_source_health(
            "nist", status="available", detail=None, checked_at=checked_at
        )

        run = repository.create_sync_run(
            source_key="nist",
            operation="nist_category",
            parameters={"category": "experiments", "fraction": 0.25},
        )
        run_id = str(run["id"])
        claimed = repository.claim_next_sync_run()
        assert claimed is not None and claimed["id"] == run_id
        assert claimed["status"] == "running"
        repository.update_sync_run(
            run_id,
            progress=40.0,
            processed_count=20,
            imported_count=17,
            skipped_count=2,
            failed_count=1,
            checkpoint={"category": "experiments", "last_external_id": "nist-20"},
        )
        repository.add_sync_failure(
            sync_run_id=run_id,
            source_key="nist",
            external_id="nist-20",
            reason="NIST record request failed.",
            retry_eligible_at=checked_at,
        )

        sources = {source["key"]: source for source in repository.source_rows()}
        assert sources["nist"]["health_status"] == "available"
        assert sources["nist"]["last_health_checked_at"] == checked_at
        failures, total = repository.list_sync_failures(run_id)
        assert total == 1
        assert failures[0]["external_id"] == "nist-20"

        assert repository.mark_unfinished_sync_runs(resume=True) == 1
        resumed = repository.get_sync_run(run_id)
        assert resumed["status"] == "queued"
        assert resumed["checkpoint"]["last_external_id"] == "nist-20"
        assert resumed["imported_count"] == 17

        repository.claim_next_sync_run()
        cancellation = repository.request_sync_cancel(run_id)
        assert cancellation["status"] == "running"
        assert cancellation["cancel_requested"] is True
        assert repository.mark_unfinished_sync_runs(resume=False) == 1
        assert repository.get_sync_run(run_id)["status"] == "interrupted"

        cancelled = repository.create_sync_run(
            source_key="cod", operation="cod_import", parameters={"cod_id": "9012345"}
        )
        assert repository.request_sync_cancel(str(cancelled["id"]))["status"] == "cancelled"
    finally:
        manager.dispose()

###############################################################################
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

###############################################################################
@pytest.mark.parametrize("failed_endpoint", ["synonyms", "conformer"])
def test_pubchem_resolution_keeps_primary_record_when_secondary_endpoint_fails(
    monkeypatch, failed_endpoint: str
) -> None:  # type: ignore[no-untyped-def]
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
                                "MolecularFormula": "CH4",
                                "MolecularWeight": "16.043",
                                "InChIKey": "VNWKTOKETHGBQD-UHFFFAOYSA-N",
                            }
                        ]
                    }
                },
            )
        if "/synonyms/" in url:
            if failed_endpoint == "synonyms":
                raise ProviderUnavailableError("simulated secondary endpoint failure")
            return httpx.Response(
                200,
                json={
                    "InformationList": {
                        "Information": [{"CID": 297, "Synonym": ["Marsh gas"]}]
                    }
                },
            )
        if failed_endpoint == "conformer":
            raise ProviderUnavailableError("simulated secondary endpoint failure")
        return httpx.Response(200, text="3D SDF")

    monkeypatch.setattr(provider, "_request", fake_request)
    payload = asyncio.run(provider.resolve("methane"))

    assert payload["cid"] == "297"
    assert payload["formula"] == "CH4"
    expected_synonyms = [] if failed_endpoint == "synonyms" else ["Marsh gas"]
    assert payload["synonyms"] == expected_synonyms
    assert (payload["conformer_3d_url"] is None) is (failed_endpoint == "conformer")

###############################################################################
def test_retrying_provider_reuses_and_closes_its_http_client(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    created: list[object] = []

    class FakeStream:
        def __init__(self, response: httpx.Response) -> None:
            self.response = response

        async def __aenter__(self):  # type: ignore[no-untyped-def]
            return self.response

        async def __aexit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

    ###############################################################################
    class FakeClient:

        # -------------------------------------------------------------------------
        def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            self.closed = False
            self.requests = 0
            created.append(self)

        # -------------------------------------------------------------------------
        def stream(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            del kwargs
            self.requests += 1
            return FakeStream(
                httpx.Response(200, request=httpx.Request(args[0], args[1]), text="{}")
            )

        # -------------------------------------------------------------------------
        async def aclose(self) -> None:
            self.closed = True

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    provider = CODProvider(
        request_timeout_seconds=1.0,
        retry_attempts=1,
        max_interactive_results=10,
    )

    async def exercise() -> None:
        await provider._request("GET", "https://www.crystallography.net/cod/one")
        await provider._request("GET", "https://www.crystallography.net/cod/two")
        assert len(created) == 1
        assert created[0].requests == 2  # type: ignore[attr-defined]
        await provider.close()
        assert created[0].closed is True  # type: ignore[attr-defined]

    asyncio.run(exercise())

###############################################################################
def test_cod_provider_backs_off_after_burst_forbidden_response(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    responses = [
        httpx.Response(
            403,
            request=httpx.Request("GET", "https://www.crystallography.net/cod/retry"),
        ),
        httpx.Response(
            200,
            request=httpx.Request("GET", "https://www.crystallography.net/cod/retry"),
            text="{}",
        ),
    ]
    sleeps: list[float] = []

    class FakeStream:
        def __init__(self, response: httpx.Response) -> None:
            self.response = response

        async def __aenter__(self):  # type: ignore[no-untyped-def]
            return self.response

        async def __aexit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

    ###############################################################################
    class FakeClient:

        # -------------------------------------------------------------------------
        def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            del kwargs

        # -------------------------------------------------------------------------
        def stream(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            del args, kwargs
            return FakeStream(responses.pop(0))

        # -------------------------------------------------------------------------
        async def aclose(self) -> None:
            return None

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    provider = CODProvider(
        request_timeout_seconds=1.0,
        retry_attempts=2,
        max_interactive_results=10,
    )
    provider.request_budget = RequestBudget(minimum_interval_seconds=0.0)

    async def exercise() -> None:
        response = await provider._request(
            "GET", "https://www.crystallography.net/cod/retry"
        )
        assert response.status_code == 200
        await provider.close()

    asyncio.run(exercise())
    assert responses == []
    assert sleeps == [5.0]

###############################################################################
def test_retrying_provider_returns_decoded_gzip_body() -> None:
    payload = b'{"PropertyTable": {"Properties": []}}'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=gzip.compress(payload),
            headers={"Content-Encoding": "gzip", "Content-Type": "application/json"},
            request=request,
        )

    provider = PubChemProvider()

    async def exercise() -> None:
        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider._client_loop = asyncio.get_running_loop()
        response = await provider._request(
            "GET", f"{provider.base_url}/compound/cid/1/property/Title/JSON"
        )
        assert response.content == payload
        assert response.json() == {"PropertyTable": {"Properties": []}}
        await provider.close()

    asyncio.run(exercise())

###############################################################################
@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (ProviderRateLimitError("rate limited"), 429),
        (ProviderUnavailableError("unavailable"), 503),
    ],
)
def test_public_provider_errors_keep_explicit_http_contract(
    error: ProviderUnavailableError,
    status_code: int,
) -> None:
    mapped = PublicDataEndpoint._provider_error(error)

    assert mapped.status_code == status_code
    assert mapped.detail == str(error)

###############################################################################
def test_pubchem_resolution_maps_successful_malformed_json_to_provider_error(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    provider = PubChemProvider(parallel_requests=1)

    async def fake_request(method: str, url: str, **kwargs):  # type: ignore[no-untyped-def]
        del method, url, kwargs
        return httpx.Response(200, text="{not valid json")

    monkeypatch.setattr(provider, "_request", fake_request)

    with pytest.raises(ProviderUnavailableError, match="malformed JSON"):
        asyncio.run(provider.resolve("methane"))

###############################################################################
def test_pubchem_empty_query_preserves_local_validation_error() -> None:
    provider = PubChemProvider(parallel_requests=1)

    with pytest.raises(ValueError, match="must not be empty"):
        asyncio.run(provider.resolve("  "))

###############################################################################
def test_pubchem_upsert_uses_strong_identity_and_does_not_merge_by_name() -> None:
    manager = _manager()
    try:
        repository = PublicDataRepository(manager)
        repository.ensure_sources(PUBLIC_SOURCE_DEFINITIONS)
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

###############################################################################
def _methane_payload(inchi_key: str = "VNWKTOKETHGBQD-UHFFFAOYSA-N") -> dict[str, object]:
    return {
        "cid": "297",
        "name": "Methane",
        "formula": "CH4",
        "molecular_weight": 16.043,
        "inchi_key": inchi_key,
        "synonyms": ["Methane", "CH4"],
        "descriptors": {},
        "elemental_composition": {"C": 1.0, "H": 4.0},
        "raw_metadata": {},
    }

###############################################################################
def test_pubchem_attach_enriches_selected_chemical_and_tracks_candidates() -> None:
    manager = _manager()
    try:
        repository = PublicDataRepository(manager)
        repository.ensure_sources(PUBLIC_SOURCE_DEFINITIONS)
        with manager.transaction() as session:
            session.add_all(
                [
                    Adsorbate(key="nist-methane", name="methane"),
                    Adsorbate(key="nist-argon", name="argon"),
                    Adsorbate(key="nist-xenon", name="xenon"),
                ]
            )
        with manager.session_factory() as session:
            ids = dict(session.execute(select(Adsorbate.key, Adsorbate.id)).all())
        repository.add_sync_failure(
            sync_run_id=repository.create_sync_run(
                source_key="pubchem", operation="pubchem_enrich", parameters={}
            )["id"],
            source_key="pubchem",
            external_id=f"adsorbate:{ids['nist-xenon']}",
            reason="xenon: not found",
        )

        attached = repository.upsert_pubchem_compound(
            _methane_payload(), ids["nist-methane"]
        )

        assert attached == ids["nist-methane"]
        with manager.session_factory() as session:
            methane = session.get(Adsorbate, attached)
            assert methane is not None
            assert (methane.name, methane.formula) == ("methane", "CH4")
            assert session.scalar(select(func.count(Adsorbate.id))) == 3
        assert repository.find_local_pubchem_adsorbate("CH4") == attached
        # Linked chemicals and permanent lookup failures are not retried.
        assert repository.pubchem_enrichment_candidates(10) == [
            {"id": ids["nist-argon"], "query": "argon"}
        ]
        with pytest.raises(ValueError, match="already linked"):
            repository.upsert_pubchem_compound(_methane_payload(), ids["nist-argon"])
    finally:
        manager.dispose()

###############################################################################
def test_navigation_filters_by_ids_and_structure_links_are_explicit() -> None:
    manager = _manager()
    try:
        repository = PublicDataRepository(manager)
        repository.ensure_sources(PUBLIC_SOURCE_DEFINITIONS)
        structure_id = _seed_listing_rows(repository)
        structure = repository.get_structure(structure_id)
        material_id = int(structure["material_id"])

        rows, total = repository.list_adsorption(
            page=1, page_size=25, material_id=material_id
        )
        assert total == 1
        assert rows[0]["material_id"] == material_id
        adsorbate_id = rows[0]["adsorbates"][0]["id"]
        _, by_adsorbate = repository.list_adsorption(
            page=1, page_size=25, adsorbate_id=adsorbate_id
        )
        assert by_adsorbate == 1

        material = repository.get_material(material_id)
        assert material["isotherm_count"] == 1
        assert [item["id"] for item in material["structures"]] == [structure_id]

        other_material = next(
            item["id"]
            for item in repository.list_materials(page=1, page_size=25)[0]
            if item["id"] != material_id
        )
        with pytest.raises(ValueError, match="unlink it first"):
            repository.set_structure_material(structure_id, other_material)
        repository.set_structure_material(structure_id, None)
        repository.set_structure_material(structure_id, other_material)
        _, linked = repository.list_structures(
            page=1, page_size=25, material_id=other_material
        )
        assert linked == 2
    finally:
        manager.dispose()

###############################################################################
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

###############################################################################
def test_cod_cif_validation_rejects_malformed_payloads() -> None:
    with pytest.raises(ProviderUnavailableError, match="valid CIF data block"):
        CODProvider.validate_cif({"has_coordinates": False}, "not CIF", [])

    with pytest.raises(ProviderUnavailableError, match="no valid atom-site rows"):
        CODProvider.validate_cif({"has_coordinates": True}, "data_test", [])

###############################################################################
def test_cod_search_maps_successful_malformed_json_to_provider_error(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    provider = CODProvider(
        request_timeout_seconds=1.0,
        retry_attempts=1,
        max_interactive_results=10,
    )

    async def fake_request(method: str, url: str, **kwargs):  # type: ignore[no-untyped-def]
        del method, url
        if kwargs["params"]["format"] == "count":
            return httpx.Response(200, text="1")
        return httpx.Response(200, text="{not valid json")

    monkeypatch.setattr(provider, "_request", fake_request)

    with pytest.raises(ProviderUnavailableError, match="malformed JSON"):
        asyncio.run(provider.search(text="silica"))

###############################################################################
def test_cod_search_validation_errors_remain_local_value_errors() -> None:
    provider = CODProvider(
        request_timeout_seconds=1.0,
        retry_attempts=1,
        max_interactive_results=10,
    )

    with pytest.raises(ValueError, match="at least three characters"):
        asyncio.run(provider.search(text="Si"))
    with pytest.raises(ValueError, match="Provide a COD ID"):
        asyncio.run(provider.search())

###############################################################################
def test_cod_reimport_without_material_preserves_existing_association() -> None:
    manager = _manager()
    try:
        repository = PublicDataRepository(manager)
        repository.ensure_sources(PUBLIC_SOURCE_DEFINITIONS)
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

###############################################################################
def test_public_data_list_queries_do_not_scale_with_page_size() -> None:
    manager = _manager()
    try:
        repository = PublicDataRepository(manager)
        repository.ensure_sources(PUBLIC_SOURCE_DEFINITIONS)
        _seed_listing_rows(repository)

        assert _count_selects(
            manager,
            lambda: repository.list_adsorption(page=1, page_size=2),
        ) <= 6
        assert _count_selects(
            manager,
            lambda: repository.list_materials(page=1, page_size=2),
        ) <= 5
        assert _count_selects(
            manager,
            lambda: repository.list_chemicals(page=1, page_size=2),
        ) <= 6
        assert _count_selects(
            manager,
            lambda: repository.list_structures(page=1, page_size=2),
        ) <= 6
    finally:
        manager.dispose()

###############################################################################
def test_structure_reference_matches_primary_provenance_record() -> None:
    manager = _manager()
    try:
        repository = PublicDataRepository(manager)
        repository.ensure_sources(PUBLIC_SOURCE_DEFINITIONS)
        structure_id = _seed_listing_rows(repository)
        with manager.transaction() as session:
            nist_id = session.scalar(select(DataSource.id).where(DataSource.key == "nist"))
            cod_id = session.scalar(select(DataSource.id).where(DataSource.key == "cod"))
            assert nist_id is not None
            assert cod_id is not None
            older = SourceRecord(
                source_id=cod_id,
                record_type="structure",
                external_id="older-structure",
                retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                raw_metadata={},
            )
            newer = SourceRecord(
                source_id=nist_id,
                record_type="structure",
                external_id="newer-structure",
                retrieved_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
                raw_metadata={},
            )
            session.add_all([older, newer])
            session.flush()
            session.add_all(
                [
                    StructureSourceRecord(
                        source_record_id=older.id,
                        structure_id=structure_id,
                    ),
                    StructureSourceRecord(
                        source_record_id=newer.id,
                        structure_id=structure_id,
                    ),
                ]
            )
            older_reference = Reference(doi="10.1000/older")
            newer_reference = Reference(doi="10.1000/newer")
            session.add_all([older_reference, newer_reference])
            session.flush()
            session.add_all(
                [
                    SourceRecordReference(
                        source_record_id=older.id,
                        reference_id=older_reference.id,
                    ),
                    SourceRecordReference(
                        source_record_id=newer.id,
                        reference_id=newer_reference.id,
                    ),
                ]
            )

        view = repository.get_structure(structure_id)
        assert view["source"] == "nist"
        assert view["external_id"] == "newer-structure"
        assert view["doi"] == "10.1000/newer"
    finally:
        manager.dispose()
