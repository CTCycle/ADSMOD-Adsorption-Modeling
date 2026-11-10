# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from server.configurations.settings import DatabaseConfig
from server.repositories.database.manager import DatabaseManager
from server.repositories.datasets import resolve_adsorbate, resolve_adsorbent
from server.repositories.identity import (
    IdentityRepository,
    IdentityResolver,
    MergeRefusedError,
    raise_review,
)
from server.repositories.schemas import Base
from server.repositories.schemas.models import (
    Adsorbate,
    Adsorbent,
    Dataset,
    Isotherm,
    IsothermComponent,
)
from server.repositories.schemas.public_data import (
    AdsorbateSynonym,
    AdsorbentSynonym,
    IdentityReview,
    Structure,
)
from server.services.data.public_data import PublicDataService
from server.services.providers.pubchem import CandidateSet, PubChemProvider
from server.common.identity import (
    clean_name,
    fuzzy_score,
    is_generic_adsorbent,
    is_inchi_key,
    looks_like_formula,
    names_compatible,
    same_except_separators,
)


###############################################################################
@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("2,3 DhaTph", "2,3-DhaTph"),
        ("4PE-1P COF", "4PE-1P-COF"),
        ("AC F", "ACF"),
        ("Beulah Zap Coal", "Beulah-zap coal"),
        ("BFTB-BCTA COF", "BFTB�BCTA-COF"),
        ("BFTB-BFTB COF", "BFTB�BFTB-COF"),
        ("BFTB-PyTA COF", "BFTB�PyTA-COF"),
        ("CarboTech D 55/1.5", "CarboTech D55/1.5"),
        ("Co-BTC", "CoBTC"),
        ("COF-366 -(OMe)2-Co", "COF-366-(OMe)2-Co"),
        ("COF-LZU-1", "COF-LZU1"),
        ("N-decane", "n decane"),
    ],
)
def test_separator_variants_are_comparable(left: str, right: str) -> None:
    assert same_except_separators(left, right)


###############################################################################
@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("AgI/UiO-66 (.1)", "AgI/UiO-66 (1)"),
        ("Chlorite, 1.5nm", "Chlorite, 15nm"),
        ("Cd-MOF-1", "CDMOF-1"),
        ("Co/Cu-BTC", "Co@CuBTC"),
        ("o-xylene", "p-xylene"),
        ("1,2-dichlorobenzene", "1,3-dichlorobenzene"),
        ("cis-2-butene", "trans-2-butene"),
        ("Eu(III)", "Eu(II)"),
        ("ZIF-8", "ZIF-8 (synthetic)"),
        ("MIL-101(Cr)", "MIL-101(Fe)"),
        ("Coal YZG with 1.48% humidity", "Coal YZG with 2.1% humidity"),
        ("UiO-66", "UiO-67"),
    ],
)
def test_distinct_entities_are_never_compatible(left: str, right: str) -> None:
    assert not same_except_separators(left, right)
    assert fuzzy_score(left, right) == 0.0 or not names_compatible(left, right)


###############################################################################
@pytest.fixture
def session() -> Iterator[Session]:
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with Session(engine) as active:
        yield active
    engine.dispose()


###############################################################################
def _adsorbate(session: Session, name: str, inchi_key: str | None = None) -> Adsorbate:
    row = Adsorbate(
        key=f"adsorbate:{name}",
        name=name,
        inchi_key=inchi_key,
        identity_status="resolved" if inchi_key else "provisional",
    )
    session.add(row)
    session.flush()
    return row


###############################################################################
def _adsorbent(session: Session, name: str, external: str | None = None) -> Adsorbent:
    row = Adsorbent(
        key=f"adsorbent:{name}",
        name=name,
        external_identifier=external,
        identity_status="resolved" if external else "provisional",
    )
    session.add(row)
    session.flush()
    return row


###############################################################################
def _species(session: Session, record: dict[str, Any]) -> Adsorbate:
    return resolve_adsorbate(session, IdentityResolver(session), record)


###############################################################################
def _material(
    session: Session, record: dict[str, Any], dataset_id: int | None = 1
) -> Adsorbent:
    return resolve_adsorbent(
        session, IdentityResolver(session, dataset_id=dataset_id), record
    )


###############################################################################
def _reviews(session: Session) -> list[IdentityReview]:
    return list(session.scalars(select(IdentityReview)))


CO2_KEY = "CURLTUGMZLYLDI-UHFFFAOYSA-N"
CO_KEY = "UGFAIRIUMAVXCW-UHFFFAOYSA-N"


###############################################################################
@pytest.mark.parametrize("name", ["CO2", "Carbon dioxide", "carbon  dioxide"])
def test_formula_and_name_reach_one_species(session: Session, name: str) -> None:
    stored = _adsorbate(session, "Carbon Dioxide", CO2_KEY)
    assert _species(session, {"name": name}).id == stored.id


###############################################################################
def test_uppercase_formula_is_never_read_as_an_element(session: Session) -> None:
    cobalt = _adsorbate(session, "Cobalt", "GUTLYIVDDKVIGB-UHFFFAOYSA-N")
    monoxide = _adsorbate(session, "Carbon monoxide", CO_KEY)
    session.add(
        AdsorbateSynonym(
            adsorbate_id=cobalt.id,
            origin="pubchem",
            synonym="Co",
            normalized_synonym="co",
        )
    )
    session.flush()
    assert _species(session, {"name": "CO"}).id == monoxide.id
    assert _species(session, {"name": "Co"}).id == cobalt.id


###############################################################################
def test_isomeric_formula_creates_provisional_row(session: Session) -> None:
    _adsorbate(session, "N-butane", "IJDNQMDRQITEOD-UHFFFAOYSA-N")
    created = _species(session, {"name": "C4H10"})
    assert created.identity_status == "provisional"
    assert created.inchi_key is None


###############################################################################
def test_alias_link_preserves_received_name(session: Session) -> None:
    stored = _adsorbate(session, "Carbon Dioxide", CO2_KEY)
    session.add(
        AdsorbateSynonym(
            adsorbate_id=stored.id,
            origin="nist",
            synonym="Carbonic anhydride",
            normalized_synonym="carbonic anhydride",
        )
    )
    session.flush()
    assert _species(session, {"name": "CARBONIC ANHYDRIDE"}).id == stored.id


###############################################################################
def test_conflicting_inchi_key_is_flagged_not_merged(session: Session) -> None:
    stored = _adsorbate(session, "Methyl Orange", "STZCRXQWRGQSJD-GEEYTBSJSA-M")
    created = _species(
        session,
        {"name": "Methyl Orange", "inchi_key": "KZXBKUXFQJRNCC-UHFFFAOYSA-M"},
    )
    assert created.id != stored.id
    review = _reviews(session)[0]
    assert (review.reason, review.candidate_id) == ("identifier_conflict", stored.id)


###############################################################################
def test_several_alias_owners_are_flagged(session: Session) -> None:
    first = _adsorbate(session, "Alpha", CO2_KEY)
    second = _adsorbate(session, "Beta", CO_KEY)
    for owner in (first, second):
        session.add(
            AdsorbateSynonym(
                adsorbate_id=owner.id,
                origin="pubchem",
                synonym="Shared name",
                normalized_synonym="shared name",
            )
        )
    session.flush()
    created = _species(session, {"name": "Shared name"})
    assert created.id not in {first.id, second.id}
    assert {item.reason for item in _reviews(session)} == {"ambiguous_alias"}


###############################################################################
@pytest.mark.parametrize("name", ["Cu-BTC", "MOF-199", "Basolite C300"])
def test_curated_material_names_link(session: Session, name: str) -> None:
    stored = _adsorbent(session, "HKUST-1", "hash-hkust")
    assert _material(session, {"name": name}).id == stored.id


###############################################################################
def test_material_separator_variant_is_only_proposed(session: Session) -> None:
    stored = _adsorbent(session, "COF-LZU1", "hash-cof")
    created = _material(session, {"name": "COF-LZU-1"})
    assert created.id != stored.id
    review = _reviews(session)[0]
    assert (review.reason, review.candidate_id) == ("separator_variant", stored.id)


###############################################################################
def test_distinct_materials_raise_no_proposal(session: Session) -> None:
    _adsorbent(session, "Chlorite, 1.5nm", "hash-chlorite")
    _material(session, {"name": "Chlorite, 15nm"})
    assert _reviews(session) == []


###############################################################################
def test_generic_materials_stay_separate_per_dataset(session: Session) -> None:
    for dataset_id in (1, 2):
        session.add(Dataset(id=dataset_id, name=f"d{dataset_id}", source="uploaded"))
    session.flush()
    first = _material(session, {"name": "Activated carbon"}, 1)
    again = _material(session, {"name": "activated carbon"}, 1)
    second = _material(session, {"name": "Activated carbon"}, 2)
    assert first.id == again.id != second.id
    assert (first.identity_status, first.scope_dataset_id) == ("generic", 1)


###############################################################################
def test_rejected_pair_is_not_raised_again(session: Session) -> None:
    arguments = {
        "kind": "adsorbent",
        "name": "x",
        "reason": "fuzzy",
        "entity_id": 1,
        "candidate_id": 2,
    }
    assert raise_review(session, **arguments)
    _reviews(session)[0].status = "rejected"
    assert not raise_review(session, **{**arguments, "entity_id": 2, "candidate_id": 1})


###############################################################################
def test_prefetch_query_count_does_not_grow_with_batch(session: Session) -> None:
    for number in range(40):
        _adsorbate(session, f"Compound {number}")
        _adsorbent(session, f"Host {number}")
    session.commit()
    counts = [_ingest_queries(session, size) for size in (5, 40)]
    assert counts[0] == counts[1]


###############################################################################
def _ingest_queries(session: Session, size: int) -> int:
    resolver = IdentityResolver(session, dataset_id=1)
    experiments = [
        {
            "adsorbent": {"name": f"Host {number}"},
            "adsorbates": [{"name": f"Compound {number}"}],
        }
        for number in range(size)
    ]
    statements: list[str] = []

    def record(_conn: Any, _cursor: Any, statement: str, *_rest: Any) -> None:
        statements.append(statement)

    engine = session.get_bind()
    event.listen(engine, "before_cursor_execute", record)
    try:
        resolver.prefetch(experiments)
        for experiment in experiments:
            resolve_adsorbent(session, resolver, experiment["adsorbent"])
            resolve_adsorbate(session, resolver, experiment["adsorbates"][0])
    finally:
        event.remove(engine, "before_cursor_execute", record)
    return len(statements)


###############################################################################
@pytest.fixture
def database(tmp_path: Path) -> Iterator[DatabaseManager]:
    manager = DatabaseManager(
        DatabaseConfig(
            embedded_database=True,
            connect_timeout=5,
            sqlite_path=str(tmp_path / "identity.db"),
        )
    )
    Base.metadata.create_all(manager.engine)
    yield manager
    manager.dispose()


###############################################################################
def _experiment(
    session: Session, external_key: str, adsorbent_id: int, adsorbate_ids: list[int]
) -> Isotherm:
    dataset = session.scalar(select(Dataset)) or Dataset(
        name="d", normalized_name="d", source="uploaded"
    )
    session.add(dataset)
    session.flush()
    isotherm = Isotherm(
        dataset_id=dataset.id,
        external_key=external_key,
        name=external_key,
        adsorbent_id=adsorbent_id,
        temperature_original=298.0,
        temperature_original_unit="K",
        temperature_k=298.0,
        pressure_basis="absolute",
    )
    session.add(isotherm)
    session.flush()
    for position, adsorbate_id in enumerate(adsorbate_ids, start=1):
        session.add(
            IsothermComponent(
                isotherm_id=isotherm.id,
                position=position,
                adsorbate_id=adsorbate_id,
                source_name=f"received-{position}",
            )
        )
    session.flush()
    return isotherm


###############################################################################
def _open_pair(session: Session, kind: str, entity: Any, candidate: Any) -> int:
    raise_review(
        session,
        kind=kind,
        name=entity.name,
        reason="separator_variant",
        entity_id=entity.id,
        candidate_id=candidate.id,
    )
    return session.scalars(select(IdentityReview.id)).all()[-1]


###############################################################################
def test_merge_moves_every_reference_and_keeps_received_names(
    database: DatabaseManager,
) -> None:
    with database.transaction() as session:
        keep = _adsorbent(session, "COF-LZU1", "hash-keep")
        remove = _adsorbent(session, "COF-LZU-1", "hash-remove")
        species = _adsorbate(session, "Methane")
        isotherm = _experiment(session, "e1", remove.id, [species.id])
        session.add(
            Structure(
                adsorbent_id=remove.id,
                format="cif",
                content="data_x",
                content_sha256="0" * 64,
            )
        )
        session.flush()
        review_id = _open_pair(session, "adsorbent", remove, keep)
        keep_id, remove_id, isotherm_id = keep.id, remove.id, isotherm.id

    IdentityRepository(database).merge(review_id, keep_id)

    with database.session_factory() as session:
        assert session.get(Adsorbent, remove_id) is None
        assert session.get(Isotherm, isotherm_id).adsorbent_id == keep_id
        assert session.scalar(select(Structure.adsorbent_id)) == keep_id
        assert session.scalar(select(IsothermComponent.source_name)) == "received-1"
        review = session.get(IdentityReview, review_id)
        assert review.status == "merged"
        assert review.merge_snapshot["external_identifier"] == "hash-remove"
        aliases = session.scalars(select(AdsorbentSynonym.synonym)).all()
        assert aliases == ["COF-LZU-1"]


###############################################################################
def test_merge_is_refused_for_shared_experiment_and_rolls_back(
    database: DatabaseManager,
) -> None:
    with database.transaction() as session:
        host = _adsorbent(session, "Host")
        first = _adsorbate(session, "Species A")
        second = _adsorbate(session, "Species-A")
        _experiment(session, "mix", host.id, [first.id, second.id])
        review_id = _open_pair(session, "adsorbate", first, second)
        first_id, second_id = first.id, second.id

    with pytest.raises(MergeRefusedError, match="separate components"):
        IdentityRepository(database).merge(review_id, first_id)

    with database.session_factory() as session:
        assert session.get(Adsorbate, second_id) is not None
        assert session.get(IdentityReview, review_id).status == "open"


###############################################################################
def test_merge_is_refused_for_different_inchi_keys(database: DatabaseManager) -> None:
    with database.transaction() as session:
        first = _adsorbate(session, "Methyl Orange", CO2_KEY)
        second = _adsorbate(session, "Methyl-Orange", CO_KEY)
        review_id = _open_pair(session, "adsorbate", first, second)
        first_id = first.id

    with pytest.raises(MergeRefusedError, match="InChIKeys"):
        IdentityRepository(database).merge(review_id, first_id)


###############################################################################
def test_audit_proposes_only_guard_compatible_pairs(database: DatabaseManager) -> None:
    with database.transaction() as session:
        _adsorbent(session, "COF-LZU-1", "h1")
        _adsorbent(session, "COF-LZU1", "h2")
        _adsorbent(session, "Chlorite, 1.5nm", "h3")
        _adsorbent(session, "Chlorite, 15nm", "h4")
        before = session.scalar(select(func.count(Adsorbent.id)))

    repository = IdentityRepository(database)
    assert repository.audit() == {"adsorbate": 0, "adsorbent": 1}
    assert repository.audit() == {"adsorbate": 0, "adsorbent": 0}
    with database.session_factory() as session:
        assert session.scalar(select(func.count(Adsorbent.id))) == before


###############################################################################
class _FakePubChem(PubChemProvider):
    def __init__(self, options: list[dict[str, Any]], total: int | None = None) -> None:
        self.options = options
        self.total = len(options) if total is None else total
        self.resolved: list[str] = []

    async def candidate_set(self, query: str, limit: int = 5) -> CandidateSet:
        return CandidateSet(self.options, len(self.options), self.total)

    async def close(self) -> None:
        return None


###############################################################################
class _FakeRepository:
    def __init__(self) -> None:
        self.reviews: list[dict[str, Any]] = []

    def open_identity_review(self, **fields: Any) -> None:
        self.reviews.append(fields)


###############################################################################
def _guarded_service(
    options: list[dict[str, Any]], total: int | None = None
) -> tuple[PublicDataService, _FakeRepository, list[str]]:
    repository = _FakeRepository()
    provider = _FakePubChem(options, total)
    provider.key = "pubchem"
    service = PublicDataService(repository=repository, providers=[provider])  # type: ignore[arg-type]
    accepted: list[str] = []

    async def accept(query: str, adsorbate_id: int | None = None) -> tuple[int, bool]:
        accepted.append(query)
        return int(adsorbate_id or 0), True

    service.resolve_pubchem = accept  # type: ignore[method-assign]
    return service, repository, accepted


###############################################################################
def _stored(name: str, formula: str | None = None) -> dict[str, Any]:
    return {"id": 7, "name": name, "inchi_key": None, "formula": formula, "smiles": None}


###############################################################################
def test_single_external_candidate_is_accepted() -> None:
    service, repository, accepted = _guarded_service([{"cid": "1", "formula": "CH4"}])
    asyncio.run(service.resolve_stored_species(_stored("Methane")))
    assert accepted == ["1"]
    assert repository.reviews == []


###############################################################################
def test_formula_evidence_selects_among_external_candidates() -> None:
    service, _, accepted = _guarded_service(
        [{"cid": "1", "formula": "C2H6O"}, {"cid": "2", "formula": "C3H8"}]
    )
    asyncio.run(service.resolve_stored_species(_stored("Solvent X", "C3H8")))
    assert accepted == ["2"]


###############################################################################
@pytest.mark.parametrize(
    ("name", "options", "reason"),
    [
        ("Compound X", [{"cid": "1"}, {"cid": "2"}], "ambiguous_external_match"),
        ("C4H10", [{"cid": "1"}], "formula_not_unique"),
        ("Eu(III)", [{"cid": "1"}], "ion_or_oxidation_state"),
    ],
)
def test_unsafe_external_matches_go_to_review(
    name: str, options: list[dict[str, Any]], reason: str
) -> None:
    service, repository, accepted = _guarded_service(options)
    with pytest.raises(ValueError, match="needs review"):
        asyncio.run(service.resolve_stored_species(_stored(name)))
    assert accepted == []
    assert repository.reviews[0]["reason"] == reason


###############################################################################
def test_unique_match_in_a_truncated_candidate_set_goes_to_review() -> None:
    service, repository, accepted = _guarded_service(
        [{"cid": "1", "formula": "C3H8"}, {"cid": "2", "formula": "C2H6O"}], total=40
    )
    with pytest.raises(ValueError, match="needs review"):
        asyncio.run(service.resolve_stored_species(_stored("Solvent X", "C3H8")))
    assert accepted == []
    assert repository.reviews[0]["reason"] == "incomplete_candidate_set"
    assert repository.reviews[0]["evidence"]["matched"] == 40


###############################################################################
def test_clean_name_repairs_dashes_and_spacing() -> None:
    assert clean_name("  BFTB–BCTA  COF ") == "BFTB-BCTA COF"


###############################################################################
@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("2D-PN-2-Me in boiling water", "2D-PN-2-OMe in boiling water"),
        ("5% -COOH Functionalized CMK-5", "5% -OH Functionalized CMK-5"),
        ("Carbon Black CC N990", "Carbon Black EC N990"),
        ("2,3 DhaTph", "2,3 DmaTph"),
    ],
)
def test_fuzzy_score_ignores_meaningful_sample_differences(
    left: str, right: str
) -> None:
    assert fuzzy_score(left, right) == 0.0


###############################################################################
def test_fuzzy_score_accepts_a_misspelled_long_word() -> None:
    assert fuzzy_score("Ammonium sulphate", "Ammonium sulphte") >= 0.9


###############################################################################
def test_inchi_key_rejects_nist_placeholder() -> None:
    assert is_inchi_key("CURLTUGMZLYLDI-UHFFFAOYSA-N")
    assert not is_inchi_key("NIST-ISODB-0001")


###############################################################################
@pytest.mark.parametrize(
    ("value", "expected"),
    [("C4H10", True), ("CO2", True), ("Methane", False), ("Titanium", False)],
)
def test_formula_detection(value: str, expected: bool) -> None:
    assert looks_like_formula(value) is expected


###############################################################################
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Activated carbon", True),
        ("Activated carbon (synthetic)", True),
        ("Coal", True),
        ("Norit R1 activated carbon", False),
        ("Coal YZG with 2.1% humidity", False),
    ],
)
def test_generic_adsorbent_detection(value: str, expected: bool) -> None:
    assert is_generic_adsorbent(value) is expected
