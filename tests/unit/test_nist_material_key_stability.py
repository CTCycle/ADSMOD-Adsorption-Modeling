from __future__ import annotations

import pandas as pd

from ADSMOD.server.repositories.database.backend import database
from ADSMOD.server.repositories.queries.nist import NISTDataSerializer


class UpsertCapture:
    def __init__(self, captured: dict[str, object]) -> None:
        self.captured = captured

    def __call__(self, df: pd.DataFrame, table_name: str) -> None:
        self.captured["table_name"] = table_name
        self.captured["frame"] = df.copy()


###############################################################################
def test_save_materials_derives_adsorbate_key_from_inchi(monkeypatch) -> None:
    serializer = NISTDataSerializer()
    captured: dict[str, object] = {}
    upsert_capture = UpsertCapture(captured)
    monkeypatch.setattr(database, "upsert_into_database", upsert_capture)

    guest_data = pd.DataFrame(
        [
            {
                "adsorbate_key": "name:stale-key-123",
                "InChIKey": "ABCDEF1234567890ABCDEF12",
                "name": "Methane",
                "molecular_weight": 16.04,
            }
        ]
    )
    serializer.save_materials_datasets(guest_data=guest_data, host_data=None)

    assert captured["table_name"] == "adsorbates"
    frame = captured["frame"]
    assert isinstance(frame, pd.DataFrame)
    assert frame.iloc[0]["adsorbate_key"] == "inchi:abcdef1234567890abcdef12"
    assert frame.iloc[0]["InChIKey"] == "ABCDEF1234567890ABCDEF12"


###############################################################################
def test_save_materials_generates_adsorbate_key_when_missing(monkeypatch) -> None:
    serializer = NISTDataSerializer()
    captured: dict[str, object] = {}
    upsert_capture = UpsertCapture(captured)
    monkeypatch.setattr(database, "upsert_into_database", upsert_capture)

    guest_data = pd.DataFrame(
        [
            {
                "adsorbate_key": pd.NA,
                "InChIKey": "ZXCVBNM1234567890ZXCVBNM12",
                "name": "Ethane",
            }
        ]
    )
    serializer.save_materials_datasets(guest_data=guest_data, host_data=None)

    frame = captured["frame"]
    assert isinstance(frame, pd.DataFrame)
    assert frame.iloc[0]["adsorbate_key"] == "inchi:zxcvbnm1234567890zxcvbnm12"


###############################################################################
def test_save_materials_derives_adsorbent_key_from_hash(monkeypatch) -> None:
    serializer = NISTDataSerializer()
    captured: dict[str, object] = {}
    upsert_capture = UpsertCapture(captured)
    monkeypatch.setattr(database, "upsert_into_database", upsert_capture)

    host_data = pd.DataFrame(
        [
            {
                "adsorbent_key": "host:HostHash001",
                "hashkey": "HostHash001",
                "name": "MOF-5",
            }
        ]
    )
    serializer.save_materials_datasets(guest_data=None, host_data=host_data)

    assert captured["table_name"] == "adsorbents"
    frame = captured["frame"]
    assert isinstance(frame, pd.DataFrame)
    assert frame.iloc[0]["adsorbent_key"] == "host:hosthash001"


###############################################################################
def test_save_materials_overrides_payload_key_with_deterministic_inchi_key(
    monkeypatch,
) -> None:
    serializer = NISTDataSerializer()
    captured: dict[str, object] = {}
    upsert_capture = UpsertCapture(captured)
    monkeypatch.setattr(database, "upsert_into_database", upsert_capture)

    guest_data = pd.DataFrame(
        [
            {
                "adsorbate_key": "name:stale-key-123",
                "InChIKey": "ABCDEF1234567890ABCDEF12",
                "name": "Ethane",
            }
        ]
    )
    serializer.save_materials_datasets(guest_data=guest_data, host_data=None)

    assert captured["table_name"] == "adsorbates"
    frame = captured["frame"]
    assert isinstance(frame, pd.DataFrame)
    assert frame.iloc[0]["adsorbate_key"] == "inchi:abcdef1234567890abcdef12"


###############################################################################
def test_save_materials_deduplicates_duplicate_inchi_rows_before_upsert(
    monkeypatch,
) -> None:
    serializer = NISTDataSerializer()
    captured: dict[str, object] = {}
    upsert_capture = UpsertCapture(captured)
    monkeypatch.setattr(database, "upsert_into_database", upsert_capture)

    guest_data = pd.DataFrame(
        [
            {
                "adsorbate_key": "name:key-a",
                "InChIKey": "ABCDEF1234567890ABCDEF12",
                "name": "ethane",
            },
            {
                "adsorbate_key": "name:key-b",
                "InChIKey": " ABCDEF1234567890ABCDEF12 ",
                "name": "ethane",
            },
        ]
    )
    serializer.save_materials_datasets(guest_data=guest_data, host_data=None)

    assert captured["table_name"] == "adsorbates"
    frame = captured["frame"]
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 1
    assert frame.iloc[0]["InChIKey"] == "ABCDEF1234567890ABCDEF12"


###############################################################################
def test_save_materials_generates_adsorbate_key_when_column_is_missing(
    monkeypatch,
) -> None:
    serializer = NISTDataSerializer()
    captured: dict[str, object] = {}
    upsert_capture = UpsertCapture(captured)
    monkeypatch.setattr(database, "upsert_into_database", upsert_capture)

    guest_data = pd.DataFrame(
        [
            {
                "InChIKey": "QWERTY1234567890QWERTY123",
                "name": "Propane",
            }
        ]
    )
    serializer.save_materials_datasets(guest_data=guest_data, host_data=None)

    assert captured["table_name"] == "adsorbates"
    frame = captured["frame"]
    assert isinstance(frame, pd.DataFrame)
    assert frame.iloc[0]["adsorbate_key"] == "inchi:qwerty1234567890qwerty123"


###############################################################################
def test_save_materials_generates_adsorbent_key_when_column_is_missing(
    monkeypatch,
) -> None:
    serializer = NISTDataSerializer()
    captured: dict[str, object] = {}
    upsert_capture = UpsertCapture(captured)
    monkeypatch.setattr(database, "upsert_into_database", upsert_capture)

    host_data = pd.DataFrame(
        [
            {
                "hashkey": "HostHashXYZ",
                "name": "Zeolite",
            }
        ]
    )
    serializer.save_materials_datasets(guest_data=None, host_data=host_data)

    assert captured["table_name"] == "adsorbents"
    frame = captured["frame"]
    assert isinstance(frame, pd.DataFrame)
    assert frame.iloc[0]["adsorbent_key"] == "host:hosthashxyz"
