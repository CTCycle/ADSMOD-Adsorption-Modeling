from __future__ import annotations

import pandas as pd

from ADSMOD.server.repositories.database.backend import database
from ADSMOD.server.repositories.queries.nist import NISTDataSerializer


###############################################################################
def test_save_materials_preserves_existing_adsorbate_key(monkeypatch) -> None:
    serializer = NISTDataSerializer()
    captured: dict[str, object] = {}

    def fake_upsert(df: pd.DataFrame, table_name: str) -> None:
        captured["table_name"] = table_name
        captured["frame"] = df.copy()

    monkeypatch.setattr(database, "upsert_into_database", fake_upsert)
    monkeypatch.setattr(serializer, "_load_adsorbate_keys_by_inchi", lambda values: {})

    guest_data = pd.DataFrame(
        [
            {
                "adsorbate_key": "inchi:ABCDEF1234567890ABCDEF12",
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
    assert frame.iloc[0]["adsorbate_key"] == "inchi:ABCDEF1234567890ABCDEF12"
    assert frame.iloc[0]["InChIKey"] == "ABCDEF1234567890ABCDEF12"


###############################################################################
def test_save_materials_generates_adsorbate_key_when_missing(monkeypatch) -> None:
    serializer = NISTDataSerializer()
    captured: dict[str, object] = {}

    def fake_upsert(df: pd.DataFrame, table_name: str) -> None:
        captured["table_name"] = table_name
        captured["frame"] = df.copy()

    monkeypatch.setattr(database, "upsert_into_database", fake_upsert)
    monkeypatch.setattr(serializer, "_load_adsorbate_keys_by_inchi", lambda values: {})

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
def test_save_materials_preserves_existing_adsorbent_key(monkeypatch) -> None:
    serializer = NISTDataSerializer()
    captured: dict[str, object] = {}

    def fake_upsert(df: pd.DataFrame, table_name: str) -> None:
        captured["table_name"] = table_name
        captured["frame"] = df.copy()

    monkeypatch.setattr(database, "upsert_into_database", fake_upsert)
    monkeypatch.setattr(serializer, "_load_adsorbent_keys_by_hash", lambda values: {})

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
    assert frame.iloc[0]["adsorbent_key"] == "host:HostHash001"


###############################################################################
def test_save_materials_reuses_existing_db_adsorbate_key_for_inchi(
    monkeypatch,
) -> None:
    serializer = NISTDataSerializer()
    captured: dict[str, object] = {}

    def fake_upsert(df: pd.DataFrame, table_name: str) -> None:
        captured["table_name"] = table_name
        captured["frame"] = df.copy()

    monkeypatch.setattr(database, "upsert_into_database", fake_upsert)
    monkeypatch.setattr(
        serializer,
        "_load_adsorbate_keys_by_inchi",
        lambda values: {"abcdef1234567890abcdef12": "legacy:ads-key-001"},
    )

    guest_data = pd.DataFrame(
        [
            {
                "adsorbate_key": pd.NA,
                "InChIKey": "ABCDEF1234567890ABCDEF12",
                "name": "Ethane",
            }
        ]
    )
    serializer.save_materials_datasets(guest_data=guest_data, host_data=None)

    frame = captured["frame"]
    assert isinstance(frame, pd.DataFrame)
    assert frame.iloc[0]["adsorbate_key"] == "legacy:ads-key-001"


###############################################################################
def test_save_materials_overrides_payload_key_when_inchi_exists_in_db(
    monkeypatch,
) -> None:
    serializer = NISTDataSerializer()
    captured: dict[str, object] = {}

    def fake_upsert(df: pd.DataFrame, table_name: str) -> None:
        captured["table_name"] = table_name
        captured["frame"] = df.copy()

    monkeypatch.setattr(database, "upsert_into_database", fake_upsert)
    monkeypatch.setattr(
        serializer,
        "_load_adsorbate_keys_by_inchi",
        lambda values: {"abcdef1234567890abcdef12": "legacy:ads-key-001"},
    )

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
    assert frame.iloc[0]["adsorbate_key"] == "legacy:ads-key-001"


###############################################################################
def test_save_materials_deduplicates_duplicate_inchi_rows_before_upsert(
    monkeypatch,
) -> None:
    serializer = NISTDataSerializer()
    captured: dict[str, object] = {}

    def fake_upsert(df: pd.DataFrame, table_name: str) -> None:
        captured["table_name"] = table_name
        captured["frame"] = df.copy()

    monkeypatch.setattr(database, "upsert_into_database", fake_upsert)
    monkeypatch.setattr(serializer, "_load_adsorbate_keys_by_inchi", lambda values: {})

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

    def fake_upsert(df: pd.DataFrame, table_name: str) -> None:
        captured["table_name"] = table_name
        captured["frame"] = df.copy()

    monkeypatch.setattr(database, "upsert_into_database", fake_upsert)
    monkeypatch.setattr(serializer, "_load_adsorbate_keys_by_inchi", lambda values: {})

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

    def fake_upsert(df: pd.DataFrame, table_name: str) -> None:
        captured["table_name"] = table_name
        captured["frame"] = df.copy()

    monkeypatch.setattr(database, "upsert_into_database", fake_upsert)
    monkeypatch.setattr(serializer, "_load_adsorbent_keys_by_hash", lambda values: {})

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
