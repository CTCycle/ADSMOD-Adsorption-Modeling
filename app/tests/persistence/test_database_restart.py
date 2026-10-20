# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select

from server.app import create_app
from server.configurations.settings import AdsmodConfig, DatabaseConfig, UserSettings
from server.repositories.schemas.models import Dataset

###############################################################################
def _temporary_config(storage_root: Path):
    return AdsmodConfig.assemble(
        database=DatabaseConfig(embedded_database=True, sqlite_path="data/test.db"),
        storage_root=storage_root,
        settings=UserSettings(),
    )

###############################################################################
def test_app_restart_preserves_persisted_dataset(tmp_path: Path) -> None:
    config = _temporary_config(tmp_path)

    with TestClient(create_app(config)) as first_client:
        assert first_client.get("/health/ready").json()["state"] == "ready"
        database = first_client.app.state.core_container.database
        with database.transaction() as session:
            session.add(Dataset(name="Restart persistence", source="uploaded"))

    with TestClient(create_app(config)) as restarted_client:
        assert restarted_client.get("/health/ready").json()["state"] == "ready"
        database = restarted_client.app.state.core_container.database
        with database.session() as session:
            dataset = session.execute(
                select(Dataset).where(
                    Dataset.normalized_name == "restart persistence"
                )
            ).scalar_one()

        assert dataset.name == "Restart persistence"
