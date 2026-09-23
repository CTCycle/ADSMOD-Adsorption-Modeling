from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select

from server.app import create_app
from server.configurations.settings import StorageConfig, load_config
from server.repositories.schemas.models import Dataset


CONFIG_PATH = Path("app/resources/adsmod.json")


###############################################################################
def _temporary_config(storage_root: Path):
    base = load_config(CONFIG_PATH)
    return base.model_copy(
        update={
            "storage": StorageConfig(root=storage_root),
            "application": base.application.model_copy(
                update={
                    "database": base.application.database.model_copy(
                        update={"sqlite_path": "data/test.db"}
                    )
                }
            ),
        }
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
