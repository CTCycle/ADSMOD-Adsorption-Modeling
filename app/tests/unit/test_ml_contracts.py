# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations
import inspect
import pytest
from pathlib import Path
from server.services.ml_container import MlServiceContainer
from types import SimpleNamespace
from server.models.training.manager import TrainingManager, TrainingProcessRunner


# Ml boundary


###############################################################################
def test_ml_container_consumes_in_process_snapshot_access() -> None:
    parameters = inspect.signature(MlServiceContainer).parameters
    assert "snapshot_access" in parameters
    assert "internal_token" not in parameters


###############################################################################
def test_ml_extension_has_no_standalone_fastapi_server_or_core_http_client() -> None:
    root = Path("app/server")
    assert (root / "app.py").exists()
    assert (root / "cli.py").exists()
    assert not (root / "http" / "entrypoint.py").exists()
    assert not (root / "clients" / "core_client.py").exists()


###############################################################################
def test_ml_source_has_no_backend_to_backend_http_boundary() -> None:
    root = Path("app/server")
    source_roots = [root / "app.py", root / "cli.py"] + [
        root / name
        for name in (
            "api",
            "common",
            "configurations",
            "domain",
            "models",
            "repositories",
            "services",
        )
    ]
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for source_root in source_roots
        for path in (
            [source_root] if source_root.is_file() else source_root.rglob("*.py")
        )
    )
    assert "CoreSnapshotClient" not in combined
    assert "core_base_url" not in combined
    assert "X-ADSMOD-Internal-Token" not in combined


# Resume validation


###############################################################################
def test_resume_validation_accepts_keras3_public_loss_state() -> None:
    optimizer = SimpleNamespace(variables=[object()])
    model = SimpleNamespace(optimizer=optimizer, loss=object())
    TrainingProcessRunner.validate_resume_model(object(), model)


###############################################################################
def test_resume_validation_rejects_missing_loss() -> None:
    optimizer = SimpleNamespace(variables=[object()])
    model = SimpleNamespace(optimizer=optimizer, loss=None)
    with pytest.raises(ValueError, match="not compiled"):
        TrainingProcessRunner.validate_resume_model(object(), model)


###############################################################################
def test_reconstructed_history_uses_one_based_epochs() -> None:
    manager = object.__new__(TrainingManager)

    history = manager.build_history_entries(
        {
            "history": {
                "loss": [0.5],
                "val_loss": [0.6],
                "MaskedR2": [0.25],
                "val_MaskedR2": [0.3],
            }
        }
    )

    assert history == [
        {
            "epoch": 1,
            "loss": 0.5,
            "val_loss": 0.6,
            "masked_r2": 0.25,
            "val_masked_r2": 0.3,
        }
    ]
