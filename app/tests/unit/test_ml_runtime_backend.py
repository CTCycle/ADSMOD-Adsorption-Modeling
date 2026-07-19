from __future__ import annotations

import importlib
import os
import sys

###############################################################################
def test_ml_app_configures_torch_backend_before_keras_import(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("KERAS_BACKEND", raising=False)
    sys.modules.pop("ml_service.app", None)
    importlib.reload(importlib.import_module("ml_service"))
    importlib.import_module("ml_service.app")
    assert os.environ["KERAS_BACKEND"] == "torch"
