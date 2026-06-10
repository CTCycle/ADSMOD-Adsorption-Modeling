from __future__ import annotations

from pathlib import Path


###############################################################################
def _iter_python_files(root: str):
    for path in Path(root).rglob("*.py"):
        if any(part in {".venv", "__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        yield path


###############################################################################
def test_core_has_no_ml_imports() -> None:
    forbidden = [
        "from ml_service",
        "import ml_service",
        "import torch",
        "from torch",
        "import keras",
        "from keras",
        "import sklearn",
        "from sklearn",
    ]
    for path in _iter_python_files("app/server/core_service"):
        text = path.read_text(encoding="utf-8")
        hits = [item for item in forbidden if item in text]
        assert not hits, f"{path}: forbidden imports {hits}"


###############################################################################
def test_shared_has_no_service_imports() -> None:
    forbidden = [
        "from core_service",
        "import core_service",
        "from ml_service",
        "import ml_service",
    ]
    for path in _iter_python_files("app/server/shared"):
        text = path.read_text(encoding="utf-8")
        hits = [item for item in forbidden if item in text]
        assert not hits, f"{path}: forbidden imports {hits}"


###############################################################################
def test_no_legacy_monolith_imports_remain() -> None:
    forbidden = [
        "app.server.api",
        "app.server.common",
        "app.server.configurations",
        "app.server.domain",
        "app.server.learning",
        "app.server.repositories",
        "app.server.services",
    ]
    for path in _iter_python_files("app/server"):
        text = path.read_text(encoding="utf-8")
        hits = [item for item in forbidden if item in text]
        assert not hits, f"{path}: forbidden legacy imports {hits}"
