from __future__ import annotations

from pathlib import Path

GENERATED_DIRS = {".venv", "__pycache__", ".pytest_cache", ".startup-temp", ".uv-cache"}

###############################################################################
def _iter_python_files(root: str):
    for path in Path(root).rglob("*.py"):
        if any(part in GENERATED_DIRS for part in path.parts):
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

###############################################################################
def test_contract_packages_have_no_framework_or_persistence_imports() -> None:
    forbidden = [
        "from fastapi",
        "import fastapi",
        "from sqlalchemy",
        "import sqlalchemy",
    ]
    roots = [
        "app/server/core_service/core_service/contracts",
        "app/server/ml_service/ml_service/contracts",
        "app/server/shared/shared/contracts",
    ]
    for root in roots:
        for path in _iter_python_files(root):
            text = path.read_text(encoding="utf-8")
            hits = [item for item in forbidden if item in text]
            assert not hits, f"{path}: contract imports {hits}"

###############################################################################
def test_retired_contract_paths_are_absent_and_unreferenced() -> None:
    retired_paths = [
        Path("app/server/core_service/core_service/domain"),
        Path("app/server/ml_service/ml_service/domain"),
        Path("app/server/shared/shared/models"),
        Path("app/server/core_service/core_service/services/data/units.py"),
    ]
    for path in retired_paths:
        assert not path.exists(), f"retired path still exists: {path}"

    forbidden_imports = [
        "core_service.domain",
        "ml_service.domain",
        "shared.models.jobs",
        "core_service.services.data.units",
    ]
    for path in _iter_python_files("app"):
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        hits = [item for item in forbidden_imports if item in text]
        assert not hits, f"{path}: retired imports {hits}"

###############################################################################
def test_extracted_v3_dependency_direction_is_explicit() -> None:
    rules = {
        "app/backend/common/src": [
            "adsmod_core",
            "adsmod_ml",
            "core_service",
            "ml_service",
            "shared",
        ],
        "app/backend/core/src": [
            "adsmod_ml",
            "core_service",
            "ml_service",
            "shared",
        ],
        "app/backend/ml/src": [
            "adsmod_core",
            "core_service",
            "ml_service",
            "shared",
        ],
    }
    for root, forbidden in rules.items():
        for path in _iter_python_files(root):
            text = path.read_text(encoding="utf-8")
            hits = [
                item
                for item in forbidden
                if f"from {item}" in text or f"import {item}" in text
            ]
            assert not hits, f"{path}: forbidden v3 dependencies {hits}"
