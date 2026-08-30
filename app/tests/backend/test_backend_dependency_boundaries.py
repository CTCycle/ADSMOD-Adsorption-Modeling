from __future__ import annotations

import re
from pathlib import Path


BACKEND_ROOT = Path("app/backend")
GENERATED_DIRS = {".venv", "__pycache__", ".pytest_cache", ".uv-cache"}


def _iter_python_files(root: Path):
    for path in root.rglob("*.py"):
        if any(part in GENERATED_DIRS for part in path.parts):
            continue
        yield path


def _has_import(text: str, package: str) -> bool:
    return bool(
        re.search(
            rf"^(?:from|import) {re.escape(package)}(?:\.|\s|$)",
            text,
            re.MULTILINE,
        )
    )


def test_common_has_no_framework_or_persistence_imports() -> None:
    forbidden = ("fastapi", "sqlalchemy")
    for path in _iter_python_files(BACKEND_ROOT / "common"):
        text = path.read_text(encoding="utf-8")
        hits = [package for package in forbidden if _has_import(text, package)]
        assert not hits, f"{path}: forbidden imports {hits}"


def test_core_does_not_import_ml_or_ml_heavy_packages() -> None:
    forbidden = ("adsmod_ml", "torch", "keras", "sklearn")
    for path in _iter_python_files(BACKEND_ROOT / "core"):
        text = path.read_text(encoding="utf-8")
        hits = [package for package in forbidden if _has_import(text, package)]
        assert not hits, f"{path}: forbidden imports {hits}"


def test_ml_does_not_import_core_or_persistence() -> None:
    forbidden = ("adsmod_core", "sqlalchemy", "alembic")
    for path in _iter_python_files(BACKEND_ROOT / "ml"):
        text = path.read_text(encoding="utf-8")
        hits = [package for package in forbidden if _has_import(text, package)]
        assert not hits, f"{path}: forbidden imports {hits}"


def test_contract_packages_have_no_framework_or_persistence_imports() -> None:
    forbidden = ("fastapi", "sqlalchemy")
    for package_root in (
        BACKEND_ROOT / "common" / "src",
        BACKEND_ROOT / "core" / "src" / "adsmod_core" / "contracts",
        BACKEND_ROOT / "ml" / "src" / "adsmod_ml" / "contracts",
    ):
        for path in _iter_python_files(package_root):
            text = path.read_text(encoding="utf-8")
            hits = [package for package in forbidden if _has_import(text, package)]
            assert not hits, f"{path}: forbidden imports {hits}"
