from __future__ import annotations

import re
from pathlib import Path

BACKEND_ROOT = Path("app/server")
GENERATED_DIRS = {".venv", "__pycache__", ".pytest_cache", ".uv-cache"}

###############################################################################
def _iter_python_files(root: Path):
    for path in root.rglob("*.py"):
        if not any(part in GENERATED_DIRS for part in path.parts):
            yield path

###############################################################################
def _has_import(text: str, package: str) -> bool:
    return bool(re.search(rf"^(?:from|import) {re.escape(package)}(?:\.|\s|$)", text, re.MULTILINE))

###############################################################################
def test_common_and_domain_have_no_framework_or_persistence_imports() -> None:
    for package_root in (BACKEND_ROOT / "common", BACKEND_ROOT / "domain"):
        for path in _iter_python_files(package_root):
            text = path.read_text(encoding="utf-8")
            hits = [
                pkg
                for pkg in ("fastapi", "sqlalchemy", "alembic")
                if _has_import(text, pkg)
            ]
            assert not hits, f"{path}: forbidden imports {hits}"

###############################################################################
def test_core_runtime_does_not_statically_import_heavy_ml_packages() -> None:
    core_paths = [
        BACKEND_ROOT / "app.py",
        BACKEND_ROOT / "configurations",
        BACKEND_ROOT / "repositories",
        BACKEND_ROOT / "api" / "datasets.py",
        BACKEND_ROOT / "api" / "entrypoint.py",
        BACKEND_ROOT / "api" / "fitting.py",
        BACKEND_ROOT / "api" / "nist.py",
        BACKEND_ROOT / "api" / "public_data.py",
        BACKEND_ROOT / "api" / "routes.py",
        BACKEND_ROOT / "services" / "container.py",
        BACKEND_ROOT / "services" / "data" / "datasets.py",
        BACKEND_ROOT / "services" / "data" / "import_parser.py",
        BACKEND_ROOT / "services" / "data" / "importer.py",
        BACKEND_ROOT / "services" / "data" / "nist_mapper.py",
        BACKEND_ROOT / "services" / "data" / "nist_service.py",
        BACKEND_ROOT / "services" / "data" / "nistads.py",
        BACKEND_ROOT / "services" / "data" / "public_data.py",
        BACKEND_ROOT / "services" / "fitting.py",
        BACKEND_ROOT / "services" / "job_responses.py",
        BACKEND_ROOT / "services" / "jobs.py",
        BACKEND_ROOT / "services" / "modeling",
        BACKEND_ROOT / "services" / "providers",
        BACKEND_ROOT / "services" / "training_data.py",
    ]
    for root in core_paths:
        for path in _iter_python_files(root) if root.is_dir() else (root,):
            text = path.read_text(encoding="utf-8")
            hits = [pkg for pkg in ("torch", "keras", "sklearn") if _has_import(text, pkg)]
            assert not hits, f"{path}: forbidden imports {hits}"

###############################################################################
def test_ml_extension_does_not_own_persistence_layer() -> None:
    ml_paths = [
        BACKEND_ROOT / "models",
        BACKEND_ROOT / "services" / "ml_container.py",
        BACKEND_ROOT / "services" / "training.py",
        BACKEND_ROOT / "services" / "data" / "builder.py",
        BACKEND_ROOT / "services" / "data" / "composition.py",
        BACKEND_ROOT / "services" / "data" / "conversion.py",
        BACKEND_ROOT / "services" / "data" / "sanitizer.py",
        BACKEND_ROOT / "services" / "data" / "sequences.py",
        BACKEND_ROOT / "api" / "training.py",
        BACKEND_ROOT / "api" / "training_configuration.py",
    ]
    for root in ml_paths:
        for path in _iter_python_files(root) if root.is_dir() else (root,):
            text = path.read_text(encoding="utf-8")
            hits = [pkg for pkg in ("sqlalchemy", "alembic") if _has_import(text, pkg)]
            assert not hits, f"{path}: forbidden imports {hits}"

###############################################################################
def test_legacy_package_trees_are_not_present() -> None:
    for layer_name in ("core", "ml"):
        layer = BACKEND_ROOT / layer_name
        assert not any(_iter_python_files(layer)), f"legacy Python files remain under {layer}"

    common_layer = BACKEND_ROOT / "common"
    for path in common_layer.iterdir():
        if path.name == "src":
            assert not any(_iter_python_files(path)), (
                f"legacy Python files remain under {path}"
            )
