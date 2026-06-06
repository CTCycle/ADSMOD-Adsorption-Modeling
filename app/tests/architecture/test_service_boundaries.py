from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVER_ROOT = REPO_ROOT / "app" / "server"
CORE_PACKAGE_ROOT = SERVER_ROOT / "core_service" / "core_service"
ML_PACKAGE_ROOT = SERVER_ROOT / "ml_service" / "ml_service"
SHARED_PACKAGE_ROOT = SERVER_ROOT / "shared" / "shared"
CORE_FRONTEND_ROOT = REPO_ROOT / "app" / "client" / "src" / "app"
ML_FRONTEND_ROOT = REPO_ROOT / "app" / "ml_client" / "src" / "app"
UNIFIED_BACKEND_ENTRYPOINT = SERVER_ROOT / "app.py"

ML_HEAVY_IMPORT_ROOTS = {"keras", "sklearn", "tensorflow", "torch"}
SERVICE_IMPORT_ROOTS = {"core_service", "ml_service"}
LEGACY_BACKEND_IMPORT_ROOTS = {
    "api",
    "common",
    "configurations",
    "domain",
    "learning",
    "repositories",
    "services",
}
CORE_FORBIDDEN_FRONTEND_PATH_PARTS = {"training"}
CORE_FORBIDDEN_FRONTEND_SNIPPETS = {
    "/api/training",
    "${API_BASE_URL}/training",
    "`/training",
    "'/training",
    '"/training',
}
ML_FORBIDDEN_FRONTEND_SNIPPETS = {
    "/api/datasets",
    "/api/fitting",
    "/api/nist",
    "${API_BASE_URL}/datasets",
    "${API_BASE_URL}/fitting",
    "${API_BASE_URL}/nist",
    "`/datasets",
    "`/fitting",
    "`/nist",
    "'/datasets",
    "'/fitting",
    "'/nist",
    '"/datasets',
    '"/fitting',
    '"/nist',
}
TEXT_EXTENSIONS = {".cjs", ".css", ".html", ".js", ".json", ".mjs", ".ts"}


def _iter_files(root: Path, suffixes: set[str]) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in suffixes:
            yield path


def _iter_python_files(root: Path) -> Iterable[Path]:
    return _iter_files(root, {".py"})


def _parse_import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    return imports


def _find_python_import_violations(root: Path, forbidden_roots: set[str]) -> list[str]:
    violations: list[str] = []
    for path in _iter_python_files(root):
        imported_roots = _parse_import_roots(path)
        forbidden = sorted(imported_roots & forbidden_roots)
        if forbidden:
            relative_path = path.relative_to(REPO_ROOT).as_posix()
            violations.append(f"{relative_path}: forbidden imports {', '.join(forbidden)}")
    return violations


def _find_text_snippet_violations(root: Path, forbidden_snippets: set[str]) -> list[str]:
    violations: list[str] = []
    for path in _iter_files(root, TEXT_EXTENSIONS):
        text = path.read_text(encoding="utf-8")
        matches = sorted(snippet for snippet in forbidden_snippets if snippet in text)
        if matches:
            relative_path = path.relative_to(REPO_ROOT).as_posix()
            violations.append(f"{relative_path}: forbidden snippets {', '.join(matches)}")
    return violations


def test_core_service_does_not_depend_on_ml_service_or_ml_frameworks() -> None:
    violations = _find_python_import_violations(
        CORE_PACKAGE_ROOT,
        {"ml_service", *ML_HEAVY_IMPORT_ROOTS},
    )
    assert not violations, "\n".join(violations)


def test_ml_service_does_not_depend_on_core_service() -> None:
    violations = _find_python_import_violations(ML_PACKAGE_ROOT, {"core_service"})
    assert not violations, "\n".join(violations)


def test_shared_layer_does_not_depend_on_service_packages() -> None:
    violations = _find_python_import_violations(SHARED_PACKAGE_ROOT, SERVICE_IMPORT_ROOTS)
    assert not violations, "\n".join(violations)


def test_active_backend_packages_do_not_import_legacy_top_level_packages() -> None:
    active_roots = [CORE_PACKAGE_ROOT, ML_PACKAGE_ROOT, SHARED_PACKAGE_ROOT]
    violations: list[str] = []
    for root in active_roots:
        violations.extend(_find_python_import_violations(root, LEGACY_BACKEND_IMPORT_ROOTS))
    assert not violations, "\n".join(violations)


def test_unified_backend_entrypoint_is_only_composition_glue() -> None:
    imports = _parse_import_roots(UNIFIED_BACKEND_ENTRYPOINT)
    forbidden = sorted(imports & ML_HEAVY_IMPORT_ROOTS)
    assert not forbidden, f"app/server/app.py must not import ML frameworks directly: {forbidden}"


def test_core_frontend_does_not_reference_training_or_ml_api_routes() -> None:
    path_violations = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in _iter_files(CORE_FRONTEND_ROOT, TEXT_EXTENSIONS)
        if set(path.relative_to(CORE_FRONTEND_ROOT).parts) & CORE_FORBIDDEN_FRONTEND_PATH_PARTS
    ]
    snippet_violations = _find_text_snippet_violations(
        CORE_FRONTEND_ROOT,
        CORE_FORBIDDEN_FRONTEND_SNIPPETS,
    )
    violations = [*path_violations, *snippet_violations]
    assert not violations, "\n".join(violations)


def test_ml_frontend_only_uses_training_api_surface() -> None:
    violations = _find_text_snippet_violations(
        ML_FRONTEND_ROOT,
        ML_FORBIDDEN_FRONTEND_SNIPPETS,
    )
    assert not violations, "\n".join(violations)


def test_core_service_dependency_manifest_excludes_ml_frameworks() -> None:
    manifest = (SERVER_ROOT / "core_service" / "pyproject.toml").read_text(encoding="utf-8").lower()
    forbidden_dependencies = sorted(dependency for dependency in ML_HEAVY_IMPORT_ROOTS if dependency in manifest)
    assert not forbidden_dependencies, f"core_service pyproject includes ML dependencies: {forbidden_dependencies}"
