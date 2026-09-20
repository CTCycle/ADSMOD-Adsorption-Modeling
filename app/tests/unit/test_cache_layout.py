from __future__ import annotations

import configparser
import json
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_CACHE_ROOT = (REPO_ROOT / "runtimes" / "cache").resolve()


###############################################################################
def _resolve_configured_path(value: str, base: Path) -> Path:
    return (base / value).resolve()


###############################################################################
def test_tool_configuration_resolves_to_the_canonical_cache_root() -> None:
    pytest_config = configparser.ConfigParser()
    pytest_config.read(REPO_ROOT / "app" / "tests" / "pytest.ini", encoding="utf-8")
    assert _resolve_configured_path(
        pytest_config["pytest"]["cache_dir"], REPO_ROOT / "app" / "tests"
    ) == CANONICAL_CACHE_ROOT / "pytest"
    assert "cache" in pytest_config["pytest"]["norecursedirs"].split()

    backend_config = tomllib.loads(
        (REPO_ROOT / "app" / "server" / "pyproject.toml").read_text(encoding="utf-8")
    )
    pytest_options = backend_config["tool"]["pytest"]["ini_options"]
    assert "cache" in pytest_options["norecursedirs"]
    assert _resolve_configured_path(
        pytest_options["cache_dir"], REPO_ROOT / "app" / "server"
    ) == CANONICAL_CACHE_ROOT / "pytest"
    assert _resolve_configured_path(
        backend_config["tool"]["ruff"]["cache-dir"], REPO_ROOT / "app" / "server"
    ) == CANONICAL_CACHE_ROOT / "ruff"

    ruff_config = tomllib.loads((REPO_ROOT / "ruff.toml").read_text(encoding="utf-8"))
    assert _resolve_configured_path(
        ruff_config["cache-dir"], REPO_ROOT
    ) == CANONICAL_CACHE_ROOT / "ruff"

    angular_config = json.loads(
        (REPO_ROOT / "app" / "client" / "angular.json").read_text(encoding="utf-8")
    )
    assert _resolve_configured_path(
        angular_config["cli"]["cache"]["path"], REPO_ROOT / "app" / "client"
    ) == CANONICAL_CACHE_ROOT / "angular"


###############################################################################
def test_supported_tooling_has_no_legacy_cache_destination() -> None:
    paths = (
        REPO_ROOT / "app" / "tests" / "pytest.ini",
        REPO_ROOT / "app" / "tests" / "run_tests.bat",
        REPO_ROOT / "app" / "server" / "pyproject.toml",
        REPO_ROOT / "app" / "client" / "angular.json",
        REPO_ROOT / "ruff.toml",
        REPO_ROOT / ".github" / "workflows" / "ci.yml",
        REPO_ROOT / "assets" / "docs" / "operations" / "commands.md",
        REPO_ROOT / "assets" / "docs" / "runtime" / "deployment.md",
    )
    forbidden_fragments = (
        "app/tests/cache",
        "app\\tests\\cache",
        "../tests/cache",
        "..\\tests\\cache",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert not any(fragment in text for fragment in forbidden_fragments), path


###############################################################################
def test_launcher_and_batch_runner_export_canonical_cache_environment() -> None:
    launcher = (REPO_ROOT / "start_on_windows.ps1").read_text(encoding="utf-8")
    runner = (REPO_ROOT / "app" / "tests" / "run_tests.bat").read_text(encoding="utf-8")

    assert '$RuntimeCacheDir = Join-Path $RuntimesDir "cache"' in launcher
    assert "$TestCacheDir" not in launcher
    assert "$LegacyUvCachePaths" not in launcher
    for variable in (
        "$env:UV_CACHE_DIR",
        "$env:PIP_CACHE_DIR",
        "$env:NPM_CONFIG_CACHE",
        "$env:PYTHONPYCACHEPREFIX",
        "$env:PYTEST_CACHE_DIR",
        "$env:PYTEST_ADDOPTS",
        "$env:RUFF_CACHE_DIR",
        "$env:COVERAGE_FILE",
        "$env:PLAYWRIGHT_BROWSERS_PATH",
    ):
        assert variable in launcher

    assert "set \"CACHE_DIR=%PROJECT_ROOT%\\runtimes\\cache\"" in runner
    assert "set \"PYTHONPYCACHEPREFIX=%CACHE_DIR%\\python\"" in runner
    assert "--basetemp \"%PYTEST_TEMP_DIR%\"" in runner
    assert "set \"STARTED_FRONTEND=0\"" in runner
    assert '--ignore "%TESTS_DIR%\\e2e"' in runner
    assert '"%TESTS_DIR%\\e2e" -k "not performance"' in runner
    assert "playwright install chromium" in launcher
