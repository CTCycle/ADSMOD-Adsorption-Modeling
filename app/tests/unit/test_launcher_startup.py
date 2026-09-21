from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPOSITORY_ROOT / "start_on_windows.ps1"


def launcher_text() -> str:
    return LAUNCHER.read_text(encoding="utf-8-sig")


def start_application_body() -> str:
    text = launcher_text()
    start = text.index("function Start-Application")
    end = text.index("function Install-UpdateDependencies", start)
    return text[start:end]


def test_launcher_has_no_unconditional_rebuild_or_import_probe() -> None:
    text = launcher_text()

    assert "ALWAYS_REBUILD" not in text
    assert "import server.app, fastapi, uvicorn" not in text
    assert "function Install-FrontendDependencies" in text
    assert "function Build-Frontend" in text
    assert "function Sync-BackendDependencies" in text
    assert "function Sync-FrontendDependencies" not in text


def test_frontend_build_fingerprint_covers_only_declared_inputs() -> None:
    text = launcher_text()
    fingerprint_start = text.index("function Get-FrontendBuildFingerprint")
    fingerprint_end = text.index("function Read-StateFile", fingerprint_start)
    fingerprint = text[fingerprint_start:fingerprint_end]

    for input_name in (
        "package.json",
        "package-lock.json",
        "angular.json",
        "tsconfig.json",
        "tsconfig.app.json",
        "index.html",
        "Join-Path $ClientDir 'src'",
        "Join-Path $ClientDir 'public'",
        "scripts\\verify-angular-migration.mjs",
        "launcher-node-version",
    ):
        assert input_name in fingerprint

    assert "proxy.conf.cjs" not in fingerprint
    assert "playwright.config.ts" not in fingerprint
    assert "eslint.config.js" not in fingerprint


def test_startup_preflights_all_ports_before_dependency_or_build_work() -> None:
    body = start_application_body()

    assert body.index("Get-PortConflicts") < body.index("Test-PortableRuntimeFilesReady")
    assert body.index("Resolve-PortConflicts") < body.index("Test-BackendDependenciesCurrent")
    assert "@($settings.BackendPort, $settings.FrontendPort)" in body
    assert "Wait-ForHealth -Url $healthUrl -TimeoutSeconds 60 -Process" in body
    assert "Wait-ForHealth -Url $frontendUrl -TimeoutSeconds 60 -Process" in body


def test_port_conflict_resolution_is_fail_closed_and_deduplicates_pids() -> None:
    text = launcher_text()
    conflict_start = text.index("function Get-PortConflicts")
    conflict_end = text.index("function Get-ApplicationProcessRecords", conflict_start)
    conflict_code = text[conflict_start:conflict_end]

    assert "GetActiveTcpListeners" in text
    assert "function Resolve-PortConflicts" in conflict_code
    assert "HashSet[int]" in conflict_code
    assert "OwnershipResolved" in conflict_code
    assert "non-interactive" in conflict_code
    assert "ownership of port" in conflict_code
    assert "remaining" in conflict_code
