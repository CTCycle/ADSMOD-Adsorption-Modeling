from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
HELPER = PROJECT_ROOT / "app" / "scripts" / "ensure_environment.ps1"

###############################################################################
def run_environment_helper(env_file: Path, env_example: Path) -> subprocess.CompletedProcess[str]:
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        pytest.skip("PowerShell is not available")

    command = (
        ". $env:ADSMOD_ENV_HELPER; "
        "$created = Ensure-EnvironmentFile "
        "-EnvFile $env:ADSMOD_ENV_FILE "
        "-EnvExample $env:ADSMOD_ENV_EXAMPLE; "
        "if ($created) { exit 0 } else { exit 7 }"
    )
    process_environment = os.environ.copy()
    process_environment.update(
        {
            "ADSMOD_ENV_HELPER": str(HELPER),
            "ADSMOD_ENV_FILE": str(env_file),
            "ADSMOD_ENV_EXAMPLE": str(env_example),
        }
    )
    return subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", command],
        env=process_environment,
        capture_output=True,
        text=True,
        check=False,
    )

###############################################################################
def test_missing_environment_file_is_copied_without_value_changes(tmp_path: Path) -> None:
    env_example = tmp_path / ".env.example"
    env_file = tmp_path / ".env"
    expected = b"BACKEND_LOGS_VISIBLE=false\nSECRET_PLACEHOLDER=example\n"
    env_example.write_bytes(expected)

    result = run_environment_helper(env_file, env_example)

    assert result.returncode == 0, result.stderr
    assert env_file.read_bytes() == expected

###############################################################################
def test_existing_environment_file_is_preserved(tmp_path: Path) -> None:
    env_example = tmp_path / ".env.example"
    env_file = tmp_path / ".env"
    original = b"BACKEND_LOGS_VISIBLE=false\nSECRET_PLACEHOLDER=local\n"
    env_example.write_bytes(b"SECRET_PLACEHOLDER=example\n")
    env_file.write_bytes(original)

    result = run_environment_helper(env_file, env_example)

    assert result.returncode == 7, result.stderr
    assert env_file.read_bytes() == original
