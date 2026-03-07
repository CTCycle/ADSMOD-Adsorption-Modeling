from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

SIDE_CAR_NAME = "adsmod_backend"
DEFAULT_TAURI_TARGET = "x86_64-pc-windows-msvc"


def run_command(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}"
        )


def build_sidecar_binary(project_root: Path, entrypoint: Path) -> Path:
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name",
        SIDE_CAR_NAME,
        "--collect-all",
        "torch",
        "--collect-all",
        "keras",
        "--collect-all",
        "numpy",
        "--collect-all",
        "sklearn",
        str(entrypoint),
    ]

    extra_args = os.getenv("ADSMOD_PYINSTALLER_ARGS", "").strip()
    if extra_args:
        command[3:3] = extra_args.split()

    run_command(command, cwd=project_root)

    sidecar_path = project_root / "dist" / f"{SIDE_CAR_NAME}.exe"
    if not sidecar_path.exists():
        raise RuntimeError(f"Expected sidecar binary was not found: {sidecar_path}")
    return sidecar_path


def copy_sidecar_to_tauri_binaries(sidecar_path: Path, tauri_binaries_dir: Path) -> Path:
    tauri_binaries_dir.mkdir(parents=True, exist_ok=True)

    target_triple = os.getenv("TAURI_TARGET_TRIPLE", DEFAULT_TAURI_TARGET)
    tauri_sidecar_name = f"{SIDE_CAR_NAME}-{target_triple}.exe"
    target_path = tauri_binaries_dir / tauri_sidecar_name

    shutil.copy2(sidecar_path, target_path)
    return target_path


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    entrypoint = project_root / "ADSMOD" / "server" / "desktop_entrypoint.py"
    tauri_binaries_dir = project_root / "ADSMOD" / "client" / "src-tauri" / "binaries"

    if not entrypoint.exists():
        raise RuntimeError(f"Backend desktop entrypoint not found: {entrypoint}")

    sidecar_path = build_sidecar_binary(project_root=project_root, entrypoint=entrypoint)
    tauri_sidecar = copy_sidecar_to_tauri_binaries(
        sidecar_path=sidecar_path,
        tauri_binaries_dir=tauri_binaries_dir,
    )

    print(f"Built backend sidecar: {sidecar_path}")
    print(f"Copied backend sidecar to: {tauri_sidecar}")


if __name__ == "__main__":
    main()
