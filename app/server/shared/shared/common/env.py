from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from shared.common.constants import ENV_FILE_PATH


def load_environment(env_path: str | Path | None = None) -> Path | None:
    resolved = Path(env_path) if env_path is not None else Path(ENV_FILE_PATH)
    if not resolved.exists():
        return None
    load_dotenv(dotenv_path=resolved, override=True)
    return resolved
