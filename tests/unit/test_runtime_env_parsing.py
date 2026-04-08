from __future__ import annotations

from pathlib import Path

from ADSMOD.server.common.utils.variables import EnvironmentVariables


# -------------------------------------------------------------------------
def test_runtime_env_parsing_loads_host_port_and_backends(
    monkeypatch,
    tmp_path: Path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "FASTAPI_HOST=0.0.0.0",
                "FASTAPI_PORT=5001",
                "UI_HOST=127.0.0.1",
                "UI_PORT=8001",
                "KERAS_BACKEND=torch",
                'MPLBACKEND="Agg"',
            ]
        ),
        encoding="utf-8",
    )

    for key in (
        "FASTAPI_HOST",
        "FASTAPI_PORT",
        "UI_HOST",
        "UI_PORT",
        "KERAS_BACKEND",
        "MPLBACKEND",
    ):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setattr(
        "ADSMOD.server.common.utils.variables.ENV_FILE_PATH",
        str(env_path),
    )

    env_variables = EnvironmentVariables()

    assert env_variables.get("FASTAPI_HOST") == "0.0.0.0"
    assert env_variables.get("FASTAPI_PORT") == "5001"
    assert env_variables.get("UI_HOST") == "127.0.0.1"
    assert env_variables.get("UI_PORT") == "8001"
    assert env_variables.get("KERAS_BACKEND") == "torch"
    assert env_variables.get("MPLBACKEND") == "Agg"
