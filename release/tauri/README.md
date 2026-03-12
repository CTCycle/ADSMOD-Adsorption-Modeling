# ADSMOD Desktop Packaging

This repository follows the XREPORT Windows Tauri packaging pattern with ADSMOD-specific runtime payloads.

## Maintainer entrypoints

Prepare the desktop runtime profile:

```bat
copy /Y ADSMOD\settings\.env.local.tauri.example ADSMOD\settings\.env
```

Provision the portable build/runtime toolchains if needed:

```bat
ADSMOD\start_on_windows.bat
```

Ensure Rust toolchain prerequisites are met:

- `cargo` must be installed and reachable in `PATH`.
- A default toolchain must be configured (recommended `stable-x86_64-pc-windows-msvc`).

Regenerate desktop icon assets from the shared web favicon:

```bat
cd ADSMOD\client
npm run tauri:icon
```

Build the packaged desktop artifacts:

```bat
release\tauri\build_with_tauri.bat
```

Clean generated desktop outputs only:

```bat
cd ADSMOD\client
npm run tauri:clean
```

## Build flow

`release\tauri\build_with_tauri.bat` is the only supported packaging entrypoint. It validates the embedded Python, `uv`, and portable Node.js runtimes plus `runtimes\uv.lock`, creates the short staging tree at `ADSMOD\client\src-tauri\r`, copies `pyproject.toml`, stages `runtimes\uv.lock` into bundled `uv.lock`, copies `ADSMOD\resources\database.db`, junctions the shipped runtime directories from root `runtimes\`, installs frontend dependencies with the repo-local Node runtime, runs `npm run tauri:build:release`, then cleans the staging tree and exports public artifacts to `release\windows`.

Node.js is build-time only and is not bundled into the shipped desktop payload.

## Runtime tree expected by the packaged app

The explicit Tauri resource map reconstructs this runtime layout:

```text
<runtime root>/
  pyproject.toml
  uv.lock
  ADSMOD/
    server/
    scripts/
    settings/
    client/dist/
    resources/
      checkpoints/
      database.db
  runtimes/
    .venv/
    python/
    uv/
```

The Rust launcher resolves a valid workspace by looking for both `pyproject.toml` and `ADSMOD/server/app.py`.

## Packaged startup behavior

The Tauri shell starts on `about:blank`, renders a Rust-driven splash screen, then reads `ADSMOD\settings\.env`, prefers a workspace that already contains `runtimes\.venv\Scripts\python.exe`, falls back to a writable per-user runtime root if needed, reuses an existing runtime-local venv when available, otherwise runs `uv sync --python <bundled-python> --frozen` with a fallback `uv sync --frozen`, launches `python -m uvicorn ADSMOD.server.app:app`, polls the backend until ready, redirects the window to `/`, and kills the backend process tree on desktop exit.

The FastAPI app serves the packaged SPA from `ADSMOD\client\dist`, exposes API routes under both their original paths and `/api`, and falls back to `/docs` only when packaged frontend assets are unavailable.

## Public output

`release\tauri\scripts\export-windows-artifacts.ps1` copies the user-facing outputs to:

- `release\windows\installers`
- `release\windows\portable`

The portable export contains the desktop executable plus `ADSMOD`, `runtimes`, `pyproject.toml`, `uv.lock`, and `_up_` when present. Keep the portable payload together.
