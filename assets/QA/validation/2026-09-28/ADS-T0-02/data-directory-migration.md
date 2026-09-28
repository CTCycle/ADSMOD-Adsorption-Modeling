# ADS-T0-02 data-directory migration validation

Date: 2026-09-28

## Scope

The repository-level runtime-data directory is now `data/`. The launcher,
frontend development proxy, batch test runner, CI workflow, editor settings,
tests, documentation, and runnable QA helpers use the same default. The
override contract is now `-DataPath`/`-DataDir`/`-DataDirectory` or
`ADSMOD_DATA_DIR`.

## Evidence

- `data/adsmod.json`, `data/adsmod.schema.json`, both sentinel files, and the
  existing local `database.db` are present; the former directory is absent.
- The focused path, launcher, frontend, configuration, backend, persistence,
  and runtime tests passed: **37 passed**.
- PowerShell parser validation passed for `start_on_windows.ps1`.
- The Node proxy loaded `data/adsmod.json` by default and loaded the same file
  when `ADSMOD_DATA_DIR` was set explicitly.
- Configuration schema generation reproduced the moved schema byte-for-byte;
  its Git blob hash remained `48e6864c8884d71a31a87c0f5ec8d54d0bf16854`.
- Ruff passed for all changed Python files.

## Boundary

The full interactive launcher lifecycle was not rerun for this path-only
migration. The launcher component remains `WORKING` until a live launcher run
covers the new default and selected alternate data path.

The protected shared Windows pytest temp residue was not modified; validation
used the repository-local `runtimes/cache/pytest-tmp-data-rename-20260928`
basetemp.
