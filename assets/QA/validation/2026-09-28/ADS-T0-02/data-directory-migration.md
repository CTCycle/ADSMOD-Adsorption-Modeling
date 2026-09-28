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
  and runtime recheck passed: **37 passed**.
- PowerShell parser validation passed for `start_on_windows.ps1`.
- The Node proxy loaded `data/adsmod.json` by default and loaded the same file
  when `ADSMOD_DATA_DIR` was set explicitly.
- Configuration schema generation reproduced the moved schema byte-for-byte;
  its Git blob hash remained `48e6864c8884d71a31a87c0f5ec8d54d0bf16854`.
- Ruff passed for all changed Python files.

## Live launcher recheck

- Baseline: `efcc858c9b41cca8d44356cc4e62baefb88a39b3` on `develop`.
- The default run reused the current dependency/build manifests, loaded
  `data/adsmod.json`, passed the backend readiness probe, served the production
  preview, and rendered `/datasets` in the Codex in-app Browser. The visible
  page showed the dataset workspace, navigation, and `Backend Online`.
- The alternate run used `-DataPath` with a disposable copied configuration.
  The backend process command line confirmed the selected `adsmod.json`,
  readiness returned HTTP 200 with `state: ready`, the preview root returned
  HTTP 200, and the reloaded `/datasets` page again showed the workspace and
  `Backend Online`.
- Both sessions stopped their launcher-owned backend/frontend processes. Ports
  `6045` and `5173` were free after each stop, and the disposable alternate
  directory was removed.

## Final status and boundary

`ADS-T0-02` is **PASS** for the stated launcher/data-directory scope. No
implementation defect was found during the live recheck. The evidence does
not claim continuous provider availability, long-duration load, or model
quality; those remain outside this slice.

The protected shared Windows pytest temp residue was not modified; validation
used the repository-local `runtimes/cache/pytest-tmp-data-rename-20260928`
basetemp.
