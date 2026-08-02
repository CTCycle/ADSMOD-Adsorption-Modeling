# ADSMOD Troubleshooting

Last updated: 2026-08-02

## Backend Or UI Unreachable

- Check the `runtime` host and port values in `app/resources/adsmod.json`.
- Confirm the launcher or backend process is running on the expected ports.

## Missing Test Dependencies

- Choose `Development` when selecting `Install / update dependencies` in the launcher.
- Rerun the launcher so dependencies are provisioned.
- Rerun the tests after the environment is updated.

## Frontend Preview Unreachable

- Confirm `npm run build` succeeds in `app/client`.
- Check `runtime.frontend_port` in `app/resources/adsmod.json`, then rerun `start_on_windows.ps1`.

## Training unavailable

- Core-only mode intentionally leaves Training unavailable.
- Start the ML service on `runtime.ml_port` (`6046` in the canonical file), or
  start the unified backend with `ADSMOD_ENABLE_ML=true`.
- Confirm the frontend proxy still lists `/api/training` before the catch-all
  `/api` target in `app/client/proxy.conf.cjs`.
