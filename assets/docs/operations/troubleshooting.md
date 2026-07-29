# ADSMOD Troubleshooting

Last updated: 2026-07-29

## Backend Or UI Unreachable

- Check the `runtime` host and port values in `app/resources/adsmod.json`.
- Confirm the launcher or backend process is running on the expected ports.

## Missing Test Dependencies

- Choose `Development` when selecting `Install / update dependencies` in the launcher.
- Rerun the launcher so dependencies are provisioned.
- Rerun the tests after the environment is updated.

## Frontend Preview Unreachable

- Confirm `npm run build` succeeds in `ADSMOD/app/client`.
- Check `runtime.frontend_port` in `app/resources/adsmod.json`, then rerun `start_on_windows.ps1`.
