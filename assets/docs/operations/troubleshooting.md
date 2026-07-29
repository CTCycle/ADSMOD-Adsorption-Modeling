# ADSMOD Troubleshooting

Last updated: 2026-07-11

## Backend Or UI Unreachable

- Check `FASTAPI_HOST`, `FASTAPI_PORT`, `UI_HOST`, and `UI_PORT` in `ADSMOD/settings/.env`.
- Confirm the launcher or backend process is running on the expected ports.

## Missing Test Dependencies

- Choose `Development` when selecting `Install / update dependencies` in the launcher.
- Rerun the launcher so dependencies are provisioned.
- Rerun the tests after the environment is updated.

## Frontend Preview Unreachable

- Confirm `npm run build` succeeds in `ADSMOD/app/client`.
- Check `UI_HOST` and `UI_PORT` in `settings/.env`, then rerun `start_on_windows.ps1`.
