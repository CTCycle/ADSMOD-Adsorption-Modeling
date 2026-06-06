# ADSMOD Troubleshooting

Last updated: 2026-06-03

## Backend Or UI Unreachable

- Check `FASTAPI_HOST`, `FASTAPI_PORT`, `UI_HOST`, and `UI_PORT` in `ADSMOD/settings/.env`.
- Confirm the launcher or backend process is running on the expected ports.

## Missing Test Dependencies

- Set `OPTIONAL_DEPENDENCIES=true` in `ADSMOD/settings/.env`.
- Rerun the launcher so dependencies are provisioned.
- Rerun the tests after the environment is updated.

## Packaging Failure

- Confirm Rust and Cargo are installed and available.
- Re-run `release\tauri\build_with_tauri.bat` after fixing the toolchain.
