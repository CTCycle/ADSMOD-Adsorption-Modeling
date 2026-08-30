# ADSMOD troubleshooting

Last updated: 2026-08-30

## Backend or UI unreachable

- Check the host and ports in `app/resources/adsmod.json`.
- Confirm Core responds at `/health/ready`.
- In `core-ml` mode, confirm ML responds at its own `/health/ready`.
- Confirm the frontend preview serves the built `dist/browser/index.html`.

## Missing dependencies

Run **Install / update dependencies** in `start_on_windows.ps1`. It uses the
locked backend workspace and `npm ci`; resolve filesystem or network errors and
rerun the action rather than reusing a stale environment.

## Training unavailable

`core` mode intentionally reports training as not configured. Set the
canonical `runtime.mode` to `core-ml` and relaunch so ML starts on
`runtime.ml_port`. The frontend proxy must match `/api/v1/training` before the
general `/api/v1` target.

## Database startup failure

Inspect the Core logs below the configured storage root. Unknown or non-empty
unversioned schemas are rejected deliberately; export or repair them manually
instead of stamping a guessed revision. For PostgreSQL, verify connectivity,
credentials, and the role's database-creation permission.
