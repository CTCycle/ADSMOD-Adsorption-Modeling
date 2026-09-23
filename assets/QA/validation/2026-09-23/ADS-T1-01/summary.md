# ADS-T1-01 — core API and capability boundaries

Date: 2026-09-23

Validated implementation SHA: `1affb39a2c4e475a8616004ee0566c30c5d189e7`

Hosted CI: [run 35832020842](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/runs/35832020842)

Status: `PASS`

Evidence strength: base-profile API smoke + focused automated tests + hosted CI

## Scope and result

Validated health/readiness, system capabilities and configuration, core
fitting route availability, and the optional ML route boundary. Capability
refresh/retry and shell gating were covered by the Angular unit suite. No
public API or product-code change was needed; the CI base-profile probe was
extended to assert the route boundary directly.

The live user-facing navigation flow remains in `ADS-T1-03`; this slice makes
no browser-workflow claim.

## Scenarios

The clean Base profile reported `datasets=true`, `fitting=true`, and
`machine_learning=false`. `/health/ready`, `/api/v1/system/configuration`,
`/api/v1/fitting/models`, and the bounded Public Data sources route returned
success. `/api/v1/training/configuration` and `/api/v1/training/status` both
returned `404`. The profile did not import `server.services.ml_container`,
Torch, Keras, or scikit-learn.

The ML-enabled hosted profile retained positive capability and training-route
coverage. The frontend unit suite exercised capability retry after transient
failure, cached response reuse, forced refresh, failed-refresh retry, and
training navigation gating.

## Validation evidence

| Gate | Result |
| --- | --- |
| Local Base-profile API smoke using disposable SQLite storage | PASS; health, configuration, fitting models, public-data source listing, capability flags, and both training-route 404s verified. |
| `test_core_runtime.py` and `test_configurations.py` in the Base environment | `12 passed, 1 deselected`; the deselected test requires the ML profile. |
| Angular `npm run test:unit` | `26 files / 68 tests passed`. |
| Hosted `base-backend` | PASS; clean no-ML installation and all Base-profile assertions. |
| Hosted `ml-backend` | PASS; complete non-E2E backend suite, Ruff, and generated contract checks. |
| Hosted `frontend` | PASS; lint, unit, preview, build, and browser layout validation. |

The existing `app/server/.venv` lacked test dependencies. The documented locked
sync command, `uv sync --locked --project app/server --all-packages --group dev`,
installed the development group into that existing Base environment. No new
virtual environment or ML extra was installed. Pytest used repository-local
basetemp under `runtimes/cache`; pre-existing untracked cache residue was left
untouched and unstaged.

GitHub reported non-blocking platform annotations for the Node.js 20 action
migration and the future `ubuntu-latest` image migration.

## Remaining Tier 1 work

- `ADS-T1-02` — SQLite/Alembic startup states and persistence across restart.
- `ADS-T1-03` — visible route, navigation, and backend recovery workflows.
