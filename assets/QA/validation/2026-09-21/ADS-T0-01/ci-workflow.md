# ADS-T0-01 — current CI workflow executability

Date: 2026-09-21
Baseline: `develop` at `57baefed0b2be9e71cfa4f053226ac45ff2ffb6d`
Workflow-fix revision: `cec373a1983d2313435cf2661e56be28e6cda32a`
Hosted rerun: [35621291855](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/runs/35621291855) (attempt 2)
Final remediation revision: `b52096bcb153a7f5724493a49374d30d352c8a6b`
Final run: [35622186997](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/runs/35622186997)
Evidence strength: remote workflow diagnostic + job logs + source diff

## Scope

Restore GitHub Actions workflow evaluation and then verify that
`base-backend`, `ml-backend`, and `frontend` jobs actually start and execute
their repository-defined commands.

## Observed failure

- Run [35593534042](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/runs/35593534042) for `57baefed` completed with `failure`, zero jobs, and no job log.
- GitHub’s run annotation identifies the exact cause: `.github/workflows/ci.yml` line 20, column 3, **`'npm_config_cache' is already defined`**.
- The preceding cache-centralization run [35398762369](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/runs/35398762369) and the following Python-update run show the same zero-job/workflow-file failure. The last known successful run was [35379288650](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/runs/35379288650) on `adbe7b7b`.

GitHub treats the uppercase `NPM_CONFIG_CACHE` and lowercase
`npm_config_cache` environment names as the same key during workflow
validation. This is a configuration defect, not a backend/frontend test
failure.

The replacement run [35620065349](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/runs/35620065349) proved that workflow evaluation was repaired: all three expected jobs were created and reached repository-defined steps. It then exposed two hosted-runner configuration failures:

- `base-backend` and `ml-backend` reached dependency installation but both failed because the pinned `uv 0.11.30` could not find Python `3.14.7` in the hosted managed installations.
- `frontend` passed `npm ci`, lint, unit tests, the production build, and Edge installation. Its seven visual projects all failed before the browser tests because Edge rejected the repository-derived temporary path as too long for its singleton socket.

The rerun attempt [35621291855](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/runs/35621291855) confirmed the browser-path remediation: all three jobs were created, `frontend` passed install, lint, unit, build, Edge installation, and all seven visual projects. Both backend jobs still failed at dependency installation with the same `No interpreter found for Python 3.14.7` error after `setup-uv` set `UV_PYTHON`.

## Surgical remediation

Removed only the duplicate lowercase entry from
`.github/workflows/ci.yml`; the canonical uppercase `NPM_CONFIG_CACHE` entry
remains under the workflow-level cache contract. No job command, dependency
pin, test, or generated artifact was changed.

The follow-up remediation keeps the repository cache contract intact while
aligning both backend setup steps to the repository-local `uv 0.11.31` and
setting `TMPDIR=/tmp` only for the hosted Playwright visual step. This keeps
dependency, npm, browser-download, coverage, and repository test caches under
`runtimes/cache`; only the short-lived Linux browser socket path is relocated.

The next CI-only remediation adds `actions/setup-python@v5` for the exact
`3.14.7` interpreter before `setup-uv`, and removes the `python-version` input
from `setup-uv`. The local launcher and CI therefore retain the same explicit
Python contract while the hosted runner supplies the interpreter on PATH.

That remediation passed in final run [35622186997](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/runs/35622186997):

- `base-backend` passed setup, install, base capability validation, and cleanup.
- `ml-backend` passed setup, ML-enabled install, Ruff, backend tests, generated OpenAPI/config-schema checks, and cleanup.
- `frontend` passed npm install, lint, unit tests, production build, Edge installation, and all seven visual projects.

## Status

`PASS` — final run `35622186997` created and completed all three expected jobs,
including backend installs/validation and all seven frontend visual projects.
The Node.js 20 deprecation and future `ubuntu-latest` migration annotations are
maintenance warnings, not failures in this validation slice.

## Required follow-up

1. Preserve the final run URL, job conclusions, and commit SHA here as the
   current CI baseline.
2. Repeat this slice after future workflow, runtime, dependency, or generated
   contract changes.
3. Keep Tier 1 closed until the separate Windows launcher slice passes.
