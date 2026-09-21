# ADS-T0-01 — current CI workflow executability

Date: 2026-09-21
Baseline: `develop` at `57baefed0b2be9e71cfa4f053226ac45ff2ffb6d`
Evidence strength: remote workflow diagnostic + source diff; rerun pending

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

## Surgical remediation

Removed only the duplicate lowercase entry from
`.github/workflows/ci.yml`; the canonical uppercase `NPM_CONFIG_CACHE` entry
remains under the workflow-level cache contract. No job command, dependency
pin, test, or generated artifact was changed.

## Status

`PARTIAL` — the local fix is present, but the push-triggered replacement run
must prove that all three jobs are created and complete. Until then, no current
remote automated gate is considered green.

## Required follow-up

1. Push the fix with this ledger and QA evidence.
2. Record the replacement run URL, created jobs, step conclusions, and final
   commit SHA here.
3. If a job fails, classify it as configuration, environment, or product
   failure and stop before advancing to Tier 1.
