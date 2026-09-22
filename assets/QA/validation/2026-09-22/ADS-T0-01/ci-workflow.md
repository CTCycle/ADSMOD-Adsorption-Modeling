# ADS-T0-01 — current CI workflow quality gates

Date: 2026-09-22
Implementation SHA: `2eb3bc13823bedae1be1791fc980a78613d00ef1`
Hosted run: [35725502460](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/runs/35725502460)
Environment: GitHub Actions Ubuntu hosted runner; Python 3.14.7; uv 0.11.31; Node 22.13.0; Edge
Result: `PASS`
Evidence strength: exact hosted workflow plus current local gates

## Hosted result

All three jobs completed successfully on the implementation SHA:

| Job | Result | Current coverage |
| --- | --- | --- |
| `base-backend` | PASS | Locked Base installation, no ML imports, `machine_learning=false`, Public Data route, and readiness. |
| `ml-backend` | PASS | Locked ML/development installation, Ruff, non-E2E backend suite, OpenAPI generation, config-schema generation, and zero generated-artifact diff. |
| `frontend` | PASS | `npm ci`, lint, Angular unit tests, `npm run test:preview`, production build, Edge installation, and all 7 configured visual projects. |

The run included the newly added `npm run test:preview` workflow step. The
visual job completed `viewport-1440x920`, `wide-1480x920`,
`desktop-1360x900`, `compact-1200x900`, `tablet-900x900`, `narrow-768x900`,
and `mobile-600x900` successfully.

GitHub reported only platform maintenance annotations: Node.js 20 action
deprecation and the future `ubuntu-latest` image migration. Neither affected a
job result.

## Local gates

The repository-defined commands were also run locally:

- Base capability gate: `uv sync --locked --project app/server --no-dev`,
  followed by the capability/import/public-data/readiness assertions. PASS.
- ML Ruff: `uv run --project app/server --extra ml --group dev ruff check
  --isolated app/server app/tests app/scripts`. PASS.
- ML non-E2E suite: the hosted-equivalent pytest command passed `150 tests`.
- Generated contracts: `generate_openapi.py` and `generate_config_schema.py`
  completed, followed by `git diff --exit-code` for both generated files. PASS.
- Frontend: `npm ci --no-audit --no-fund`, `npm run lint`,
  `npm run test:unit` (`26 files / 68 tests`), `npm run test:preview` (`3/3`),
  `npm run build`, and the Edge layout suite (`7/7`). PASS.
- PowerShell parser and launcher contract tests: PASS.
- Development/Base profile runner: `106 passed, 1 deselected`; the deselection
  is the ML-positive `test_unified_runtime_contracts` assertion.

The first local ML pytest invocation produced `126 passed, 24 errors` because
the host attempted to enumerate the pre-existing ACL-protected global
`pytest-of-TV` temp directory. This was classified as a workflow/environment
defect, not an application defect. Re-running with an explicit disposable
repository-local basetemp passed all 150 tests. The protected cache residue was
preserved.

The first non-CI Edge invocation also hung while sharing a pre-existing local
dev-server/browser state. It was stopped, rerun in CI mode with one bounded
project, then rerun across all configured projects: `7 passed in 52.6s`.
This was classified as local runner state, not a product or hosted-workflow
failure.

## Cleanup

Generated artifacts remained unchanged. The final local runtime was restored to
the official `Standard / Base` profile. No application or launcher process was
left running, both configured ports were free, and pre-existing cache residue
was not staged, deleted, or broadened.
