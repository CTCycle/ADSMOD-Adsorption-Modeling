# ADSMOD project status ledger

Last updated: 2026-09-30

This is the canonical current-state catalog for ADSMOD. It records validated
scope, meaningful limitations, open product decisions, evidence anchors, and
the smallest useful revalidation path. It is not a chronological QA journal.

## Baseline and branch status

The current application line in this checkout is `develop` at
`f9729efb7c2771f0d860b396e91c8eb08b5ce2b5`. It is the v3 architecture:
Angular client, one FastAPI backend, canonical configuration under `data/`,
and optional ML capabilities loaded in-process.

The local refs are not release-equivalent:

- `main` is `e8394e8`; `origin/main` is `64ee8ca`.
- `develop` is 261 commits ahead of `origin/main` in the local refs.
- `origin/main` still contains the former React/Tauri-era client and launcher
  layout, while `develop` contains the current Angular/local-web line.
- The launcher update action intentionally targets a clean `main` checkout.
  Branch synchronization and release-baseline selection therefore remain
  follow-up work; this ledger describes `develop`, not `main`.

The documentation cleanup is scoped to documentation and tracked QA artifacts.
It does not change application behavior.

## Current campaign

The stable slice definitions and evidence rules are in
[`validation/roadmap.md`](validation/roadmap.md). The current bounded
campaign has closed every tier:

| Tier | Slices | Status | Scope boundary |
| --- | --- | --- | --- |
| Tier 0 — environment and startup | `ADS-T0-01`–`ADS-T0-02` | `PASS` | Hosted CI and the Windows launcher lifecycle are covered; no production-scale stress claim |
| Tier 1 — application foundations | `ADS-T1-01`–`ADS-T1-03` | `PASS` | Core API, persistence, shell navigation, and recovery are covered |
| Tier 2 — core workflows | `ADS-T2-01`–`ADS-T2-06` | `PASS` | CSV/Excel import and fitting lifecycle are covered with disposable data |
| Tier 3 — providers | `ADS-T3-01`–`ADS-T3-05` | `PASS` for bounded scopes | NIST coverage remains partial under the accepted skip-and-report policy |
| Tier 4 — ML workflows | `ADS-T4-01`–`ADS-T4-04` | `PASS` for bounded scopes | Synthetic ML lifecycle is covered; scientific quality and release readiness are not claimed |
| Tier 5 — resilience and closure | `ADS-T5-01`–`ADS-T5-05` | `PASS` for bounded scopes | Screen-reader audio, long-duration stress, and host-specific local sub-gates remain limited |

### Evidence anchors

- Hosted run [36474039764](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/runs/36474039764) passed the three CI jobs on publication SHA `1eab9c7`. Later source changes were rechecked locally where the scope required it; this run is not treated as a blanket current-revision claim.
- The 2026-09-28 live recheck covered the ML profile, provider recovery, launcher rebuild, restart/persistence, rendered routes, keyboard/focus behavior, responsive captures, and documentation/contract checks. Its source baselines are recorded by the affected commits `dc841b4` and `efcc858`.
- The 2026-09-25 provider campaign covered bounded NIST, PubChem, and COD workflows. The 2026-09-24 campaign covered the fitting lifecycle.
- Reproducible evidence now points to repository tests, generated contracts, architecture documents, and hosted runs. Dated raw QA bundles were consolidated; current status must not depend on an ignored or machine-local artifact.

## Component status

| Concept | Status | Current guarantee | Evidence anchors | Limitation or next action |
| --- | --- | --- | --- | --- |
| `architecture.v3` | `VALIDATED` | One Angular client, one FastAPI process, one canonical configuration, repository-owned migrations, and optional in-process ML | [`architecture/system_overview.md`](architecture/system_overview.md), [`architecture/v3_migration_status.md`](architecture/v3_migration_status.md), [`../../app/client/src/app/app.routes.ts`](../../app/client/src/app/app.routes.ts), [`../../app/server/app.py`](../../app/server/app.py) | Reconcile `main` with this baseline before treating branches as interchangeable |
| `runtime.startup.windows` | `VALIDATED` | Launcher setup, profile-aware dependency reuse, readiness, rendered local preview, port conflict protection, owned-process stop, relaunch, and data-directory selection | [`runtime/startup.md`](runtime/startup.md), [`../../start_on_windows.ps1`](../../start_on_windows.ps1), [`../../app/tests/unit/test_launcher_startup.py`](../../app/tests/unit/test_launcher_startup.py), [retained live migration evidence](../QA/validation/2026-09-28/ADS-T0-02/data-directory-migration.md) | Revalidate after launcher, port, process, or startup-configuration changes |
| `runtime.installation-profiles` | `VALIDATED` | Base and ML profiles use the same backend; capability discovery gates ML routes and navigation | [`runtime/modes.md`](runtime/modes.md), [`architecture/service_boundaries.md`](architecture/service_boundaries.md), [`../../app/tests/backend/test_ml_routes.py`](../../app/tests/backend/test_ml_routes.py) | Current positive ML evidence is a small synthetic run; no representative-data quality claim |
| `backend.core.api` | `VALIDATED` | Health, capabilities, datasets, public-data, fitting, and core workflow routes use the versioned unified API | [`architecture/api_surface.md`](architecture/api_surface.md), [`../../app/tests/backend/test_core_routes.py`](../../app/tests/backend/test_core_routes.py), [`../../app/tests/e2e/test_datasets_api.py`](../../app/tests/e2e/test_datasets_api.py) | Revalidate affected routes and contracts after API or service-boundary changes |
| `persistence.database-migrations` | `VALIDATED` | SQLite/Alembic startup fails closed for unknown or incomplete schemas; records and cancelled fitting runs persist across restart | [`architecture/persistence_and_packages.md`](architecture/persistence_and_packages.md), [`../../app/tests/unit/test_database_initialization.py`](../../app/tests/unit/test_database_initialization.py), [`../../app/tests/persistence/test_database_restart.py`](../../app/tests/persistence/test_database_restart.py) | Revalidate after schema, migration, or database-runtime changes |
| `data.local-dataset` | `VALIDATED` | CSV, XLS, and XLSX import, mapping, validation, inspection, experiment switching, reload, and deletion are covered | [`runtime/configuration.md`](runtime/configuration.md), [`../../app/tests/e2e/test_dataset_import_browser.py`](../../app/tests/e2e/test_dataset_import_browser.py), [`../../app/tests/unit/test_canonical_adsorption_import.py`](../../app/tests/unit/test_canonical_adsorption_import.py) | Revalidate both workbook formats after parser, dependency, or schema changes |
| `workflow.fitting` | `VALIDATED` | Nine-model configuration, asynchronous fitting, metrics, persistence, reload, cancellation, duplicate prevention, and recovery are covered | [`../../app/tests/e2e/test_fitting_api.py`](../../app/tests/e2e/test_fitting_api.py), [`../../app/tests/unit/test_fitting_metrics.py`](../../app/tests/unit/test_fitting_metrics.py), [`operations/workflows.md`](operations/workflows.md) | Fit quality remains dataset- and parameter-dependent; lifecycle validation is not a scientific correctness claim |
| `data.public.nist` | `PARTIAL` | Bounded NIST status, index, fetch, normalization, skip reporting, and guest/host enrichment are covered | [`architecture/public_data.md`](architecture/public_data.md), [`../../app/tests/e2e/test_nist_api.py`](../../app/tests/e2e/test_nist_api.py), [`../../app/tests/unit/test_nist_repository.py`](../../app/tests/unit/test_nist_repository.py), [retained provider result](../QA/validation/2026-09-25/ADS-T3-03/provider-jobs.json) | In the bounded run 26 of 40 experiment rows persisted and 14 were skipped; do not infer unsupported conversions or full-catalog coverage |
| `data.public.pubchem-enrichment` | `VALIDATED` for bounded scope | Positive identity/property resolution, provenance, reload persistence, and graceful secondary-endpoint failure are covered | [`architecture/public_data.md`](architecture/public_data.md), [`../../app/tests/unit/test_public_data.py`](../../app/tests/unit/test_public_data.py), [retained PubChem record](../QA/validation/2026-09-25/ADS-T3-04/pubchem-record.json) | External availability is not continuous; secondary failures may omit optional enrichment |
| `data.public.cod-import` | `VALIDATED` for bounded scope | Bounded search, CIF retention, normalized structure fields, provenance, explicit association, and safe re-import are covered | [`architecture/public_data.md`](architecture/public_data.md), [`../../app/tests/unit/test_public_data.py`](../../app/tests/unit/test_public_data.py), [retained COD/CIF evidence](../QA/validation/2026-09-25/ADS-T3-05/original-cif-1011195.cif) | Natural cross-provider material matching remains unproven; validate when a natural matching fixture is available |
| `runtime.ml-capabilities` | `VALIDATED` for bounded scope | Base profile hides ML routes; the ML profile registers training/checkpoint capabilities in the same backend | [`runtime/modes.md`](runtime/modes.md), [`../../app/tests/backend/test_ml_routes.py`](../../app/tests/backend/test_ml_routes.py), [`../../app/server/openapi/backend.json`](../../app/server/openapi/backend.json) | Revalidate both profiles after optional dependency or capability-contract changes |
| `workflow.ml-training` | `VALIDATED` for bounded scope | Processed dataset/snapshot creation, a real training run, cancellation, checkpoint resume/deletion, and populated training-dashboard metrics are covered | [`../../app/tests/e2e/test_training_api.py`](../../app/tests/e2e/test_training_api.py), [`../../app/tests/unit/test_ml_jobs.py`](../../app/tests/unit/test_ml_jobs.py), [`../../app/client/src/app/features/training/pages/machine-learning-page.component.ts`](../../app/client/src/app/features/training/pages/machine-learning-page.component.ts), [retained ML profile evidence](../QA/validation/2026-09-28/ML-blocker-recheck.md) | Synthetic fixture and short run do not establish convergence, model quality, long-duration stability, or release readiness |
| `ui.shell.navigation` | `VALIDATED` | Routes, redirects, unknown-route recovery, capability-aware training navigation, backend status, Help focus, and unavailable Docs/Settings controls are covered | [`../../app/client/src/app/app.routes.spec.ts`](../../app/client/src/app/app.routes.spec.ts), [`../../app/client/src/app/layout/core-shell.component.spec.ts`](../../app/client/src/app/layout/core-shell.component.spec.ts), [`ui/experience.md`](ui/experience.md) | Revalidate affected routes and responsive states after shell or global-style changes |
| `ui.responsive-accessibility` | `VALIDATED` for rendered/keyboard/responsive scope; screen-reader sub-gate `PARTIAL` | Rendered primary routes, dense tables, narrow layouts, overflow containment, dialog focus, Escape dismissal, and trigger restoration are covered | [`../../app/client/tests/visual/application-shell.spec.ts`](../../app/client/tests/visual/application-shell.spec.ts), [`../../app/client/tests/visual/public-data-layout.spec.ts`](../../app/client/tests/visual/public-data-layout.spec.ts), [`ui/standards.md`](ui/standards.md), [retained desktop capture](../QA/validation/2026-09-28/ADS-T5-03/datasets-1280x720.png), [retained narrow capture](../QA/validation/2026-09-28/ADS-T5-03/datasets-600x900.png), [retained dialog capture](../QA/validation/2026-09-28/ADS-T5-03/import-1280x720.png) | Narrator/Speech Recap output was not observable; DOM/AX inspection is not a screen-reader PASS |
| `ui.dashboards` | `PARTIAL` | The top-level route is contained and its navigation actions work; the training dashboard is populated separately | [`../../app/client/src/app/features/dashboards/dashboards-page.component.ts`](../../app/client/src/app/features/dashboards/dashboards-page.component.ts), [`ui/experience.md`](ui/experience.md) | Decide and implement the scope of the general `/dashboards` view |
| `quality.validation-gates` | `VALIDATED` for recorded hosted and focused local scope | Hosted CI, Ruff, frontend lint/unit/preview/development build, generated-contract checks, and focused integration gates are represented | [`coding/quality_gates.md`](coding/quality_gates.md), [`operations/commands.md`](operations/commands.md), [hosted run 36474039764](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/runs/36474039764) | The current host's production Angular builder exited with `0xC0000005` and the configured visual runner hung during teardown; rerun those local sub-gates on a stable host |
| `deployment.container` | `NOT_IMPLEMENTED` | Windows local web deployment is the supported target | [`runtime/deployment.md`](runtime/deployment.md) | No action unless a container target is explicitly added to scope |

## Open limitations and decisions

| ID | Concept | Status | Durable statement | Follow-up |
| --- | --- | --- | --- | --- |
| `ISSUE-002` | `data.public.nist` | Accepted limitation | Unsupported NIST units or insufficient source metadata are skipped and reported; no inferred conversion is allowed | Revisit only with safe source metadata or an approved conversion basis |
| `ISSUE-004` | `data.public.cod-import` | Validation gap | The bounded COD import/link/re-import path passes, but natural cross-provider material matching has not been demonstrated | Test a natural matching source fixture when one exists |
| `ISSUE-005` | `ui.dashboards` | Open product decision | General `/dashboards` remains a placeholder; populated `/training/dashboard` is separate and validated | Select scope, implement, and validate desktop/mobile states |
| — | `ui.responsive-accessibility` | Validation boundary | The bounded rendered/keyboard/responsive scope passes, but audible screen-reader output was not observable | Repeat on a host with observable Narrator or equivalent output |
| — | `quality.validation-gates` | Host boundary | Hosted CI passed; local production-build and visual-runner failures are host/tooling limits, not observed application failures | Repeat on a stable host before claiming those local sub-gates |
| — | `workflow.ml-training` | Evidence boundary | Current ML validation is synthetic and bounded | Use representative data and longer runs for scientific or release claims |
| — | `repository.branch-baseline` | Integration follow-up | `develop` and `main` are not equivalent; this ledger intentionally describes `develop` | Choose and execute the release-baseline reconciliation, then rerun documentation/contract checks |

## Resolved decisions worth preserving

| Decision | Current rule | Source |
| --- | --- | --- |
| Backend topology | One FastAPI process owns core and optional ML routes; there is no legacy service-to-service boundary | [`architecture/findings_and_remediation.md`](architecture/findings_and_remediation.md), [`architecture/v3_migration_status.md`](architecture/v3_migration_status.md) |
| Configuration ownership | `data/adsmod.json` is canonical; the generated schema is a validation aid, not a second authority | [`runtime/configuration.md`](runtime/configuration.md), [`../../data/adsmod.json`](../../data/adsmod.json) |
| Database safety | Unknown, empty-but-nonfresh, and incomplete schemas fail closed instead of being inferred | [`runtime/startup.md`](runtime/startup.md), [`../../app/server/repositories/database/migrator.py`](../../app/server/repositories/database/migrator.py) |
| Import support | `.csv`, `.xls`, and `.xlsx` are the configured upload formats and have real parser/browser coverage | [`runtime/configuration.md`](runtime/configuration.md), [`../../app/tests/unit/test_canonical_adsorption_import.py`](../../app/tests/unit/test_canonical_adsorption_import.py) |
| Fitting cancellation | Cancelled fitting runs use the migrated SQLite status constraint and remain recoverable for later work | [`../../app/server/migrations/versions/20260924_fitting_cancellation.py`](../../app/server/migrations/versions/20260924_fitting_cancellation.py), [`../../app/tests/e2e/test_fitting_api.py`](../../app/tests/e2e/test_fitting_api.py) |
| ML snapshots | Grouped training data preserves required uploaded features and same-content snapshot creation is idempotent | [`../../app/tests/unit/test_data_processing.py`](../../app/tests/unit/test_data_processing.py), [`../../app/tests/unit/test_ml_jobs.py`](../../app/tests/unit/test_ml_jobs.py) |
| Contract generation | Canonical OpenAPI includes ML routes; Base-profile generation fails closed rather than overwriting it | [`../../app/scripts/generate_openapi.py`](../../app/scripts/generate_openapi.py), [`operations/commands.md`](operations/commands.md) |

## Evidence rules

- `VALIDATED` always names the scope that was exercised; it does not mean
  production readiness, scientific correctness, or continuous provider uptime.
- Unit, lint, static, schema, or build results cannot alone prove a live browser,
  provider, hardware, or long-running workflow.
- A later source change lowers confidence until the affected slice is rerun.
- Keep active limitations here; move resolved findings into the compact decision
  table or the relevant concept document.
- Keep detailed run artifacts out of this ledger. Use repository tests, hosted
  URLs, canonical contracts, and only irreplaceable retained evidence.

## Revalidation map

| Change area | Revalidate |
| --- | --- |
| Launcher, ports, startup, cache roots, or data-directory selection | `runtime.startup.windows`, readiness, rendered load, stop, and final listener state |
| Routes, service boundaries, API contracts, schema, or migrations | `backend.core.api`, `persistence.database-migrations`, affected workflow, and generated contracts |
| Import parser, mapping, units, or inspection UI | `data.local-dataset`, persistence reload, and fitting selection |
| NIST, PubChem, or COD adapters | The affected provider concept, including provenance and explicit external-failure behavior |
| ML dependencies, capability discovery, training data, models, or checkpoints | `runtime.ml-capabilities` and the full positive `workflow.ml-training` path |
| Angular routes, shared styles, responsive layout, or accessibility states | `ui.shell.navigation`, affected component tests, and rendered/browser viewport checks |
| Documentation, generated snapshots, or validation organization | `ADS-T5-05`, link/reference scan, and this ledger |
