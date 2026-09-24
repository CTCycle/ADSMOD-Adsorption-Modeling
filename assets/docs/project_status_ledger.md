# ADSMOD Project Status Ledger

Last updated: 2026-09-24

This is the canonical current operational status catalog for ADSMOD. It is a
compact index of what is working, validated, partial, blocked, unvalidated, or
not implemented. Detailed architecture, debugging narratives, implementation
plans, and long validation logs remain in their dedicated documents.

`ADS-T1-01` passed on implementation SHA
`1affb39a2c4e475a8616004ee0566c30c5d189e7`. `ADS-T1-02` passed on tested code
SHA `ed80c0975e583cd9842338fca5f59157c4071f97`; hosted workflow run
`35852874401` passed all three jobs. `ADS-T1-03` passed on implementation SHA
`6154532dd6628bb70e98fae6fa427e2d9108cd5f`; hosted workflow run `35876297668`
passed all three jobs. The official Windows launcher evidence
for `ADS-T0-02` remains anchored to `2eb3bc13823bedae1be1791fc980a78613d00ef1`;
no launcher source changed since that validation. Pre-existing untracked cache
residue, protected paths, and unrelated data were preserved and are not part
of the evidence claim.

## Maintenance rules

1. Inspect this ledger before substantial implementation or validation work.
2. Use it to find known defects and previously validated behavior before
   duplicating investigation.
3. Update affected entries after implementation, remediation, or a regression.
4. Add or refresh evidence after meaningful tests, browser runs, or manual
   checks; link the detailed artifact instead of copying it here.
5. Never mark a component `VALIDATED` without evidence that supports the
   claimed scope.
6. Downgrade a status when a regression or an evidence boundary is discovered.
7. Close or move an issue to the historical section only after remediation and
   successful revalidation.
8. Do not create duplicate issue entries for the same underlying defect or
   validation gap.
9. Keep current limitations in the active fields and keep resolved findings out
   of the active issue catalog.
10. Keep this ledger synchronized with the current repository, configuration,
    provider availability, and validation artifacts.

## Validation campaign checkpoint

The long-term campaign and stable slice catalog live in
[`validation/roadmap.md`](validation/roadmap.md). Tier 0 is the first gate and
is now closed; Tier 1 is now closed. Tier 2 is now closed: `ADS-T2-01` through
`ADS-T2-03` passed on implementation commit `c876a063`, while `ADS-T2-04`
through `ADS-T2-06` passed on the 2026-09-24 working tree based on `9fde82b`.

| Tier | Slice IDs | Status | Current gate |
| --- | --- | --- | --- |
| Tier 0 — environment and startup | `ADS-T0-01`, `ADS-T0-02` | `PASS` / `PASS` | CI run `35910483168` passed all three jobs on `270e94d`; the official launcher lifecycle and conflict/race evidence remains at `2eb3bc1`, with no launcher source changes since. |
| Tier 1 — application foundations | `ADS-T1-01`–`ADS-T1-03` | `PASS` | `ADS-T1-01` passed on `1affb39`; `ADS-T1-02` passed on `ed80c09`; `ADS-T1-03` passed on `6154532` with hosted run `35876297668`. |
| Tier 2 — core product workflows | `ADS-T2-01`–`ADS-T2-06` | `PASS` | `ADS-T2-01`–`ADS-T2-03` passed on `c876a06`; `ADS-T2-04`–`ADS-T2-06` passed in the 2026-09-24 live fitting campaign. |
| Tier 3 — feature families/providers | `ADS-T3-01`–`ADS-T3-05` | `UNTESTED` | Follow the provider-specific positive/degraded evidence rules. |
| Tier 4 — ML workflows | `ADS-T4-01`–`ADS-T4-04` | `UNTESTED` | Requires an ML-enabled runtime and valid positive fixture. |
| Tier 5 — resilience and closure | `ADS-T5-01`–`ADS-T5-05` | `UNTESTED` | Final evidence, contract, and ledger reconciliation. |

### Tier 0 slice ledger

| Slice ID | Capability | Feature exists | Exercised | Status | Baseline revision | Issues | Evidence | Remaining gap | Evidence strength |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ADS-T0-01` | Current GitHub Actions quality gates | YES | YES | `PASS` | `270e94df63969feb6fbe81b081dc7d3e85744c2d` | All three jobs passed; hosted Node.js and runner-image annotations are platform maintenance warnings | [`ADS-T0-01/ci-workflow.md`](../QA/validation/2026-09-22/ADS-T0-01/ci-workflow.md); [hosted run 35910483168](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/runs/35910483168) | — | hosted three-job CI |
| `ADS-T0-02` | Official Windows launcher lifecycle and static preview | YES | YES | `PASS` | `2eb3bc13823bedae1be1791fc980a78613d00ef1` | Chrome direct JSON navigation was blocked by `ERR_BLOCKED_BY_CLIENT`; rendered browser and direct local proxy checks passed | [`tier-0-summary.md`](../QA/validation/2026-09-22/tier-0-summary.md); [`ADS-T0-02/launcher-lifecycle.md`](../QA/validation/2026-09-22/ADS-T0-02/launcher-lifecycle.md) | — | official launcher + browser + HTTP + process/port checks |

### Tier 1 slice ledger

| Slice ID | Capability | Feature exists | Exercised | Status | Baseline revision | Issues | Evidence | Remaining gap | Evidence strength |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ADS-T1-01` | Core API and capability boundaries | YES | YES | `PASS` | `1affb39a2c4e475a8616004ee0566c30c5d189e7` | None observed | [`ADS-T1-01/summary.md`](../QA/validation/2026-09-23/ADS-T1-01/summary.md); [hosted run 35832020842](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/runs/35832020842) | — | local API smoke + hosted CI + frontend unit |
| `ADS-T1-02` | SQLite/Alembic startup states and dataset persistence across application restart | YES | YES | `PASS` | `ed80c0975e583cd9842338fca5f59157c4071f97` | Unknown stamped revisions are rejected through the database migration error contract; no unresolved issue observed | [`ADS-T1-02/summary.md`](../QA/validation/2026-09-23/ADS-T1-02/summary.md); [hosted run 35852874401](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/runs/35852874401) | — | focused SQLite/Alembic tests + app lifespan integration + hosted CI |
| `ADS-T1-03` | Shell routes, redirects, profile-aware Training navigation, Help focus/closing, backend status recovery, and unavailable Docs/Settings controls | YES | YES | `PASS` | `6154532dd6628bb70e98fae6fa427e2d9108cd5f` | None observed | [`ADS-T1-03/summary.md`](../QA/validation/2026-09-23/ADS-T1-03/summary.md); [hosted run 35876297668](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/runs/35876297668) | — | frontend unit/lint/build + official launcher browser/HTTP + hosted CI |

### Tier 2 slice ledger

| Slice ID | Capability | Feature exists | Exercised | Status | Baseline revision | Issues | Evidence | Remaining gap | Evidence strength |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ADS-T2-01` | CSV import, mapping, validation, persistence, experiment switching, reload, and deletion | YES | YES | `PASS` | `c876a063d3fcef4de1ea074aaebba861da4b2c4c` | None observed | [`ADS-T2-01/summary.md`](../QA/validation/2026-09-23/ADS-T2-01/summary.md) | — | Browser E2E + rendered in-app browser + isolated persistence |
| `ADS-T2-02` | Real `.xls` and `.xlsx` binary import through persisted inspection and deletion | YES | YES | `PASS` | `c876a063d3fcef4de1ea074aaebba861da4b2c4c` | None observed | [`ADS-T2-02/summary.md`](../QA/validation/2026-09-23/ADS-T2-02/summary.md) | — | Workbook unit coverage + browser E2E + rendered in-app browser |
| `ADS-T2-03` | Empty/corrupt uploads, missing required mappings, malformed values, unsupported types, and parser-boundary regressions | YES | YES | `PASS` | `c876a063d3fcef4de1ea074aaebba861da4b2c4c` | None observed in the exercised boundaries | [`ADS-T2-03/summary.md`](../QA/validation/2026-09-23/ADS-T2-03/summary.md) | — | API E2E + parser unit regression |
| `ADS-T2-04` | Dataset/experiment selection, fitting configuration, all nine model cards, and parameter forms | YES | YES | `PASS` | `9fde82b` + working tree | None observed; corrected parameter range produced a quality fit | [`ADS-T2-04/summary.md`](../QA/validation/2026-09-24/ADS-T2-04/summary.md) | — | Official-launcher rendered UI + fitting API + frontend/backend checks |
| `ADS-T2-05` | Positive asynchronous fitting, metrics/result rendering, persistence, and reload | YES | YES | `PASS` | `9fde82b` + working tree | Fixed result loss after page reload by restoring the last completed run | [`ADS-T2-05/summary.md`](../QA/validation/2026-09-24/ADS-T2-05/summary.md) | — | Official-launcher rendered UI and reload + persisted API result + frontend checks |
| `ADS-T2-06` | Fitting cancellation, duplicate-job prevention, recovery, and clean follow-up execution | YES | YES | `PASS` | `9fde82b` + working tree | Fixed cancelled-status SQLite constraint failure with revision `20260924_fitting_cancel` | [`ADS-T2-06/summary.md`](../QA/validation/2026-09-24/ADS-T2-06/summary.md) | — | Official-launcher live cancellation + duplicate rejection + recovery + focused tests |

## Status taxonomy

| Status | Meaning |
| --- | --- |
| `VALIDATED` | Implemented and confirmed through meaningful testing for the stated scope. |
| `WORKING` | Believed to work from implementation or limited evidence, but the stated scope is not fully validated on the current baseline. |
| `PARTIAL` | Implemented, but incomplete, degraded, or valid for only part of the expected behavior. |
| `BROKEN` | Known not to work correctly for the stated scope. |
| `BLOCKED` | Cannot currently be completed or positively validated because of a dependency, missing fixture, unavailable service, hardware constraint, or similar external blocker. |
| `UNVALIDATED` | Implementation or an intended capability exists, but the available evidence is insufficient to claim that it works. |
| `NOT_IMPLEMENTED` | The expected capability is currently absent. |
| `DEPRECATED` | Retained only for compatibility or scheduled for removal. |

`BROKEN` is reserved for observed failure. A missing test is not evidence of a
broken feature. `DEPRECATED` is included for taxonomy completeness; no current
component is classified that way in this baseline.

## Evidence rules

- Implemented, observed working, and formally validated are different claims.
- Unit, lint, static, schema, or build checks can support `WORKING` or a
  narrower validated sub-scope, but do not alone prove a live browser,
  provider, hardware, or long-running workflow.
- `Last Validated` records the latest meaningful evidence for the stated scope.
  If the component changes afterward, keep the older date as provenance and
  lower the status until the current checkout is retested.
- `—` means that no active blocker or next action is recorded for that field.

## Current component ledger

The entries below are the current state, not a chronological change log.

| Component | Status | Scope | Evidence | Known Issues | Blocker | Last Validated | Validation Level | Related Docs | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `runtime.startup.windows` | `VALIDATED` | Official Windows launcher, static preview, backend/frontend lifecycle, aggregate port ownership, browser opening, stop, relaunch, process cleanup, stale-state repair, profile preservation, and readiness failure cleanup. | [`ADS-T0-02/launcher-lifecycle.md`](../QA/validation/2026-09-22/ADS-T0-02/launcher-lifecycle.md) records the current SHA’s normal lifecycle, live listeners, full deterministic conflict matrix, invalidation matrix, ML repair, early process failures, and final free ports. | Direct JSON-only Chrome navigation is blocked by the browser client; proxy JSON was verified with direct local HTTP. | — | 2026-09-22 | Manual E2E + browser + HTTP + process/port checks | [`validation/roadmap.md`](validation/roadmap.md); [`runtime/startup.md`](runtime/startup.md); [`ADS-T0-02/launcher-lifecycle.md`](../QA/validation/2026-09-22/ADS-T0-02/launcher-lifecycle.md) | Revalidate after future launcher, port, process, or startup changes. |
| `runtime.installation-profiles` | `VALIDATED` | Base and ML dependency profiles use the same FastAPI process; capability discovery gates ML routes and navigation. | The 2026-09-23 `ADS-T1-01` evidence records clean Base and hosted ML profile checks, including capability flags and training-route boundaries; earlier full profile and browser evidence remains linked in the historical validation reports. | Positive long-running training remains a separate blocked scope. | — | 2026-09-23 | Hosted profile CI + focused API/unit | [`ADS-T1-01/summary.md`](../QA/validation/2026-09-23/ADS-T1-01/summary.md); [`architecture/v3_migration_status.md`](architecture/v3_migration_status.md); [`runtime/modes.md`](runtime/modes.md); [`../../app/tests/backend/test_ml_routes.py`](../../app/tests/backend/test_ml_routes.py) | Revalidate when the backend package, optional dependency extra, or capability contract changes. |
| `backend.core.api` | `VALIDATED` | Unified backend health, capabilities, datasets, public-data, fitting, and core workflow routes. | `ADS-T1-01` rechecked Base-profile readiness, capabilities, system configuration, fitting models, and public-data source availability on the current implementation; 2026-09-16 package-cutover evidence retains the broader workflow E2E coverage. | External-provider health is intentionally reported separately from local API readiness. | — | 2026-09-23 | Automated integration + historical E2E | [`ADS-T1-01/summary.md`](../QA/validation/2026-09-23/ADS-T1-01/summary.md); [`architecture/api_surface.md`](architecture/api_surface.md); [`architecture/v3_migration_status.md`](architecture/v3_migration_status.md); [`../../app/tests/backend/test_core_routes.py`](../../app/tests/backend/test_core_routes.py); [`adsmod-end-to-end-ui-system-validation-2026-09-10.md`](../QA/adsmod-end-to-end-ui-system-validation-2026-09-10.md) | Run the relevant API and browser regression gates after route, contract, or service-boundary changes. |
| `persistence.database-migrations` | `VALIDATED` | SQLite Alembic startup, canonical schema, locking, dataset persistence across application restart, and cancelled fitting-run state. | `ADS-T1-02` passed 22 focused local tests and hosted run `35852874401`; the 2026-09-24 `ADS-T2-06` run applied `20260924_fitting_cancel`, persisted a cancelled run, and passed 20 focused DB/job/fitting/restart tests. | — | — | 2026-09-24 | Focused automated + application lifespan integration + hosted CI + live migration | [`ADS-T1-02/summary.md`](../QA/validation/2026-09-23/ADS-T1-02/summary.md); [`ADS-T2-06/summary.md`](../QA/validation/2026-09-24/ADS-T2-06/summary.md); [`../../app/tests/unit/test_database_initialization.py`](../../app/tests/unit/test_database_initialization.py); [`../../app/tests/persistence/test_database_restart.py`](../../app/tests/persistence/test_database_restart.py); [`architecture/persistence_and_packages.md`](architecture/persistence_and_packages.md); [`runtime/startup.md`](runtime/startup.md) | Revalidate migration startup and disposable-database persistence after schema or database-runtime changes. |
| `ui.shell.navigation` | `VALIDATED` | Landing and Public Data redirects, top-level routes, unknown-route recovery, profile-aware Training navigation, backend status messaging/recovery, Help focus containment/return, unavailable Docs/Settings controls, and compact shell visibility. | `ADS-T1-03` passed on code SHA `6154532` with local frontend gates, official-launcher in-app browser interactions, HTTP checks, and hosted run `35876297668`; rendered states and exact viewport are recorded in the summary. | — | — | 2026-09-23 | Frontend unit/lint/build + official launcher + browser/HTTP + hosted CI | [`ADS-T1-03/summary.md`](../QA/validation/2026-09-23/ADS-T1-03/summary.md); [`browser-state.md`](../QA/validation/2026-09-23/ADS-T1-03/browser-state.md); [`ui/experience.md`](ui/experience.md); [`ui/standards.md`](ui/standards.md) | Revalidate the affected routes and responsive states after navigation, shell, or global-style changes. |
| `data.local-dataset.csv-lifecycle` | `VALIDATED` | CSV preview, column detection, mapping, validation, save, persistence, inspection, experiment switching, fresh-session reload, and deletion. | [`ADS-T2-01/summary.md`](../QA/validation/2026-09-23/ADS-T2-01/summary.md) records the current browser lifecycle and persistence evidence, including the successful DELETE response and post-reload absence. | None observed in the exercised CSV lifecycle. | — | 2026-09-23 | Browser E2E + rendered in-app browser + isolated persistence | [`ADS-T2-01/summary.md`](../QA/validation/2026-09-23/ADS-T2-01/summary.md); [`../../app/tests/e2e/test_dataset_import_browser.py`](../../app/tests/e2e/test_dataset_import_browser.py); [`operations/workflows.md`](operations/workflows.md) | Re-run the relevant import and persistence flow after parser, schema, or inspector changes. |
| `data.local-dataset.excel-import` | `VALIDATED` | Real `.xls` and `.xlsx` binary upload, parsing, mapping, validation, save, persisted inspection, experiment switching, reload, and deletion. | Both real workbook fixtures completed the canonical browser path; unit coverage confirmed row/column counts and normalized values. | None observed in the exercised Excel import flow. | — | 2026-09-23 | Workbook unit coverage + browser E2E + isolated persistence | [`ADS-T2-02/summary.md`](../QA/validation/2026-09-23/ADS-T2-02/summary.md); [`../../app/tests/e2e/test_dataset_import_browser.py`](../../app/tests/e2e/test_dataset_import_browser.py); [`../../app/tests/unit/test_canonical_adsorption_import.py`](../../app/tests/unit/test_canonical_adsorption_import.py); [`runtime/configuration.md`](runtime/configuration.md) | Revalidate both workbook formats after parser, dependency, schema, or inspector changes. |
| `workflow.fitting` | `VALIDATED` | Dataset/experiment selection, nine-model configuration, asynchronous fitting, cancellation, duplicate prevention, recovery, result metrics, persistence, reload, and reset. | The 2026-09-24 official-launcher browser campaign validated all nine cards, a configured fit, restored persisted results after reload, duplicate rejection, persisted cancellation, and a clean follow-up. See the three ADS-T2 summaries. | Individual model fit quality remains dataset- and parameter-dependent; one narrow-range fit had poor metrics, and the validated follow-up was lifecycle proof rather than a quality claim. | — | 2026-09-24 | Focused automated + official-launcher live UI/API + rendered reload | [`ADS-T2-04/summary.md`](../QA/validation/2026-09-24/ADS-T2-04/summary.md); [`ADS-T2-05/summary.md`](../QA/validation/2026-09-24/ADS-T2-05/summary.md); [`ADS-T2-06/summary.md`](../QA/validation/2026-09-24/ADS-T2-06/summary.md); [`../../app/tests/e2e/test_fitting_api.py`](../../app/tests/e2e/test_fitting_api.py); [`operations/workflows.md`](operations/workflows.md) | Revalidate the fitting flow after parameter, optimizer, lifecycle, persistence, or UI changes. |
| `data.public.nist` | `PARTIAL` | NIST health, index/fetch jobs, canonical unit mapping, persistence, and truthful UI counts. | The 2026-09-02 and 2026-09-10 runs exercised live status and fetch paths; the latter recorded requested/fetched/local/skipped counts. | Provider health can be false or unavailable; 14 records in the validated acquisition were skipped because their units or metadata could not be mapped to canonical units. | Full coverage depends on external NIST availability and source measurements that can be converted safely. | 2026-09-10 | Integration + manual E2E | [`architecture/public_data.md`](architecture/public_data.md); [`../../app/tests/e2e/test_nist_api.py`](../../app/tests/e2e/test_nist_api.py); [`adsmod-end-to-end-ui-system-validation-2026-09-10.md`](../QA/adsmod-end-to-end-ui-system-validation-2026-09-10.md) | Preserve explicit skip counts; add or validate broader unit-normalization/coverage handling and rerun against a reachable provider. |
| `data.public.pubchem-enrichment` | `UNVALIDATED` | Positive PubChem identity resolution, enrichment, properties, and structure retrieval from local records. | Provider architecture and unit-test inventory exist, but no recent positive live enrichment evidence was found in the inspected QA reports. | Secondary PubChem endpoints may be unavailable even when a primary compound resolves; no current application defect is established. | Current positive provider evidence is missing and the remote service is external. | — | Unit only | [`architecture/public_data.md`](architecture/public_data.md); [`../../app/tests/unit/test_public_data.py`](../../app/tests/unit/test_public_data.py) | Add deterministic mocked coverage for normalization/failure branches and a bounded live enrichment run when provider access is available. |
| `data.public.cod-import` | `UNVALIDATED` | Successful COD search result, CIF import, normalized persistence, and material/structure linking. | The 2026-09-10 run validated a bounded no-result search (`items: []`) but did not capture a successful import/link. | Successful provider-result handling and linking remain unproven; this is a validation gap, not an observed failure. | A deterministic COD fixture/provider stub or reachable record is needed. | 2026-09-10 (no-result path only) | Integration + manual E2E (negative path) | [`architecture/public_data.md`](architecture/public_data.md); [`adsmod-end-to-end-ui-system-validation-2026-09-10.md`](../QA/adsmod-end-to-end-ui-system-validation-2026-09-10.md) | Validate count/search/import/persistence/linking with a bounded positive fixture and browser evidence. |
| `runtime.ml-capabilities` | `VALIDATED` | Optional ML import boundary, capability response, route registration, and base-profile unavailability behavior. | The 2026-09-23 hosted profiles verified Base capabilities false, absent training routes, and ML-profile capability/routes; frontend capability retry/refresh and shell gating unit coverage passed. | This status does not claim a successful training run; that is tracked by `workflow.ml-training`. | — | 2026-09-23 | Hosted profile CI + frontend unit | [`ADS-T1-01/summary.md`](../QA/validation/2026-09-23/ADS-T1-01/summary.md); [`runtime/modes.md`](runtime/modes.md); [`../../app/tests/backend/test_ml_routes.py`](../../app/tests/backend/test_ml_routes.py); [`../../app/tests/unit/test_ml_boundary.py`](../../app/tests/unit/test_ml_boundary.py); [`adsmod-end-to-end-audit-2026-09-10.md`](../QA/adsmod-end-to-end-audit-2026-09-10.md) | Revalidate both dependency profiles whenever optional imports, capability contracts, or route registration changes. |
| `workflow.ml-training` | `BLOCKED` | Positive processed-dataset creation, training, checkpoint creation, resume, and populated dashboard metrics. | The negative build path was live-validated and correctly returned `Training data missing adsorbate_SMILE values.`; automated ML route tests passed, but no positive live run is recorded. | Positive training, checkpoint, resume, and populated-dashboard behavior remain unproven; this is not evidence that the implementation is broken. | A valid `adsorbate_SMILE`-bearing fixture plus an ML-enabled runtime and suitable local hardware/compute are required. | 2026-09-10 (negative validation only) | Integration + manual E2E (negative path) | [`runtime/modes.md`](runtime/modes.md); [`operations/workflows.md`](operations/workflows.md); [`../../app/tests/e2e/test_training_api.py`](../../app/tests/e2e/test_training_api.py); [`adsmod-end-to-end-ui-system-validation-2026-09-10.md`](../QA/adsmod-end-to-end-ui-system-validation-2026-09-10.md) | Add an isolated valid fixture and run dataset build, training, checkpoint, resume, and dashboard-positive checks. |
| `ui.dashboards` | `PARTIAL` | Top-level dashboards route and populated training-dashboard experience. | The 2026-09-10 run exercised dashboard/training states and documented the current route behavior. | The top-level `/dashboards` view remains a placeholder; populated training metrics were not verified because positive training is blocked. | Product scope is needed for the placeholder; populated metrics share the ML-training blocker. | 2026-09-10 | Manual E2E (placeholder/empty states) | [`operations/workflows.md`](operations/workflows.md); [`../../app/client/src/app/features/dashboards/dashboards-page.component.ts`](../../app/client/src/app/features/dashboards/dashboards-page.component.ts); [`adsmod-end-to-end-ui-system-validation-2026-09-10.md`](../QA/adsmod-end-to-end-ui-system-validation-2026-09-10.md) | Decide the top-level dashboard scope, then implement and validate it; revalidate populated training metrics after `ISSUE-001`. |
| `quality.validation-gates` | `VALIDATED` | Backend/frontend tests, lint, build, generated contracts, visual checks, profile-correct runner behavior, and repository-local cache layout. | Hosted run `35910483168` on SHA `270e94d` passed the frontend, Base backend, and ML backend jobs; local focused import tests, Ruff, lock consistency, and whitespace checks also passed. | Hosted Node.js 20 and future Ubuntu image annotations are platform maintenance warnings. Existing protected Windows cache directories can still emit local ACL warnings. | — | 2026-09-23 | Hosted CI + focused local automated gates | [`validation/roadmap.md`](validation/roadmap.md); [`coding/quality_gates.md`](coding/quality_gates.md); [`operations/commands.md`](operations/commands.md); [`ADS-T1-03/summary.md`](../QA/validation/2026-09-23/ADS-T1-03/summary.md); [`ADS-T1-02/summary.md`](../QA/validation/2026-09-23/ADS-T1-02/summary.md); [`ADS-T0-02/launcher-lifecycle.md`](../QA/validation/2026-09-22/ADS-T0-02/launcher-lifecycle.md); [hosted run 35910483168](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/runs/35910483168) | Revalidate the three-job workflow and affected profile suites after future changes. |
| `deployment.container` | `NOT_IMPLEMENTED` | Containerized deployment target. | [`runtime/deployment.md`](runtime/deployment.md) explicitly documents Windows local deployment as supported and no container target as implemented. | No container packaging or deployment contract exists in the current scope. | Product scope decision, not a runtime incident. | — | None | [`runtime/deployment.md`](runtime/deployment.md); [`architecture/system_overview.md`](architecture/system_overview.md) | No action unless a container deployment target is explicitly added to project scope. |

## Open issues

Severity describes impact or urgency; it is independent of component status.
The catalog includes actionable defects and validation blockers, not every old
finding mentioned in a report.

| ID | Affected Component | Severity | Description | Current Impact | Evidence / Reproduction | Suspected Cause | Blocker | Remediation Status | Required Revalidation | Related Docs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ISSUE-001` | `workflow.ml-training` | `HIGH` | No positive end-to-end training run is currently evidenced. | Dataset build, training, checkpoint, resume, and populated dashboard claims cannot be made. | Build the available fixture in the 2026-09-10 validation flow; it returned `Training data missing adsorbate_SMILE values.` | The available fixture lacks the required `adsorbate_SMILE` field; this is a data prerequisite, not a confirmed code defect. | Valid SMILES-bearing fixture, ML-enabled install, and suitable compute. | Open validation/data prerequisite; negative validation is working. | Run positive build, training, checkpoint, resume, and dashboard flows in an isolated ML-enabled environment. | [`adsmod-end-to-end-ui-system-validation-2026-09-10.md`](../QA/adsmod-end-to-end-ui-system-validation-2026-09-10.md); [`runtime/modes.md`](runtime/modes.md) |
| `ISSUE-002` | `data.public.nist` | `MEDIUM` | Some NIST records remain outside canonical coverage because their unit/metadata basis cannot be converted safely. | The validated acquisition skipped 14 records; local results are incomplete although counts are surfaced. | The 2026-09-10 live acquisition recorded 14 skipped records; the 2026-09-02 report records the same class of limitation. | Unsupported uptake units or insufficient source metadata, as documented by the mapper/report. | Reachable provider data and a supported conversion basis. | Open coverage limitation; skip-and-report behavior is implemented. | Exercise representative unsupported units and reachable-provider data after normalization/coverage changes. | [`architecture/public_data.md`](architecture/public_data.md); [`adsmod-end-to-end-ui-system-validation-2026-09-10.md`](../QA/adsmod-end-to-end-ui-system-validation-2026-09-10.md) |
| `ISSUE-004` | `data.public.cod-import` | `MEDIUM` | Successful COD import and linking have not been demonstrated. | Search/no-result behavior is known; positive structure ingestion and association are not yet trustworthy claims. | The 2026-09-10 run returned a bounded successful no-result response, not an imported record. | Unknown until a positive provider record or deterministic fixture is exercised. | Reachable positive COD record or provider stub. | Open validation gap; no failure is currently known. | Run bounded search, import, CIF persistence, normalized fields, and linking checks in API and browser paths. | [`architecture/public_data.md`](architecture/public_data.md); [`adsmod-end-to-end-ui-system-validation-2026-09-10.md`](../QA/adsmod-end-to-end-ui-system-validation-2026-09-10.md) |
| `ISSUE-005` | `ui.dashboards` | `LOW` | The top-level `/dashboards` route remains a placeholder and the populated training dashboard is unverified. | Users do not have a completed general dashboard view; training metrics cannot be confirmed without a positive run. | The 2026-09-10 report recommends deciding whether to replace or retain the dashboard placeholder. | Product scope is undecided; the populated training state is also constrained by `ISSUE-001`. | Product decision and valid training output for the populated state. | Open product-scope decision; not a runtime defect. | After scope is selected, validate the implemented dashboard at desktop/mobile sizes and with real training output. | [`operations/workflows.md`](operations/workflows.md); [`adsmod-end-to-end-ui-system-validation-2026-09-10.md`](../QA/adsmod-end-to-end-ui-system-validation-2026-09-10.md) |

| `ISSUE-007` | `quality.validation-gates` | `MEDIUM` | The hosted CI workflow briefly failed before backend validation because `setup-uv` selected a Python version without installing it on the runner. | No current impact after the explicit Python setup remediation; the current CI gate is green. | [Current run 35910483168](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/runs/35910483168) passed all three jobs; earlier runs preserve the `No interpreter found for Python 3.14.7` evidence. | `setup-uv`'s `python-version` input selected `UV_PYTHON` but did not supply the hosted interpreter; `actions/setup-python@v5` now does. | — | Resolved; current SHA passed the complete workflow. | Repeat the three-job gate after future workflow/runtime/dependency changes; watch the Node.js 20 and `ubuntu-latest` migration annotations. | [`ADS-T1-03/summary.md`](../QA/validation/2026-09-23/ADS-T1-03/summary.md); [`ADS-T1-02/summary.md`](../QA/validation/2026-09-23/ADS-T1-02/summary.md); [`validation/roadmap.md`](validation/roadmap.md) |
## Validation debt

Validation debt is a confidence gap, not a defect classification. It is kept
separate from the open-issue catalog so future sessions can choose the next
useful test without treating every untested path as broken.

| Component | Current Confidence | Missing Validation | Priority |
| --- | --- | --- | --- |
| `ADS-T0-01` | High / current hosted pass | No remaining Tier 0 gate debt. | — |
| `ADS-T0-02` | High / current live pass | No remaining Tier 0 launcher gate debt; Tier 4 positive ML training remains separate. | — |
| `ADS-T1-01` | High / current hosted pass | No remaining API/capability gate debt in this slice. | — |
| `ADS-T1-02` | High / current local and hosted pass | No remaining SQLite/Alembic startup or app-restart persistence gate debt in this slice. | — |
| `ADS-T1-03` | High / current local, browser, and hosted pass | No remaining Tier 1 shell navigation/recovery gate debt in this slice. | — |
| `ADS-T2-01` | High / current browser pass | No remaining CSV import lifecycle gate debt in this slice. | — |
| `ADS-T2-02` | High / current browser and unit pass | No remaining `.xls`/`.xlsx` import gate debt in this slice. | — |
| `ADS-T2-03` | High / current API and unit pass | No remaining exercised import-boundary gate debt in this slice. | — |
| `ADS-T2-04` | High / current live UI and fitting pass | No remaining fitting configuration gate debt in this slice. | — |
| `ADS-T2-05` | High / current live result and reload pass | No remaining fitting result/persistence gate debt in this slice. | — |
| `ADS-T2-06` | High / current live cancellation and recovery pass | No remaining fitting cancellation/recovery gate debt in this slice. | — |
| `ADS-T3-01` | Low / next actionable slice | Validate persisted Public Data browsing, filtering, pagination, and provenance in the browser. | Medium |
| `data.public.nist` | Medium / partial | Broader unit normalization and reachable-provider coverage; 14 historical records remain skipped. | Medium |
| `runtime.startup.windows` | High / current live pass | Revalidate only after future launcher, process, port, or startup changes. | — |
| `workflow.ml-training` | Low / blocked | Positive fixture, processing, training, checkpoint, resume, and populated dashboard on the ML-enabled runtime. | High |
| `data.public.cod-import` | Low | Positive COD result, CIF import, persistence, and linking with deterministic or live evidence. | Medium |
| `data.public.pubchem-enrichment` | Low | Positive enrichment plus secondary-endpoint failure handling against mocked and reachable-provider cases. | Medium |
| `quality.validation-gates` | High / current hosted pass | No current three-job CI gate debt; revalidate after future workflow, dependency, or generated-contract changes. | — |
| `ui.dashboards` | Low | Product-scope decision, completed top-level view, and populated training metrics. | Low |

## Resolved / historical findings

These entries preserve concise provenance only. They are not active issues and
must not be used to infer a current regression without a new failure.

| Finding | Resolution | Evidence |
| --- | --- | --- |
| `ISSUE-003`: the advertised `.xls` and `.xlsx` import path lacked real binary validation and base-runtime readers. | Added pinned `openpyxl` and `xlrd` dependencies and actual workbook fixtures; both formats passed preview, validation, save, inspection, reload, and deletion. | [`ADS-T2-02/summary.md`](../QA/validation/2026-09-23/ADS-T2-02/summary.md) |
| Saved dataset selection had no inspection surface. | Added the read-only import-inspection route and `DatasetInspectorComponent`; CSV selection, experiment switching, and persisted-row inspection passed. | [`adsmod-dataset-e2e-20260916.md`](../QA/adsmod-dataset-e2e-20260916.md) |
| Training wizard controls overlapped and supplied defaults rendered blank. | Added shared wizard layout styles and moved default patching to the initialized Angular lifecycle; live retest and regression test passed. | [`adsmod-end-to-end-ui-system-validation-2026-09-10.md`](../QA/adsmod-end-to-end-ui-system-validation-2026-09-10.md) |
| Fitting results were returned but not rendered, and restored dataset selection was not reflected in the control. | The result panel and selection synchronization were corrected and rechecked through live fitting/browser flows. | [`e2e-ui-system-validation-2026-09-02.md`](../QA/e2e-ui-system-validation-2026-09-02.md); [`e2e-ui-system-validation-2026-08-16.md`](../QA/e2e-ui-system-validation-2026-08-16.md) |
| One unsupported NIST measurement could abort a category fetch. | The mapper now skips only the unsupported record and reports skip counts. The resulting coverage limitation remains active as `ISSUE-002`. | [`e2e-ui-system-validation-2026-09-02.md`](../QA/e2e-ui-system-validation-2026-09-02.md); [`adsmod-end-to-end-ui-system-validation-2026-09-10.md`](../QA/adsmod-end-to-end-ui-system-validation-2026-09-10.md) |
| KaTeX depended on a CDN stylesheet. | The stylesheet was moved into the npm bundle; the post-fix browser probe reported no failed requests. | [`adsmod-end-to-end-ui-system-validation-2026-09-10.md`](../QA/adsmod-end-to-end-ui-system-validation-2026-09-10.md) |
| The repository carried a dual-backend architecture. | The current architecture is one FastAPI process with optional in-process ML registration and aligned packaging/contracts. | [`architecture/findings_and_remediation.md`](architecture/findings_and_remediation.md); [`architecture/v3_migration_status.md`](architecture/v3_migration_status.md) |
| The launcher could miss an active TCP listener during interactive occupied-port preflight. | Added an `IPGlobalProperties.GetActiveTcpListeners()` fallback that fails closed when the owner PID cannot be resolved; the current SHA additionally rechecks PID/name/start-time identity before termination and passed the full conflict matrix and live listener checks. | [`ADS-T0-02/launcher-lifecycle.md`](../QA/validation/2026-09-22/ADS-T0-02/launcher-lifecycle.md); [`../../start_on_windows.ps1`](../../start_on_windows.ps1) |
| Unknown Alembic revisions escaped startup validation as a raw `CommandError`. | Revision lookup failures are translated into `DatabaseMigrationError`; regression coverage confirms an unknown revision and a head-stamped incomplete schema both fail closed without schema inference. | [`ADS-T1-02/summary.md`](../QA/validation/2026-09-23/ADS-T1-02/summary.md); [`../../app/server/repositories/database/migrator.py`](../../app/server/repositories/database/migrator.py) |
| Fitting cancellation updated a run to `cancelled`, but SQLite rejected the state under `ck_fitting_runs_status`, returning HTTP 500. | Added Alembic revision `20260924_fitting_cancel` and aligned the canonical model constraint; official startup applied it, and a live 200,000-observation cancellation persisted as `cancelled` with successful follow-up recovery. | [`ADS-T2-06/summary.md`](../QA/validation/2026-09-24/ADS-T2-06/summary.md) |

## Revalidation map

Use the smallest relevant path, then expand when a change crosses boundaries:

| Change area | Revalidate |
| --- | --- |
| Launcher, ports, process cleanup, cache roots, or startup configuration | `runtime.startup.windows`, `quality.validation-gates`, readiness, browser opening, and final listener state. |
| Routes, service boundaries, API contracts, database schema, or migrations | `backend.core.api`, `persistence.database-migrations`, affected workflow E2E, and generated OpenAPI/schema artifacts. |
| Import parser, mapping, units, or inspection UI | `data.local-dataset.csv-lifecycle`, `data.local-dataset.excel-import`, persistence reload, and fitting selection. |
| NIST, PubChem, or COD adapters and normalization | Only the affected provider component, including explicit external-failure and provenance behavior. |
| ML dependencies, capability discovery, training data, models, or checkpoints | `runtime.installation-profiles`, `runtime.ml-capabilities`, and the full positive `workflow.ml-training` path when its blocker is available. |
| Angular routes, shared styles, responsive layout, or accessibility states | `ui.shell.navigation`, affected UI component tests, and the configured visual/browser viewport suite. |
