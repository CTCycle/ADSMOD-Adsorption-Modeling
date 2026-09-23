# ADSMOD Project Status Ledger

Last updated: 2026-09-23

This is the canonical current operational status catalog for ADSMOD. It is a
compact index of what is working, validated, partial, blocked, unvalidated, or
not implemented. Detailed architecture, debugging narratives, implementation
plans, and long validation logs remain in their dedicated documents.

The validated implementation baseline for `ADS-T1-01` is `develop` at SHA
`1affb39a2c4e475a8616004ee0566c30c5d189e7`. Hosted workflow run `35832020842`
passed all three jobs, including the clean Base-profile route assertions. The
official Windows launcher evidence for `ADS-T0-02` remains anchored to
`2eb3bc13823bedae1be1791fc980a78613d00ef1`; no launcher source changed since
that validation. Pre-existing untracked cache residue, protected paths, and
unrelated data were preserved and are not part of the evidence claim.

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
is now closed; Tier 1 is in progress.

| Tier | Slice IDs | Status | Current gate |
| --- | --- | --- | --- |
| Tier 0 — environment and startup | `ADS-T0-01`, `ADS-T0-02` | `PASS` / `PASS` | CI run `35832020842` passed all three jobs on `1affb39`; the official launcher lifecycle and conflict/race evidence remains at `2eb3bc1`, with no launcher source changes since. |
| Tier 1 — application foundations | `ADS-T1-01`–`ADS-T1-03` | `PARTIAL` | `ADS-T1-01` passed on `1affb39`; `ADS-T1-02` and `ADS-T1-03` remain untested. |
| Tier 2 — core product workflows | `ADS-T2-01`–`ADS-T2-06` | `UNTESTED` | Depends on Tier 1 and disposable runtime evidence. |
| Tier 3 — feature families/providers | `ADS-T3-01`–`ADS-T3-05` | `UNTESTED` | Follow the provider-specific positive/degraded evidence rules. |
| Tier 4 — ML workflows | `ADS-T4-01`–`ADS-T4-04` | `UNTESTED` | Requires an ML-enabled runtime and valid positive fixture. |
| Tier 5 — resilience and closure | `ADS-T5-01`–`ADS-T5-05` | `UNTESTED` | Final evidence, contract, and ledger reconciliation. |

### Tier 0 slice ledger

| Slice ID | Capability | Feature exists | Exercised | Status | Baseline revision | Issues | Evidence | Remaining gap | Evidence strength |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ADS-T0-01` | Current GitHub Actions quality gates | YES | YES | `PASS` | `1affb39a2c4e475a8616004ee0566c30c5d189e7` | None in the required gate; hosted annotations are platform maintenance warnings only | [`ADS-T1-01/summary.md`](../QA/validation/2026-09-23/ADS-T1-01/summary.md); [`ADS-T0-01/ci-workflow.md`](../QA/validation/2026-09-22/ADS-T0-01/ci-workflow.md); [hosted run 35832020842](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/runs/35832020842) | — | hosted CI + local exact gates |
| `ADS-T0-02` | Official Windows launcher lifecycle and static preview | YES | YES | `PASS` | `2eb3bc13823bedae1be1791fc980a78613d00ef1` | Chrome direct JSON navigation was blocked by `ERR_BLOCKED_BY_CLIENT`; rendered browser and direct local proxy checks passed | [`tier-0-summary.md`](../QA/validation/2026-09-22/tier-0-summary.md); [`ADS-T0-02/launcher-lifecycle.md`](../QA/validation/2026-09-22/ADS-T0-02/launcher-lifecycle.md) | — | official launcher + browser + HTTP + process/port checks |

### Tier 1 slice ledger

| Slice ID | Capability | Feature exists | Exercised | Status | Baseline revision | Issues | Evidence | Remaining gap | Evidence strength |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ADS-T1-01` | Core API and capability boundaries | YES | YES | `PASS` | `1affb39a2c4e475a8616004ee0566c30c5d189e7` | None observed | [`ADS-T1-01/summary.md`](../QA/validation/2026-09-23/ADS-T1-01/summary.md); [hosted run 35832020842](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/runs/35832020842) | `ADS-T1-02` and `ADS-T1-03` remain untested | local API smoke + hosted CI + frontend unit |

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
| `persistence.database-migrations` | `VALIDATED` | Alembic startup, canonical database schema, local persistence, and dataset record lifecycle. | The 2026-09-16 package-cutover validation checked isolated migrations and 142 backend/persistence/unit cases; the dataset run saved, reloaded, inspected, and deleted records in a fresh session. | Restricted sandbox ACLs caused an initial SQLite `readonly database` result; the same workflow passed in the normal writable user context. Protected pytest-cache directories also produce environment warnings. | Windows ACLs can block a restricted validation context; this is not an observed normal-user application failure. | 2026-09-16 | Integration + E2E + manual E2E | [`architecture/persistence_and_packages.md`](architecture/persistence_and_packages.md); [`runtime/startup.md`](runtime/startup.md); [`../../app/tests/persistence/test_schema_contract.py`](../../app/tests/persistence/test_schema_contract.py); [`adsmod-dataset-e2e-20260916.md`](../QA/adsmod-dataset-e2e-20260916.md) | Revalidate migration startup and disposable-database writes after schema or database-runtime changes. |
| `ui.shell.navigation` | `VALIDATED` | Landing redirect, primary navigation, route states, status messaging, help dialog, and responsive shell behavior. | The 2026-09-10 browser run exercised the main routes and seven responsive viewports; frontend unit, lint, build, and visual gates passed. | — | — | 2026-09-10 | Manual E2E + visual + unit | [`ui/experience.md`](ui/experience.md); [`ui/standards.md`](ui/standards.md); [`adsmod-end-to-end-ui-system-validation-2026-09-10.md`](../QA/adsmod-end-to-end-ui-system-validation-2026-09-10.md) | Revalidate the affected routes and responsive states after navigation, shell, or global-style changes. |
| `data.local-dataset.csv-lifecycle` | `VALIDATED` | CSV preview, column detection, mapping, validation, save, persistence, inspection, experiment switching, fresh-session reload, and deletion. | [`adsmod-dataset-e2e-20260916.md`](../QA/adsmod-dataset-e2e-20260916.md) records PASS for the complete exercised CSV lifecycle, including clean browser-console checks. | The evidence covers CSV fixtures; binary Excel behavior is tracked separately. | — | 2026-09-16 | Manual E2E | [`operations/workflows.md`](operations/workflows.md); [`../../app/tests/e2e/test_datasets_api.py`](../../app/tests/e2e/test_datasets_api.py); [`adsmod-dataset-e2e-20260916.md`](../QA/adsmod-dataset-e2e-20260916.md) | Re-run the relevant import and persistence flow after parser, schema, or inspector changes. |
| `data.local-dataset.excel-import` | `UNVALIDATED` | Real `.xls` and `.xlsx` binary upload, parsing, mapping, validation, save, and inspection. | Runtime configuration and the file picker advertise `.csv`, `.xls`, and `.xlsx`; the 2026-09-16 report explicitly says no real Excel binary was executed. | No product failure is established because the binary path was not exercised. | A real fixture and an available parsing engine/browser file-chooser path are required for meaningful validation. | — | None | [`runtime/configuration.md`](runtime/configuration.md); [`../../app/tests/e2e/test_datasets_api.py`](../../app/tests/e2e/test_datasets_api.py); [`adsmod-dataset-e2e-20260916.md`](../QA/adsmod-dataset-e2e-20260916.md) | Execute binary Excel preview through inspection on a disposable database, then add focused evidence. |
| `workflow.fitting` | `VALIDATED` | Dataset/experiment selection, asynchronous fitting, polling, result metrics, per-model outcomes, and reset. | The 2026-09-10 browser run completed a live fit with 8 of 9 models and rendered the result summary; the failed model row remained explicit rather than fabricating metrics. | Individual models can fail for a given dataset; the current UI exposes that outcome. No evidence shows a global fitting-workflow defect. | — | 2026-09-10 | Integration + manual E2E | [`operations/workflows.md`](operations/workflows.md); [`../../app/tests/e2e/test_fitting_api.py`](../../app/tests/e2e/test_fitting_api.py); [`adsmod-end-to-end-ui-system-validation-2026-09-10.md`](../QA/adsmod-end-to-end-ui-system-validation-2026-09-10.md) | Revalidate selection, job completion, result rendering, and reset after fitting or polling changes. |
| `data.public.nist` | `PARTIAL` | NIST health, index/fetch jobs, canonical unit mapping, persistence, and truthful UI counts. | The 2026-09-02 and 2026-09-10 runs exercised live status and fetch paths; the latter recorded requested/fetched/local/skipped counts. | Provider health can be false or unavailable; 14 records in the validated acquisition were skipped because their units or metadata could not be mapped to canonical units. | Full coverage depends on external NIST availability and source measurements that can be converted safely. | 2026-09-10 | Integration + manual E2E | [`architecture/public_data.md`](architecture/public_data.md); [`../../app/tests/e2e/test_nist_api.py`](../../app/tests/e2e/test_nist_api.py); [`adsmod-end-to-end-ui-system-validation-2026-09-10.md`](../QA/adsmod-end-to-end-ui-system-validation-2026-09-10.md) | Preserve explicit skip counts; add or validate broader unit-normalization/coverage handling and rerun against a reachable provider. |
| `data.public.pubchem-enrichment` | `UNVALIDATED` | Positive PubChem identity resolution, enrichment, properties, and structure retrieval from local records. | Provider architecture and unit-test inventory exist, but no recent positive live enrichment evidence was found in the inspected QA reports. | Secondary PubChem endpoints may be unavailable even when a primary compound resolves; no current application defect is established. | Current positive provider evidence is missing and the remote service is external. | — | Unit only | [`architecture/public_data.md`](architecture/public_data.md); [`../../app/tests/unit/test_public_data.py`](../../app/tests/unit/test_public_data.py) | Add deterministic mocked coverage for normalization/failure branches and a bounded live enrichment run when provider access is available. |
| `data.public.cod-import` | `UNVALIDATED` | Successful COD search result, CIF import, normalized persistence, and material/structure linking. | The 2026-09-10 run validated a bounded no-result search (`items: []`) but did not capture a successful import/link. | Successful provider-result handling and linking remain unproven; this is a validation gap, not an observed failure. | A deterministic COD fixture/provider stub or reachable record is needed. | 2026-09-10 (no-result path only) | Integration + manual E2E (negative path) | [`architecture/public_data.md`](architecture/public_data.md); [`adsmod-end-to-end-ui-system-validation-2026-09-10.md`](../QA/adsmod-end-to-end-ui-system-validation-2026-09-10.md) | Validate count/search/import/persistence/linking with a bounded positive fixture and browser evidence. |
| `runtime.ml-capabilities` | `VALIDATED` | Optional ML import boundary, capability response, route registration, and base-profile unavailability behavior. | The 2026-09-23 hosted profiles verified Base capabilities false, absent training routes, and ML-profile capability/routes; frontend capability retry/refresh and shell gating unit coverage passed. | This status does not claim a successful training run; that is tracked by `workflow.ml-training`. | — | 2026-09-23 | Hosted profile CI + frontend unit | [`ADS-T1-01/summary.md`](../QA/validation/2026-09-23/ADS-T1-01/summary.md); [`runtime/modes.md`](runtime/modes.md); [`../../app/tests/backend/test_ml_routes.py`](../../app/tests/backend/test_ml_routes.py); [`../../app/tests/unit/test_ml_boundary.py`](../../app/tests/unit/test_ml_boundary.py); [`adsmod-end-to-end-audit-2026-09-10.md`](../QA/adsmod-end-to-end-audit-2026-09-10.md) | Revalidate both dependency profiles whenever optional imports, capability contracts, or route registration changes. |
| `workflow.ml-training` | `BLOCKED` | Positive processed-dataset creation, training, checkpoint creation, resume, and populated dashboard metrics. | The negative build path was live-validated and correctly returned `Training data missing adsorbate_SMILE values.`; automated ML route tests passed, but no positive live run is recorded. | Positive training, checkpoint, resume, and populated-dashboard behavior remain unproven; this is not evidence that the implementation is broken. | A valid `adsorbate_SMILE`-bearing fixture plus an ML-enabled runtime and suitable local hardware/compute are required. | 2026-09-10 (negative validation only) | Integration + manual E2E (negative path) | [`runtime/modes.md`](runtime/modes.md); [`operations/workflows.md`](operations/workflows.md); [`../../app/tests/e2e/test_training_api.py`](../../app/tests/e2e/test_training_api.py); [`adsmod-end-to-end-ui-system-validation-2026-09-10.md`](../QA/adsmod-end-to-end-ui-system-validation-2026-09-10.md) | Add an isolated valid fixture and run dataset build, training, checkpoint, resume, and dashboard-positive checks. |
| `ui.dashboards` | `PARTIAL` | Top-level dashboards route and populated training-dashboard experience. | The 2026-09-10 run exercised dashboard/training states and documented the current route behavior. | The top-level `/dashboards` view remains a placeholder; populated training metrics were not verified because positive training is blocked. | Product scope is needed for the placeholder; populated metrics share the ML-training blocker. | 2026-09-10 | Manual E2E (placeholder/empty states) | [`operations/workflows.md`](operations/workflows.md); [`../../app/client/src/app/features/dashboards/dashboards-page.component.ts`](../../app/client/src/app/features/dashboards/dashboards-page.component.ts); [`adsmod-end-to-end-ui-system-validation-2026-09-10.md`](../QA/adsmod-end-to-end-ui-system-validation-2026-09-10.md) | Decide the top-level dashboard scope, then implement and validate it; revalidate populated training metrics after `ISSUE-001`. |
| `quality.validation-gates` | `VALIDATED` | Backend/frontend tests, lint, build, generated contracts, visual checks, profile-correct runner behavior, and repository-local cache layout. | Hosted run `35832020842` on SHA `1affb39` passed all three jobs, including the expanded Base route boundary, ML backend checks, and frontend browser layout validation; see [`ADS-T1-01/summary.md`](../QA/validation/2026-09-23/ADS-T1-01/summary.md). | Protected pre-existing Windows cache directories can produce local ACL warnings; the hosted gate passes. | — | 2026-09-23 | Hosted CI + local automated gates | [`validation/roadmap.md`](validation/roadmap.md); [`coding/quality_gates.md`](coding/quality_gates.md); [`operations/commands.md`](operations/commands.md); [`ADS-T1-01/summary.md`](../QA/validation/2026-09-23/ADS-T1-01/summary.md); [`ADS-T0-02/launcher-lifecycle.md`](../QA/validation/2026-09-22/ADS-T0-02/launcher-lifecycle.md) | Revalidate the three-job workflow and affected profile suites after future changes. |
| `deployment.container` | `NOT_IMPLEMENTED` | Containerized deployment target. | [`runtime/deployment.md`](runtime/deployment.md) explicitly documents Windows local deployment as supported and no container target as implemented. | No container packaging or deployment contract exists in the current scope. | Product scope decision, not a runtime incident. | — | None | [`runtime/deployment.md`](runtime/deployment.md); [`architecture/system_overview.md`](architecture/system_overview.md) | No action unless a container deployment target is explicitly added to project scope. |

## Open issues

Severity describes impact or urgency; it is independent of component status.
The catalog includes actionable defects and validation blockers, not every old
finding mentioned in a report.

| ID | Affected Component | Severity | Description | Current Impact | Evidence / Reproduction | Suspected Cause | Blocker | Remediation Status | Required Revalidation | Related Docs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ISSUE-001` | `workflow.ml-training` | `HIGH` | No positive end-to-end training run is currently evidenced. | Dataset build, training, checkpoint, resume, and populated dashboard claims cannot be made. | Build the available fixture in the 2026-09-10 validation flow; it returned `Training data missing adsorbate_SMILE values.` | The available fixture lacks the required `adsorbate_SMILE` field; this is a data prerequisite, not a confirmed code defect. | Valid SMILES-bearing fixture, ML-enabled install, and suitable compute. | Open validation/data prerequisite; negative validation is working. | Run positive build, training, checkpoint, resume, and dashboard flows in an isolated ML-enabled environment. | [`adsmod-end-to-end-ui-system-validation-2026-09-10.md`](../QA/adsmod-end-to-end-ui-system-validation-2026-09-10.md); [`runtime/modes.md`](runtime/modes.md) |
| `ISSUE-002` | `data.public.nist` | `MEDIUM` | Some NIST records remain outside canonical coverage because their unit/metadata basis cannot be converted safely. | The validated acquisition skipped 14 records; local results are incomplete although counts are surfaced. | The 2026-09-10 live acquisition recorded 14 skipped records; the 2026-09-02 report records the same class of limitation. | Unsupported uptake units or insufficient source metadata, as documented by the mapper/report. | Reachable provider data and a supported conversion basis. | Open coverage limitation; skip-and-report behavior is implemented. | Exercise representative unsupported units and reachable-provider data after normalization/coverage changes. | [`architecture/public_data.md`](architecture/public_data.md); [`adsmod-end-to-end-ui-system-validation-2026-09-10.md`](../QA/adsmod-end-to-end-ui-system-validation-2026-09-10.md) |
| `ISSUE-003` | `data.local-dataset.excel-import` | `LOW` | Accepted `.xls` and `.xlsx` paths have no real binary validation evidence. | CSV users have validated coverage; Excel users have no equivalent confidence claim. | The 2026-09-16 dataset report states that no Excel binary fixture or parsing-engine run was available. | Validation environment lacked the fixture/parser path; no product cause is established. | Binary fixture, parser dependency, and usable file chooser. | Open validation gap; not classified as broken. | Execute preview through save, inspection, and reload for both relevant Excel formats or document unsupported format behavior. | [`runtime/configuration.md`](runtime/configuration.md); [`adsmod-dataset-e2e-20260916.md`](../QA/adsmod-dataset-e2e-20260916.md) |
| `ISSUE-004` | `data.public.cod-import` | `MEDIUM` | Successful COD import and linking have not been demonstrated. | Search/no-result behavior is known; positive structure ingestion and association are not yet trustworthy claims. | The 2026-09-10 run returned a bounded successful no-result response, not an imported record. | Unknown until a positive provider record or deterministic fixture is exercised. | Reachable positive COD record or provider stub. | Open validation gap; no failure is currently known. | Run bounded search, import, CIF persistence, normalized fields, and linking checks in API and browser paths. | [`architecture/public_data.md`](architecture/public_data.md); [`adsmod-end-to-end-ui-system-validation-2026-09-10.md`](../QA/adsmod-end-to-end-ui-system-validation-2026-09-10.md) |
| `ISSUE-005` | `ui.dashboards` | `LOW` | The top-level `/dashboards` route remains a placeholder and the populated training dashboard is unverified. | Users do not have a completed general dashboard view; training metrics cannot be confirmed without a positive run. | The 2026-09-10 report recommends deciding whether to replace or retain the dashboard placeholder. | Product scope is undecided; the populated training state is also constrained by `ISSUE-001`. | Product decision and valid training output for the populated state. | Open product-scope decision; not a runtime defect. | After scope is selected, validate the implemented dashboard at desktop/mobile sizes and with real training output. | [`operations/workflows.md`](operations/workflows.md); [`adsmod-end-to-end-ui-system-validation-2026-09-10.md`](../QA/adsmod-end-to-end-ui-system-validation-2026-09-10.md) |

| `ISSUE-007` | `quality.validation-gates` | `MEDIUM` | The hosted CI workflow briefly failed before backend validation because `setup-uv` selected a Python version without installing it on the runner. | No current impact after the explicit Python setup remediation; the current CI gate is green. | [Current run 35832020842](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/runs/35832020842) passed all three jobs; earlier runs preserve the `No interpreter found for Python 3.14.7` evidence. | `setup-uv`'s `python-version` input selected `UV_PYTHON` but did not supply the hosted interpreter; `actions/setup-python@v5` now does. | — | Resolved; current SHA passed the complete workflow. | Repeat the three-job gate after future workflow/runtime/dependency changes; watch the Node.js 20 and `ubuntu-latest` migration annotations. | [`ADS-T1-01/summary.md`](../QA/validation/2026-09-23/ADS-T1-01/summary.md); [`validation/roadmap.md`](validation/roadmap.md) |
## Validation debt

Validation debt is a confidence gap, not a defect classification. It is kept
separate from the open-issue catalog so future sessions can choose the next
useful test without treating every untested path as broken.

| Component | Current Confidence | Missing Validation | Priority |
| --- | --- | --- | --- |
| `ADS-T0-01` | High / current hosted pass | No remaining Tier 0 gate debt. | — |
| `ADS-T0-02` | High / current live pass | No remaining Tier 0 launcher gate debt; Tier 4 positive ML training remains separate. | — |
| `ADS-T1-01` | High / current hosted pass | No remaining API/capability gate debt in this slice. | — |
| `runtime.startup.windows` | High / current live pass | Revalidate only after future launcher, process, port, or startup changes. | — |
| `workflow.ml-training` | Low / blocked | Positive fixture, processing, training, checkpoint, resume, and populated dashboard on the ML-enabled runtime. | High |
| `data.local-dataset.excel-import` | Low | Real `.xls` and `.xlsx` binary preview-to-inspection flows. | Medium |
| `data.public.cod-import` | Low | Positive COD result, CIF import, persistence, and linking with deterministic or live evidence. | Medium |
| `data.public.pubchem-enrichment` | Low | Positive enrichment plus secondary-endpoint failure handling against mocked and reachable-provider cases. | Medium |
| `quality.validation-gates` | High / current hosted pass | No current three-job CI gate debt; revalidate after future workflow, dependency, or generated-contract changes. | — |
| `ui.dashboards` | Low | Product-scope decision, completed top-level view, and populated training metrics. | Low |

## Resolved / historical findings

These entries preserve concise provenance only. They are not active issues and
must not be used to infer a current regression without a new failure.

| Finding | Resolution | Evidence |
| --- | --- | --- |
| Saved dataset selection had no inspection surface. | Added the read-only import-inspection route and `DatasetInspectorComponent`; CSV selection, experiment switching, and persisted-row inspection passed. | [`adsmod-dataset-e2e-20260916.md`](../QA/adsmod-dataset-e2e-20260916.md) |
| Training wizard controls overlapped and supplied defaults rendered blank. | Added shared wizard layout styles and moved default patching to the initialized Angular lifecycle; live retest and regression test passed. | [`adsmod-end-to-end-ui-system-validation-2026-09-10.md`](../QA/adsmod-end-to-end-ui-system-validation-2026-09-10.md) |
| Fitting results were returned but not rendered, and restored dataset selection was not reflected in the control. | The result panel and selection synchronization were corrected and rechecked through live fitting/browser flows. | [`e2e-ui-system-validation-2026-09-02.md`](../QA/e2e-ui-system-validation-2026-09-02.md); [`e2e-ui-system-validation-2026-08-16.md`](../QA/e2e-ui-system-validation-2026-08-16.md) |
| One unsupported NIST measurement could abort a category fetch. | The mapper now skips only the unsupported record and reports skip counts. The resulting coverage limitation remains active as `ISSUE-002`. | [`e2e-ui-system-validation-2026-09-02.md`](../QA/e2e-ui-system-validation-2026-09-02.md); [`adsmod-end-to-end-ui-system-validation-2026-09-10.md`](../QA/adsmod-end-to-end-ui-system-validation-2026-09-10.md) |
| KaTeX depended on a CDN stylesheet. | The stylesheet was moved into the npm bundle; the post-fix browser probe reported no failed requests. | [`adsmod-end-to-end-ui-system-validation-2026-09-10.md`](../QA/adsmod-end-to-end-ui-system-validation-2026-09-10.md) |
| The repository carried a dual-backend architecture. | The current architecture is one FastAPI process with optional in-process ML registration and aligned packaging/contracts. | [`architecture/findings_and_remediation.md`](architecture/findings_and_remediation.md); [`architecture/v3_migration_status.md`](architecture/v3_migration_status.md) |
| The launcher could miss an active TCP listener during interactive occupied-port preflight. | Added an `IPGlobalProperties.GetActiveTcpListeners()` fallback that fails closed when the owner PID cannot be resolved; the current SHA additionally rechecks PID/name/start-time identity before termination and passed the full conflict matrix and live listener checks. | [`ADS-T0-02/launcher-lifecycle.md`](../QA/validation/2026-09-22/ADS-T0-02/launcher-lifecycle.md); [`../../start_on_windows.ps1`](../../start_on_windows.ps1) |

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
