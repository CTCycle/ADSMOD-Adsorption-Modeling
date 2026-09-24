# ADSMOD validation campaign

Last updated: 2026-09-24

## Purpose and baseline

This document is the durable, repository-local digest of the comprehensive
validation roadmap. It separates source evidence, automated checks, live API
checks, browser workflows, provider access, and hardware-dependent ML runs so
that future analysis can resume at a stable slice instead of treating the
presence of tests as proof of current behavior.

The official Windows launcher evidence remains anchored to committed `develop`
SHA `2eb3bc13823bedae1be1791fc980a78613d00ef1`; no launcher source changed on
the path to the current application gates. `ADS-T1-01` passed on SHA
`1affb39a2c4e475a8616004ee0566c30c5d189e7`. `ADS-T1-02` passed on tested code
SHA `ed80c0975e583cd9842338fca5f59157c4071f97`; hosted run `35852874401` on
that SHA passed all three CI jobs. `ADS-T1-03` passed on tested code SHA
`6154532dd6628bb70e98fae6fa427e2d9108cd5f`; hosted run `35876297668` passed all
three CI jobs. Tier 0 was the first gate because the remote
CI workflow had previously failed before creating any jobs and the Windows
launcher had changed after the last surviving live report.
The three-job workflow most recently passed on SHA `270e94df63969feb6fbe81b081dc7d3e85744c2d`
in hosted run `35910483168`, including Base/ML dependency installation,
frontend checks, and browser layout validation.

The canonical current status is [`../project_status_ledger.md`](../project_status_ledger.md).
Execution evidence is under [`../../QA/validation/2026-09-22/`](../../QA/validation/2026-09-22/)
for Tier 0 and [`../../QA/validation/2026-09-23/ADS-T1-01/`](../../QA/validation/2026-09-23/ADS-T1-01/)
for `ADS-T1-01`, and [`../../QA/validation/2026-09-23/ADS-T1-02/`](../../QA/validation/2026-09-23/ADS-T1-02/)
for `ADS-T1-02`, and [`../../QA/validation/2026-09-23/ADS-T1-03/`](../../QA/validation/2026-09-23/ADS-T1-03/)
for `ADS-T1-03`.

## Evidence contract

Use these distinctions in every slice:

- `Feature exists` records source/configuration presence only.
- `Exercised` records whether the named scenarios actually ran: `YES`, `NO`,
  or `PARTIAL`.
- `Status` is `PASS`, `PARTIAL`, `FAIL`, `BLOCKED`, `UNTESTED`, or `UNKNOWN`.
- Historical evidence never upgrades a changed current revision by itself.
- A browser claim requires the visible user-facing workflow; API calls, DOM
  inspection, tests, and logs are supporting evidence rather than substitutes.
- A `BLOCKED` result must name the external dependency or environment cause;
  it must not hide an observed product failure.

Each slice should record its baseline SHA, exact scenarios, environment,
issues, fixes, adjacent regression, evidence paths, remaining gaps, timestamp,
validator, and evidence strength (`live UI`, `live API`, `automated`,
`source-only`, or `historical`).

## Ordered campaign

| Tier | Stable slices | Focus | Gate to leave the tier |
| --- | --- | --- | --- |
| Tier 0 | `ADS-T0-01`–`ADS-T0-02` | CI executability, Windows runtime, readiness, browser load, ownership-safe shutdown | All three CI jobs start and finish; launcher clean lifecycle and occupied-port protection pass |
| Tier 1 | `ADS-T1-01`–`ADS-T1-03` | Core API/capability boundaries, SQLite/Alembic startup, shell navigation and recovery | Base runtime, persistence, and shell navigation/recovery are current-revision validated |
| Tier 2 | `ADS-T2-01`–`ADS-T2-06` | CSV/Excel input, invalid boundaries, fitting configuration, execution, persistence, cancellation | Core product workflows pass with disposable data |
| Tier 3 | `ADS-T3-01`–`ADS-T3-05` | Local public data, NIST, PubChem, and COD positive/degraded paths | Provider claims have explicit positive or externally blocked evidence |
| Tier 4 | `ADS-T4-01`–`ADS-T4-04` | ML profile, valid training data, real training, checkpoints, resume, dashboard | Positive ML lifecycle is demonstrated on the ML profile |
| Tier 5 | `ADS-T5-01`–`ADS-T5-05` | Recovery, repetition/restart integrity, responsive/accessibility, performance, documentation/contracts | Evidence and ledger reconcile against the final tested SHA |

The dependency order is:

`T0-01 → T0-02 → T1-01 → T1-02 → T1-03 → T2-01 → T2-02 → T2-03 → T2-04 → T2-05 → T2-06 → T3-01 → T3-02 → T3-03 → T3-04/T3-05 → T4-01 → T4-02 → T4-03 → T4-04 → T5-01 → T5-02 → T5-03 → T5-04 → T5-05`

Do not advance when a slice exposes a reproducible defect. Stop, isolate the
smallest code path, apply the smallest fix, rerun the failed scenario and its
adjacent regression, then update the ledger.

## Slice catalog

### Tier 0 — environment and startup

| Slice | Scope | Current checkpoint |
| --- | --- | --- |
| `ADS-T0-01` | Diagnose the remote workflow, validate the YAML/configuration, repair it surgically, then confirm `base-backend`, `ml-backend`, and `frontend` execute their repository-defined commands. | Run `35852874401` on SHA `ed80c09` passed all three jobs; the Base profile verifies system configuration, fitting availability, and absent training routes. |
| `ADS-T0-02` | Use the official Windows launcher for dependency reuse, backend/frontend readiness, browser load, occupied-port refusal, owned-process stop, and clean relaunch. | The official lifecycle passed on SHA `2eb3bc1`; no launcher source changed through `ed80c09`. The detailed conflict/race, invalidation, ML-profile repair, readiness cleanup, and final-port evidence remains in the 2026-09-22 report. |

### Tier 1 — application foundations

| Slice | Scope | Current checkpoint |
| --- | --- | --- |
| `ADS-T1-01` | Health, system capabilities/configuration, core route availability, base-profile absence of ML routes, capability refresh/retry, and fitting configuration. | `PASS` on SHA `1affb39`; hosted run `35832020842` and focused local API/frontend tests. |
| `ADS-T1-02` | Missing/empty/current/invalid SQLite startup states, Alembic head and lock behavior, and minimal record persistence across restart. | `PASS` on tested code SHA `ed80c09`; 22 focused local tests passed, including fail-closed invalid states and dataset persistence across two application lifespans. Hosted run `35852874401` passed all three CI jobs. |
| `ADS-T1-03` | Implemented top-level routes, redirects, unknown-route recovery, Help focus behavior, backend Offline/Online recovery, and truthful unavailable Docs/Settings controls. | `PASS` on tested code SHA `6154532`; 70 frontend unit tests, lint, build, official-launcher browser/HTTP evidence, and hosted run `35876297668` passed. See [`ADS-T1-03/summary.md`](../../QA/validation/2026-09-23/ADS-T1-03/summary.md). |

### Tier 2 — core product workflows

| Slice | Scope | Current checkpoint |
| --- | --- | --- |
| `ADS-T2-01` | Complete browser CSV import through preview, mapping, validation, save, persisted inspection, experiment switching, reload, and deletion. | PASS on `c876a06`; see [`ADS-T2-01`](../../QA/validation/2026-09-23/ADS-T2-01/summary.md). |
| `ADS-T2-02` | Real `.xlsx` and `.xls` files through the canonical browser/API path. | PASS on `c876a06`; see [`ADS-T2-02`](../../QA/validation/2026-09-23/ADS-T2-02/summary.md). |
| `ADS-T2-03` | Invalid uploads, missing columns, malformed values, mismatched arrays, and other input-boundary failures. | PASS on `c876a06`; see [`ADS-T2-03`](../../QA/validation/2026-09-23/ADS-T2-03/summary.md). |
| `ADS-T2-04` | Dataset/experiment selection, fitting configuration, all nine model cards, and parameter forms. | PASS in the 2026-09-24 live fitting campaign; see [`ADS-T2-04`](../../QA/validation/2026-09-24/ADS-T2-04/summary.md). |
| `ADS-T2-05` | Positive asynchronous fitting, metrics/result rendering, persistence, and reload. | PASS after fixing result restoration on page reload; see [`ADS-T2-05`](../../QA/validation/2026-09-24/ADS-T2-05/summary.md). |
| `ADS-T2-06` | Fitting cancellation, duplicate-job prevention, recovery, and clean follow-up execution. | PASS after adding `20260924_fitting_cancel`; see [`ADS-T2-06`](../../QA/validation/2026-09-24/ADS-T2-06/summary.md). |

### Tier 3 — feature families and providers

| Slice | Scope | Current checkpoint |
| --- | --- | --- |
| `ADS-T3-01` | Locally persisted Public Data browsing, filtering, pagination, and provenance without requiring new retrieval. | PASS on `396e2e0`; isolated-database browser evidence covers lists, filters, pagination, detail provenance, and empty structures state. |
| `ADS-T3-02` | NIST status, index, fetch, and job lifecycle without chemical enrichment. | PARTIAL on `396e2e0`; pings/index and experiments fetch passed, guest/host fetch remains unrun after browser access was denied at the usage limit. |
| `ADS-T3-03` | NIST guest/host enrichment, unsupported-unit handling, skip counts, and normalization. | PARTIAL on `396e2e0`; focused tests and the live 14/40 unsupported-unit skip counter passed; guest/host enrichment remains unvalidated. |
| `ADS-T3-04` | Positive PubChem resolution and normalized persistence. | UNTESTED in this campaign; positive enrichment and persistence still need a bounded run. |
| `ADS-T3-05` | Positive COD search, CIF import, normalized persistence, linking, and re-import; no 3D viewer claim. | UNTESTED in this campaign; historical no-result search is not positive import/link evidence. |

### Tier 4 — ML and integration-heavy workflows

| Slice | Scope |
| --- | --- |
| `ADS-T4-01` | ML installation profile, one-backend topology, capability detection, and base-profile gating. |
| `ADS-T4-02` | Valid processed training dataset and immutable snapshot creation. |
| `ADS-T4-03` | Short real training run, status/metrics, and cancellation. |
| `ADS-T4-04` | Checkpoint creation, compatibility, resume, deletion, and populated dashboard. |

### Tier 5 — resilience and closure

| Slice | Scope |
| --- | --- |
| `ADS-T5-01` | Temporary network/provider/backend failures and truthful recovery without changing product behavior. |
| `ADS-T5-02` | Repeated operations, backend restart, stale state, duplicate records, and process/job cleanup. |
| `ADS-T5-03` | Keyboard operation, focus management, responsive workflows from desktop through the configured narrow viewports, and overflow. |
| `ADS-T5-04` | Bounded public-data queries, imports, polling, and other performance-sensitive boundaries with measured fixtures. |
| `ADS-T5-05` | Documentation, generated OpenAPI/configuration contracts, stale references, QA links, and final ledger reconciliation. |

## Evidence layout and regression checkpoints

Use stable slice directories below `assets/QA/validation/<date>/`, for example:

```text
assets/QA/validation/2026-09-22/
  ADS-T0-01/
  ADS-T0-02/
```

UI slices should retain screenshots or an equivalent durable rendered-state
record, route and viewport, console errors, relevant network status, and any
failure trace. Backend/persistence slices should retain commands, exit status,
logs, API results, database state before/after, and migration/job identifiers.
Provider records must include provider, timestamp, query/identifier, result
classification, and whether the boundary was local, provider-side,
rate-limited, or network-related; never store secrets.

Run the full application checkpoint after Tier 0, Tier 2, Tier 4, and for the
final release candidate. The entire application does not need to be rerun
after every surgical fix; the defined adjacent regression is the minimum.

## Current stopping point

Tier 0 remains closed. `ADS-T0-01` passed in hosted run `35910483168` on SHA
`270e94d`; all three jobs completed, including the Base/ML profile checks,
generated contracts, frontend checks, and browser layout validation.
`ADS-T0-02` remains `PASS` on its official lifecycle evidence at `2eb3bc1`;
launcher source was unchanged through SHA `6154532`.

`ADS-T1-01` is `PASS` on SHA `1affb39`. The Base profile reported core
capabilities, system configuration, fitting models, readiness, and Public
Data availability while keeping ML unavailable and both training routes
absent. The hosted ML profile and frontend capability retry/refresh tests also
passed. See [`ADS-T1-01/summary.md`](../../QA/validation/2026-09-23/ADS-T1-01/summary.md).
`ADS-T1-02` also passed on code SHA `ed80c09`, covering SQLite startup state
handling, Alembic head/lock behavior, and one dataset persisted across app
shutdown and restart. See
[`ADS-T1-02/summary.md`](../../QA/validation/2026-09-23/ADS-T1-02/summary.md).
`ADS-T1-03` passed on code SHA `6154532`, covering shell routes and recovery,
Help focus/closing, unavailable controls, and backend status recovery. Local
frontend unit/lint/build gates and hosted run `35876297668` passed. See
[`ADS-T1-03/summary.md`](../../QA/validation/2026-09-23/ADS-T1-03/summary.md).
Tier 1 is closed. Tier 2 slices `ADS-T2-01`, `ADS-T2-02`, and `ADS-T2-03`
passed on implementation SHA `c876a063d3fcef4de1ea074aaebba861da4b2c4c`.
Evidence covers CSV import/persistence/deletion, real `.xls` and `.xlsx`
browser imports, and invalid-input boundaries. See the three linked
[`ADS-T2-01`](../../QA/validation/2026-09-23/ADS-T2-01/summary.md),
[`ADS-T2-02`](../../QA/validation/2026-09-23/ADS-T2-02/summary.md), and
[`ADS-T2-03`](../../QA/validation/2026-09-23/ADS-T2-03/summary.md) summaries.
Tier 2 is closed. On the 2026-09-24 official-launcher campaign, `ADS-T2-04`
validated the fitting controls and all nine model cards, `ADS-T2-05` completed
a fit and verified that rendered metrics return after a page reload, and
`ADS-T2-06` exercised cancellation, duplicate rejection, and follow-up
recovery. The campaign found and fixed a missing `cancelled` SQLite status and
the result-reload gap. See the linked slice summaries and the canonical
[`project status ledger`](../project_status_ledger.md).

On baseline SHA `396e2e092d810ef5182713a563f1de69eb0b5eca`, `ADS-T3-01` passed
for local persisted browsing. `ADS-T3-02` and `ADS-T3-03` are `PARTIAL`: NIST
ping/index and an experiments fetch completed, and the live job reported 14
records without canonical units among 40 requested. Guest/host fetch and
enrichment were not run because browser auto-review denied further access after
the usage limit was reached. See the current
[`ADS-T3-01`](../../QA/validation/2026-09-24/ADS-T3-01/summary.md),
[`ADS-T3-02`](../../QA/validation/2026-09-24/ADS-T3-02/summary.md), and
[`ADS-T3-03`](../../QA/validation/2026-09-24/ADS-T3-03/summary.md) evidence.

The next actionable work is to complete guest/host NIST fetch and enrichment
through the rendered workflow on an isolated database, then confirm persisted
records and counts. `ISSUE-002` remains open until the 14 skipped measurement
bases can be converted safely or a coverage policy explicitly handles them.
Only after the NIST slices are complete should the positive PubChem (`ADS-T3-04`)
and COD (`ADS-T3-05`, `ISSUE-004`) paths be selected.

The ML training lifecycle remains `BLOCKED` (`ISSUE-001`) for the current
baseline: a valid-SMILES fixture and historical CPU certification exist, but
the current Base virtual environment has neither Torch nor RDKit, so the
positive lifecycle has not been rerun against this revision. The top-level
dashboards remain `PARTIAL` (`ISSUE-005`) pending a product-scope decision and
positive current training output. These ML/dashboard gates remain separate
follow-up work. See the [current ML blocker recheck](../../QA/validation/2026-09-24/ML-blocker-recheck.md).
