# ADSMOD validation campaign

Last updated: 2026-09-30

## Purpose

This document defines the stable validation campaign for the current v3
application. It owns slice IDs, scope, evidence rules, and dependency order.
The [project status ledger](../project_status_ledger.md) owns current status,
validated guarantees, limitations, and evidence anchors. Do not copy execution
narratives or historical run logs into this roadmap.

The campaign describes the current `develop` application line: one Angular
client, one FastAPI backend, the canonical `data/adsmod.json` configuration,
and optional ML capabilities loaded in-process.

## Evidence contract

Use these distinctions for every slice:

- **Feature exists** means source or configuration presence only.
- **Exercised** means the named scenario actually ran.
- **Status** is `PASS`, `PARTIAL`, `FAIL`, `BLOCKED`, or `UNTESTED`.
- Evidence is revision-scoped. Historical evidence does not upgrade a changed
  revision by itself.
- A browser claim requires the visible user-facing workflow. API calls, DOM or
  accessibility-tree inspection, tests, and logs are supporting evidence, not
  substitutes for rendered interaction.
- A provider or ML claim must state whether it is local, external, hardware-
  dependent, synthetic, bounded, or representative.
- A blocker names the external dependency or environment cause; it must not hide
  an observed product failure.

Prefer source tests, generated contracts, and hosted CI links as evidence
anchors. Retain a screenshot, provider response, or raw run artifact only when
it supports a current claim that cannot be reproduced from the repository.

## Ordered campaign

| Tier | Stable slices | Focus | Exit gate |
| --- | --- | --- | --- |
| Tier 0 | `ADS-T0-01`–`ADS-T0-02` | CI execution and Windows launcher lifecycle | Hosted jobs complete; launcher readiness, ownership-safe shutdown, and occupied-port protection pass |
| Tier 1 | `ADS-T1-01`–`ADS-T1-03` | Core API, persistence, shell navigation, and recovery | Base runtime and persistence are current-revision validated |
| Tier 2 | `ADS-T2-01`–`ADS-T2-06` | Dataset import and fitting workflows | Core workflows pass with disposable data |
| Tier 3 | `ADS-T3-01`–`ADS-T3-05` | Local public data and provider boundaries | Provider claims have explicit positive or externally bounded evidence |
| Tier 4 | `ADS-T4-01`–`ADS-T4-04` | ML profile, training data, training, and checkpoints | Positive ML lifecycle is demonstrated on the ML profile |
| Tier 5 | `ADS-T5-01`–`ADS-T5-05` | Recovery, repetition, UI/accessibility, performance, and contracts | Evidence and ledger reconcile against the tested source |

The dependency order is:

`T0-01 → T0-02 → T1-01 → T1-02 → T1-03 → T2-01 → T2-02 → T2-03 → T2-04 → T2-05 → T2-06 → T3-01 → T3-02 → T3-03 → T3-04/T3-05 → T4-01 → T4-02 → T4-03 → T4-04 → T5-01 → T5-02 → T5-03 → T5-04 → T5-05`

Stop at the smallest reproducible defect, fix only the affected path, rerun that
slice and its adjacent regression, then update the ledger.

## Stable slice catalog

| Slice | Ontology concept | Scope |
| --- | --- | --- |
| `ADS-T0-01` | `quality.validation-gates` | Hosted workflow configuration and Base/ML/frontend jobs |
| `ADS-T0-02` | `runtime.startup.windows` | Launcher setup, readiness, browser load, conflict refusal, stop, relaunch, and cleanup |
| `ADS-T1-01` | `backend.core.api` | Health, capabilities, core routes, Base ML gating, and fitting configuration |
| `ADS-T1-02` | `persistence.database-migrations` | SQLite/Alembic startup states, locks, and restart persistence |
| `ADS-T1-03` | `ui.shell.navigation` | Routes, redirects, Help focus behavior, backend status, and unavailable controls |
| `ADS-T2-01` | `data.local-dataset.csv-lifecycle` | CSV preview, mapping, validation, save, inspect, reload, and delete |
| `ADS-T2-02` | `data.local-dataset.excel-import` | Real `.xls` and `.xlsx` parsing through the browser path |
| `ADS-T2-03` | `data.local-dataset.validation` | Empty, corrupt, malformed, unsupported, and mismatched input boundaries |
| `ADS-T2-04` | `workflow.fitting` | Dataset selection, model cards, options, and parameter forms |
| `ADS-T2-05` | `workflow.fitting` | Asynchronous fitting, metrics, persistence, and reload |
| `ADS-T2-06` | `workflow.fitting` | Cancellation, duplicate prevention, recovery, and follow-up execution |
| `ADS-T3-01` | `data.public.local-browsing` | Persisted public-data lists, filters, pagination, and provenance |
| `ADS-T3-02` | `data.public.nist` | NIST status, reachability, index, fetch, and job lifecycle |
| `ADS-T3-03` | `data.public.nist` | NIST enrichment, normalization, and unsupported-unit reporting |
| `ADS-T3-04` | `data.public.pubchem-enrichment` | PubChem identity, properties, structures, provenance, and secondary failures |
| `ADS-T3-05` | `data.public.cod-import` | COD search, CIF import, normalized persistence, linking, and re-import |
| `ADS-T4-01` | `runtime.ml-capabilities` | ML profile, one-backend topology, capability discovery, and Base gating |
| `ADS-T4-02` | `workflow.ml-training` | Processed dataset and immutable snapshot creation |
| `ADS-T4-03` | `workflow.ml-training` | Real training run, status/metrics, and cancellation |
| `ADS-T4-04` | `workflow.ml-training` | Checkpoints, compatibility, resume, deletion, and training dashboard |
| `ADS-T5-01` | `backend.core.api` and `data.public.*` | Provider/network/backend failures, retry contracts, and recovery |
| `ADS-T5-02` | `runtime.startup.windows` and `persistence.database-migrations` | Repetition, restart integrity, stale state, duplicate jobs, and cleanup |
| `ADS-T5-03` | `ui.responsive-accessibility` | Keyboard/focus behavior, responsive layouts, overflow, and rendered states |
| `ADS-T5-04` | `quality.validation-gates` | Bounded public-data queries, imports, polling, and measured fixtures |
| `ADS-T5-05` | `quality.validation-gates` | Documentation, generated contracts, stale references, and ledger reconciliation |

## Retention and ownership

- Put current guarantees, decisions, limitations, and next actions in the
  ledger or the relevant architecture, runtime, operations, or UI document.
- Put reusable implementation checks in `app/tests` or the relevant package
  test suite, not in a dated QA report.
- Use `assets/QA/validation/<slice>/` only for durable evidence that is not
  represented by source tests, generated contracts, or hosted CI.
- Do not commit duplicate screenshots, generated contract copies, raw logs, or
  per-run narrative files when the underlying source and a concise ledger entry
  are sufficient.
- Never store credentials, tokens, or machine-specific secrets in evidence.

## Revalidation map

| Change area | Minimum revalidation |
| --- | --- |
| Launcher, ports, startup, cache roots, or data-directory selection | `runtime.startup.windows` plus readiness, rendered load, stop, and final listener checks |
| Routes, service boundaries, API contracts, schema, or migrations | `backend.core.api`, `persistence.database-migrations`, affected workflow, and generated contracts |
| Import parser, units, mapping, or inspection UI | `data.local-dataset.csv-lifecycle`, `data.local-dataset.excel-import`, persistence reload, and fitting selection |
| NIST, PubChem, or COD adapters | The affected provider slice, including provenance and explicit external-failure behavior |
| ML dependencies, capability discovery, training data, models, or checkpoints | `runtime.ml-capabilities` and the full positive `workflow.ml-training` lifecycle |
| Angular routes, shared styles, responsive layout, or accessibility states | `ui.shell.navigation`, affected component tests, and rendered/browser viewport checks |
| Documentation, generated snapshots, or validation organization | `ADS-T5-05`, link/reference scan, and ledger reconciliation |
