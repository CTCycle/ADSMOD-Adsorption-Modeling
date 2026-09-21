# ADSMOD validation campaign

Last updated: 2026-09-21

## Purpose and baseline

This document is the durable, repository-local digest of the comprehensive
validation roadmap. It separates source evidence, automated checks, live API
checks, browser workflows, provider access, and hardware-dependent ML runs so
that future analysis can resume at a stable slice instead of treating the
presence of tests as proof of current behavior.

The campaign baseline is `develop` at `57baefed` (full SHA:
`57baefed0b2be9e71cfa4f053226ac45ff2ffb6d`). The first gate is Tier 0 because
the remote CI workflow was failing before creating any jobs, and the Windows
launcher had changed after the last surviving live report.

The canonical current status is [`../project_status_ledger.md`](../project_status_ledger.md).
The first execution evidence is under
[`../../QA/validation/2026-09-21/`](../../QA/validation/2026-09-21/).

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
| Tier 1 | `ADS-T1-01`–`ADS-T1-03` | Core API/capability boundaries, SQLite/Alembic startup, shell navigation and recovery | Base runtime and persistence are current-revision validated |
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
| `ADS-T0-01` | Diagnose the remote workflow, validate the YAML/configuration, repair it surgically, then confirm `base-backend`, `ml-backend`, and `frontend` execute their repository-defined commands. | Final run `35622186997` created and completed all three jobs: base/ML installs and validation, generated contracts, frontend checks, and all seven visual projects passed. |
| `ADS-T0-02` | Use the official Windows launcher for dependency reuse, backend/frontend readiness, browser load, occupied-port refusal, owned-process stop, and clean relaunch. | Official clean launch/browser/stop/relaunch passed; controlled listeners on both 6045 and 5173 were refused safely after the active-listener fallback. |

### Tier 1 — application foundations

| Slice | Scope |
| --- | --- |
| `ADS-T1-01` | Health, system capabilities/configuration, core route availability, base-profile absence of ML routes, capability refresh/retry, and fitting configuration. |
| `ADS-T1-02` | Missing/empty/current/invalid SQLite startup states, Alembic head and lock behavior, and minimal record persistence across restart. |
| `ADS-T1-03` | Implemented top-level routes, redirects, unknown-route recovery, Help focus behavior, backend Offline/Online recovery, and truthful unavailable Docs/Settings controls. |

### Tier 2 — core product workflows

| Slice | Scope |
| --- | --- |
| `ADS-T2-01` | Complete browser CSV import through preview, mapping, validation, save, persisted inspection, experiment switching, reload, and deletion. |
| `ADS-T2-02` | Real `.xlsx` and `.xls` files through the canonical browser/API path. |
| `ADS-T2-03` | Invalid uploads, missing columns, malformed values, mismatched arrays, and other input-boundary failures. |
| `ADS-T2-04` | Dataset/experiment selection, fitting configuration, all nine model cards, and parameter forms. |
| `ADS-T2-05` | Positive asynchronous fitting, metrics/result rendering, persistence, and reload. |
| `ADS-T2-06` | Fitting cancellation, duplicate-job prevention, recovery, and clean follow-up execution. |

### Tier 3 — feature families and providers

| Slice | Scope |
| --- | --- |
| `ADS-T3-01` | Locally persisted Public Data browsing, filtering, pagination, and provenance without requiring new retrieval. |
| `ADS-T3-02` | NIST status, index, fetch, and job lifecycle without chemical enrichment. |
| `ADS-T3-03` | NIST guest/host enrichment, unsupported-unit handling, skip counts, and normalization. |
| `ADS-T3-04` | Positive PubChem resolution and normalized persistence. |
| `ADS-T3-05` | Positive COD search, CIF import, normalized persistence, linking, and re-import; no 3D viewer claim. |

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
assets/QA/validation/2026-09-21/
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

Tier 0 is green for the validated baseline below. `ADS-T0-01` is `PASS`:
final run `35622186997` completed all three remote jobs, including backend
profiles, generated contracts, frontend checks, and all seven visual projects.
`ADS-T0-02` is `PASS`: the official launcher now refuses controlled listeners
on both 6045 and 5173 without terminating them, and the clean lifecycle
regression passed around the fix. Tier 0 is closed; Tier 1 and later slices
remain `UNTESTED` and are the next campaign work.
