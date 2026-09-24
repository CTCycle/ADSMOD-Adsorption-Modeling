# ADS-T3-01 — Persisted Public Data browsing

Date: 2026-09-24
Baseline: `396e2e092d810ef5182713a563f1de69eb0b5eca`
Status: `PASS` for the stated local-browsing scope

## Scope and result

Validated the current Public Data browser against a disposable SQLite backup of
the locally persisted database. The rendered adsorption, materials, chemicals,
structures, source, filter, pagination, and record-provenance states are listed
in [`browser-state.md`](browser-state.md). The visible layout was inspected at
1280×720. The browser evidence is a written rendered-state record; no screenshot
file or browser-console capture was retained.

The source database SHA-256 was
`5446E0BAB31479D86D7A4C80EA5D2349008C06B13F0CC70E97FA7284F65E7722` before
the backup and was unchanged after the session. All fetches in the adjacent
NIST slices used the isolated copy, not the user's configured database.

## Observed local records

The backed-up database contained 307 adsorption records, 158 materials, 27
chemicals, and zero imported structures. The browser showed all 307 adsorptions,
filtered NIST adsorption records, material and chemical results, and the empty
structures state. NIST detail view included provider identifier, source, and
source URL. The UI counts and interactions are recorded in
[`browser-state.md`](browser-state.md).

## Runtime and automated evidence

The current frontend production build and the focused public-data tests passed.
The backend ran against an isolated database snapshot with a temporary config
and the built frontend preview; this was not official-launcher evidence. The
clone applied the current `20260924_fitting_cancel` migration. Focused test and
lint commands/results are retained below; the runtime migration and NIST job
output is in [`backend.stderr.log`](backend.stderr.log):

- Backend: `test_nist_repository.py` and `test_public_data.py` — 14 passed; see
  [`backend-tests.log`](backend-tests.log).
- Frontend: `nist-collection-rows.component.spec.ts` — 3 passed; see
  [`frontend-tests.log`](frontend-tests.log).
- Ruff: mapper, public-data schema, and focused tests — all checks passed; see
  [`ruff.log`](ruff.log).
- Frontend lint: Angular migration verification and ESLint passed; see
  [`frontend-lint.log`](frontend-lint.log).
- Frontend production build completed successfully; see
  [`frontend-build.log`](frontend-build.log).

## Cleanup and limits

The isolated database/config and temporary process tree were removed; ports
6045 and 5173 were free after validation. The source database hash remained
unchanged. This pass validates locally persisted browsing and provenance only;
it does not certify provider retrieval, PubChem enrichment, COD import, or
live-training behavior. Browser console telemetry was not captured.
