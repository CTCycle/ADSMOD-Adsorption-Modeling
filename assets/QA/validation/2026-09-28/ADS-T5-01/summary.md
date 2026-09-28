# ADS-T5-01 — current-host provider and recovery recheck

Date: 2026-09-28
Validated implementation SHA: `2f52eb0fcfbe955418b6b5071d1b5811226d2c2c`
Status: `PASS` for the bounded slice

## Scope

This slice covers temporary provider/network failures, backend offline/online
recovery, retry/error contracts, truthful provider state, and the official
launcher build path. It does not claim continuous availability of external
services or a long-duration stress campaign.

## Current validation

- The non-ML provider and architecture regression subset passed: `21 passed`.
  It includes the transient `503 -> 200` retry contract, explicit `429`/`503`
  mapping, client cleanup, and the current one-backend dependency-boundary
  checks.
- An isolated backend using the current Base environment became ready with
  HTTP 200. `check_health=true` reported COD, NIST, and PubChem as
  `available`; the durable response is in [`provider-health.json`](provider-health.json).
- A live PubChem resolve for `carbon dioxide` returned `Carbon Dioxide`,
  formula `CO2`, and a persisted local record. A live COD lookup for
  `1011195` returned HTTP 200 with one coordinate-bearing Zinc sulfide result.
- The official launcher menu option 5 (`Rebuild frontend`) completed using
  portable Node.js `22.13.0`, created the current build-state fingerprint, and
  returned to the menu without starting application listeners.
- The prior rendered in-app Browser recovery evidence remains applicable to
  the unchanged backend/frontend runtime source: provider cards rendered
  `UNAVAILABLE` without claiming success, and backend Offline/Online retry
  recovery passed in [`2026-09-26/ADS-T5-01/browser-state.md`](../../2026-09-26/ADS-T5-01/browser-state.md).

## Cleanup and limitation

The isolated backend was stopped and ports 6045 and 5173 were free. No product
error-contract defect was found. External provider availability is an
observation of this validation window, not a continuous-service guarantee;
long-duration load/stress remains outside this slice and is tracked with
`ADS-T5-02`.
