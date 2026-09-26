# ADS-T5-01 — failure and recovery validation

Date: 2026-09-26
Validation base: `bcc81e824f62fbbc13a8059a8b989b9ce31f8d25` plus the scoped
validation tests in this campaign
Status: `PARTIAL`

## Scope

This slice covers temporary provider/network failures, backend offline/online
recovery, retry behavior, and truthful API/UI error states. It does not claim
continuous availability of external services.

## Evidence

- `app/tests/unit/test_public_data.py` passed the transient `503 → 200` retry
  regression with the expected `0.5` second backoff, and explicit provider
  errors mapped to HTTP `429` and `503`.
- The focused backend regression command passed `42` tests with one intentional
  ML-positive deselection. The only warning was the pre-existing pytest warning
  that `cache_dir` is not recognized by the installed pytest version.
- Frontend unit tests passed: `26` files and `72` tests.
- The isolated live backend served `/health/ready` as `ready`. With
  `check_health=true`, COD, NIST, and PubChem each returned `unavailable` with
  provider-specific detail; the full response is in
  [`provider-health.json`](provider-health.json).
- A live PubChem resolve for `carbon dioxide` returned HTTP `503` with
  `PubChem could not be reached after 3 attempts.`
- The rendered in-app Browser showed the three provider cards as
  `UNAVAILABLE` while the local backend status remained `Online`.
- Stopping the isolated backend and reloading the rendered shell showed
  `HTTP error 502`, a `Retry` control, and `Backend Offline`. After
  restarting the same backend profile and activating `Retry`, the page
  returned to a clean rendered state with `Backend Online`. The observed route
  and states are recorded in [`browser-state.md`](browser-state.md).

## Remaining limitation

This slice remains `PARTIAL` for two environment boundaries. The current host
could not obtain a positive live provider response during this run, so
provider retry-to-success was validated deterministically with a fake HTTP
client but not against a live upstream. Also, the official launcher build
attempt exited with Windows status `-1073741819` (access violation) before the
frontend listener was created on both the repository Node runtime and the
system Node runtime. The unchanged existing `app/client/dist/browser` bundle
was served through the repository preview server solely to exercise the live
backend/UI recovery path; this run makes no new official-launcher pass claim.

No product error-contract defect was found. Reopen this slice for a full pass
when the provider/network boundary is reachable and the launcher build can
complete on the validation host.
