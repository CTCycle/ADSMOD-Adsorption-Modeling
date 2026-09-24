# ADS-T2-05 — Fitting result, persistence, and reload

- **Date:** 2026-09-24
- **Result:** PASS
- **Environment:** Windows, official `start_on_windows.ps1` launcher, isolated temporary data root, in-app browser at `/fitting`
- **Code baseline:** working tree based on `9fde82b` (`develop`)

Rendered state record: [`browser-state.md`](browser-state.md), captured from
the in-app browser at `/fitting` in a `554 × 637` viewport.

## Scope and evidence

Ran an asynchronous fit against a persisted 14-observation dataset and
experiment. The completed API record contained one model result and persisted
metrics. After reloading `/fitting`, the page displayed an informational
restore notice and rendered the dataset, experiment, best model, observation
count, and RMSE/R²/AICc result row from the persisted run.

The browser run uncovered that completed results were not restored after a page
reload. The client now retains the last successful run ID and display context,
fetches the persisted run on page construction, restores the result summary,
and clears stale context after a dataset/experiment change, reset, or new run.
Transient lookup errors preserve the reference; a missing run clears it. A
store unit test covers restoration, and the rendered post-reload result was
confirmed in the live browser.

The restored follow-up run displayed RMSE `2.49285`, R² `-8.44186`, and AICc
`73.7062`. Those values are retained accurately as lifecycle evidence and are
not a positive scientific-quality claim. The separately configured run in
[`ADS-T2-04`](../ADS-T2-04/summary.md) produced the higher-quality metrics
recorded there.

## Automated checks

- `npm run test:unit` — 26 files, 72 tests passed, including persisted-result
  restoration coverage; output: [`frontend-unit.log`](frontend-unit.log).
- `npm run lint` and `npm run build -- --verbose` — passed; logs are linked in
  [`ADS-T2-04`](../ADS-T2-04/summary.md).
- Fitting API E2E — 5 passed with the official server active; see
  [`ADS-T2-06/fitting-api-e2e.log`](../ADS-T2-06/fitting-api-e2e.log).

The visible reload state is documented in [`browser-state.md`](browser-state.md);
a durable browser console dump was not captured.

## Limitations

No remaining T2-05 gate. The tested reload contract restores the last
successful fitting run for the current browser profile; clearing that browser's
local storage removes the run reference.
