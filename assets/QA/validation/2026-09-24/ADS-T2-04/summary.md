# ADS-T2-04 — Fitting configuration

- **Date:** 2026-09-24
- **Result:** PASS
- **Environment:** Windows, official `start_on_windows.ps1` launcher, isolated temporary data root, in-app browser at `/fitting`
- **Code baseline:** working tree based on `9fde82b` (`develop`)

## Scope and evidence

Validated saved dataset and experiment selection, fitting configuration, and all
nine model cards in the rendered fitting page. The cards exposed the expected
parameter fields for Langmuir, Sips, Freundlich, Temkin, Toth,
Dubinin–Radushkevich, Dual-Site Langmuir, Redlich–Peterson, and Jovanovic.
Inspected the dataset/experiment controls and parameter forms in the live UI.
The fitting request now carries the selected parameter bounds and a clamped
initial value for each configured model.

With a saved adsorption fixture selected, a Langmuir fit using deliberately
wide parameter bounds completed. The result was RMSE `0.209493`, R² `0.933318`,
and AICc `4.36452`. An earlier attempt with an intentionally narrow range
completed with poor metrics; widening the range resolved that fixture-specific
constraint. This is evidence for the configuration and execution path, not a
claim that all models or datasets produce a high-quality fit.

The browser validation also found a reload gap: completed results were not
restored in a fresh page. The result reference and restore path were completed
under ADS-T2-05, and the live reload proof is recorded there.

## Automated checks

- `npm run test:unit` — 26 files, 72 tests passed.
- `npm run lint` — passed; output: [`frontend-lint.log`](frontend-lint.log).
- `npm run build -- --verbose` — passed; output: [`frontend-build.log`](frontend-build.log).
- Focused backend suite — 20 passed (job lifecycle, fitting metrics,
  database initialization, and restart persistence); see
  [`ADS-T2-06/backend-focused.log`](../ADS-T2-06/backend-focused.log).
- Fitting API E2E — 5 passed with the official server active; see
  [`ADS-T2-06/fitting-api-e2e.log`](../ADS-T2-06/fitting-api-e2e.log).
- Frontend unit output: [`ADS-T2-05/frontend-unit.log`](../ADS-T2-05/frontend-unit.log).
- Ruff check and format check on changed Python files — passed.

## Limitations

No remaining T2-04 gate. Fit quality remains dataset- and parameter-dependent;
the measured values above are for one selected Langmuir configuration.
