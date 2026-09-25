# ADS-T5-03 responsive and accessibility validation

Date: 2026-09-25
Validated revision: `503dc10` (final commit)
Scope: Shared Angular shell, primary route containment, narrow responsive layout, active navigation semantics, dataset rename/delete interactions, and confirmation-dialog keyboard focus behavior.

## Result

`PASS` for the exercised ADS-T5-03 scope. The historical 1280x720 Sources-page overflow did not reproduce after the current Angular UI changes. The tested routes remained contained at every configured viewport, and the dataset confirmation dialog retained focus, wrapped Tab focus, closed on Escape, and restored focus to the initiating Delete button.

## Current implementation evidence

- Official `start_on_windows.ps1` launch succeeded with the current checkout: backend `127.0.0.1:6045`, frontend `127.0.0.1:5173`.
- In-app browser at 1280x720 measured document/body width 1280px, header height 112px, status-bar height 34px, aligned main/header origins, and no horizontal expansion on `/datasets`, `/public-data/overview`, `/dashboards`, or `/fitting`. The Base-profile `/training/processing` route redirected to `/datasets`.
- The in-app rendered Public Data screenshot is retained as [`sources-shell-1280x720.png`](sources-shell-1280x720.png). Additional desktop and mobile captures are [`public-data-1440x920.png`](public-data-1440x920.png) and [`datasets-600x900.png`](datasets-600x900.png).

## Automated evidence

- `npm run lint` — passed, including Angular migration verification.
- `npm run test:unit -- --no-progress` — 26 files / 72 tests passed.
- `npm run test:preview` — 3 preview-server tests passed.
- `npm run visual:compare` — 32 tests passed across 1440x920, 1480x920, 1360x900, the historical 1280x720 regression viewport, 1200x900, 900x900, 768x900, and 600x900.
- The visual suite covers primary-route shell geometry, `aria-current` navigation, Base-profile Training gating, contained Public Data dense views, in-page rename, explicit delete confirmation, Tab focus cycling, Escape dismissal, and focus restoration.
- Angular CLI production build passed with `npm run build -- --verbose` and with the direct equivalent `node node_modules/@angular/cli/bin/ng.js build --progress=false`. The plain non-TTY `npm run build` wrapper exited silently with code 1 on this host; no compiler error reproduced in the equivalent CLI build.

## Remaining limits

- This is a bounded rendered/browser and keyboard pass, not a full screen-reader audit of every secondary dialog, table, or provider workflow.
- `ADS-T5-01`, `ADS-T5-02`, `ADS-T5-04`, and `ADS-T5-05` remain untested.
- `workflow.ml-training` remains `BLOCKED` by the current Base environment without Torch/RDKit; `ui.dashboards` remains `PARTIAL` because the top-level view is still a placeholder and positive training metrics are unavailable.
