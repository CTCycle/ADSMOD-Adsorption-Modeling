# ADS-T1-03 browser rendered-state record

Date: 2026-09-23
Runtime: official Windows launcher, Base profile
Browser: Codex in-app Browser
Viewport: 569 × 654 CSS pixels

## Route and navigation states

| Action | Rendered result |
| --- | --- |
| Open `/` | Redirected to `/datasets`; heading `Custom Datasets`. |
| Select Public Data | `/public-data/overview`; heading `Public Data`; workspace visible. |
| Open `/public-data` directly | Redirected to `/public-data/overview`. |
| Select Dashboards | `/dashboards`; heading `Dashboards`; existing placeholder remains visible. |
| Select Fitting | `/fitting`; heading `Fitting`; page content visible. |
| Open `/training` in Base profile | Redirected to `/datasets`; Training link absent. |
| Open an unknown route | Redirected to `/datasets`; heading `Custom Datasets`. |

## Help and unavailable controls

- Help opened on Dashboards with focus on `Close help`. `Shift+Tab` wrapped to
  `Done`; `Tab` wrapped to `Close help`. Escape, Done, and backdrop click each
  closed the dialog and returned focus to Help.
- Docs and Settings remained visible at this compact viewport. Both were native
  disabled buttons with accessible unavailable labels and explanatory titles;
  the disabled style rendered at 0.5 opacity.
- At 569 × 654, document and body widths were both 569 pixels; no horizontal
  overflow was observed.
- The rendered Dashboards shell and Help dialog were visually inspected in the
  in-app Browser. The Browser screenshot API returned in-memory captures only,
  so this file preserves the route, viewport, visible content, and interaction
  results as the durable rendered-state record.

## Backend and browser diagnostics

- The visible shell status was `Backend Online`.
- Proxied `/health/ready` and `/api/v1/system/capabilities` each returned HTTP
  200. The Base capability response reported `machine_learning: false`,
  `training: false`, and `checkpoints: false`; see `http-checks.json`.
- In-app Browser console error and warning logs were empty after the route and
  Help interactions.
- The frontend component test exercised Online → Offline → Online recovery,
  including Training navigation hidden in Base and shown in the ML profile.

No dataset or persistent application data was modified during the browser pass.
