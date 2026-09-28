# ISSUE-005 rendered browser state

## Environment

- Application revision: `0dfdfc4cca5167e0bdfbd5566ec37f609b01cf77` (`develop`)
- Runtime: official `start_on_windows.ps1` launcher; isolated config selected with `-DataPath` pointing to `G:\Projects\Repositories\Active projects\ADSMOD Adsorption Modeling\assets\QA\validation\2026-09-28\ISSUE-005\session\config`
- Backend readiness: HTTP 200, `{"service":"backend","version":"3.0.0","state":"ready","details":{}}`
- Frontend route response: `GET /dashboards` returned HTTP 200, `text/html; charset=utf-8`
- Browser viewport: 1280×720
- Data: empty isolated database under the issue QA session storage; the launcher process command line referenced the copied config and the resolved database existed there.

## Rendered `/dashboards`

The visible shell had the ADSMOD identity, primary navigation, active `Dashboards` item, and `Backend Online` status. The page heading was `Dashboards` with subtitle `Monitor workspace activity and results.` The main card showed `No dashboards yet` and `Workspace dashboard views will appear here as activity and results become available.`

The card exposed two links. `Open Custom Datasets` navigated to `/datasets`, which rendered the empty workspace state (`Add your first dataset`; no datasets imported). `Explore Public Data` navigated to `/public-data/overview`, which rendered its workspace and zero-record local counts. The dashboard route was then restored through primary navigation and rendered again with the same placeholder and online backend state.

The in-app Browser screenshot from the final dashboard state was visually inspected during this run. This written state record preserves the route, viewport, visible labels, status, and exercised navigation. Console logs were not available through this Browser surface and were not assessed.
