# ADS-T5-03 — residual responsive, keyboard, and accessibility recheck

Date: 2026-09-28
Tested source/evidence revision: `dc841b4f251eff3f868385cf52d3d16a0b05aea4`.
The subsequent documentation publication commit only records this tested SHA.
Status: `PASS` for the bounded rendered/keyboard/responsive scope; audible
screen-reader coverage remains `PARTIAL`.

## Current implementation

The current revision now gives the dataset file input the accessible name
`Choose a dataset file`. The import wizard focuses its close control when
opened, keeps Tab/Shift+Tab inside the dialog, closes on Escape, and restores
focus to `Add dataset`. The focused regression is in
[`application-shell.spec.ts`](../../../../app/client/tests/visual/application-shell.spec.ts).

## Rendered evidence

The Codex in-app Browser rendered the shell and route states at `1280x720`:

- `/datasets`: Custom Datasets shell, Add dataset control, labelled file input,
  Backend Online;
- `/public-data/overview`: Overview tabs, normalized workspace counts, provider
  coverage, Backend Online;
- `/public-data/adsorption`: dense filter/table view with no horizontal
  expansion;
- `/fitting`: configuration controls, nine model cards, fitting log, Backend
  Online;
- `/dashboards`: contained product-scope placeholder, `No dashboards yet`,
  both navigation actions, Backend Online.

Durable route captures are [`datasets-1280x720.png`](datasets-1280x720.png),
[`public-data-overview-1280x720.png`](public-data-overview-1280x720.png),
[`public-data-adsorption-1280x720.png`](public-data-adsorption-1280x720.png),
[`fitting-1280x720.png`](fitting-1280x720.png), and
[`dashboards-1280x720.png`](dashboards-1280x720.png). The actual `600x900`
responsive captures are [`datasets-600x900.png`](datasets-600x900.png) and
[`training-600x900.png`](training-600x900.png). The in-app Browser surface did
not expose a viewport override, so the 600x900 rendered proof uses the
repository Playwright/Chromium screenshot path as supporting automation; it is
not described as an in-app Browser 600x900 capture.

## Keyboard and dialog evidence

The direct Playwright browser scenario uploaded the repository CSV fixture and
observed: `Close import wizard` focused on open; Tab moved to `Review mapping`;
Shift+Tab wrapped to `Close import wizard`; Escape removed the dialog and
restored focus to `Add dataset`. The import screenshot is
[`import-1280x720.png`](import-1280x720.png). The Codex in-app Browser Help
dialog likewise exposed its heading and controls, wrapped keyboard focus, and
returned focus to the Help trigger after Escape. Shell navigation, active-page
semantics, confirmation behavior, and dense Public Data containment were also
observed in the rendered route checks.

## Automated checks

- Frontend lint and Angular migration verification: passed.
- Frontend unit suite: `26` files / `72` tests passed.
- Preview-server tests: `3` passed.
- Development Angular build used by the rendered server: passed.
- Direct route screenshot checks at `1280x720` and `600x900`: passed and saved
  above.
- The configured visual test runner was also attempted with writable output;
  its route screenshots were produced, but the runner hung during browser
  teardown on this host. The earlier focused visual regression passed before
  the final lint-only cleanup; the direct Playwright focus scenario above was
  rerun after that cleanup and passed.

The production `npm run build` path repeatedly terminated inside the bundled
Angular native builder with Windows `0xC0000005` and no compiler diagnostic,
including a fresh output path and development configuration. This is recorded
as a host/runtime build limitation; the source compiled in the Angular unit
build and development server, and lint/type/template checks passed.

## Accessibility and remaining limits

Windows Narrator was started for this recheck, but no observable speech or
Speech Recap output was available. DOM/AX roles and visible focus behavior are
therefore supporting evidence only; this run does not claim audible
screen-reader PASS. Full product-wide keyboard/screen-reader review remains
`PARTIAL`, and this bounded viewport pass is not production-scale stress.

Console limitation: the in-app Browser tab exposed no error-level console
entries for the observed route checks. Its API did not export a durable network
trace; HTTP/API outcomes are covered by the adjacent E2E and soak evidence.
